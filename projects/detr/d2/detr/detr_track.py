# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import math
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss
from models.backbone import Joiner
from models.detrtrack import DETRTRACK, SetCriterion
from models.matcher import HungarianMatcher, HungarianMatcherTrack
from models.position_encoding import PositionEmbeddingSine
from models.transformer import Transformer
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from util.misc import NestedTensor
import copy

__all__ = ["DetrTrack"]


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[cfg.MODEL.DETR.INDEX_FEEDFORWARD]].channels

    def forward(self, images):
        cur_images = images.tensor
        cur_dim = cur_images.shape[1]
        if cur_dim == 3:
            pre_images = cur_images.clone()
        elif cur_dim == 6:
            pre_images = cur_images[:, 3:, :, :]
            cur_images = cur_images[:, :3, :, :]
        else:
            raise NotImplementedError

        features = self.backbone(cur_images, pre_images)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks


@META_ARCH_REGISTRY.register()
class DetrTrack(nn.Module):
    """
    Implement Detr
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.DETR.NUM_CLASSES
        hidden_dim = cfg.MODEL.DETR.HIDDEN_DIM
        num_queries = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        # Transformer parameters:
        nheads = cfg.MODEL.DETR.NHEADS
        dropout = cfg.MODEL.DETR.DROPOUT
        dim_feedforward = cfg.MODEL.DETR.DIM_FEEDFORWARD
        enc_layers = cfg.MODEL.DETR.ENC_LAYERS
        dec_layers = cfg.MODEL.DETR.DEC_LAYERS
        pre_norm = cfg.MODEL.DETR.PRE_NORM

        # Loss parameters:
        giou_weight = cfg.MODEL.DETR.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DETR.L1_WEIGHT
        track_weight = cfg.MODEL.DETR.TRACK_WEIGHT
        deep_supervision = cfg.MODEL.DETR.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DETR.NO_OBJECT_WEIGHT

        # Track parameters.
        self.track_on = cfg.MODEL.DETR.TRACK_ON
        self.track_aug = cfg.MODEL.DETR.TRACK_AUG

        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        backbone = Joiner(d2_backbone, PositionEmbeddingSine(N_steps, normalize=True))
        backbone.num_channels = d2_backbone.num_channels

        transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            return_intermediate_dec=deep_supervision,
        )

        self.detr = DETRTRACK(
            backbone, transformer, num_classes=self.num_classes, num_queries=num_queries, aux_loss=deep_supervision,
            cfg=cfg
        )
        self.detr.to(self.device)

        # building criterion
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        track_matcher = HungarianMatcherTrack(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        weight_dict["loss_track"] = track_weight
        if deep_supervision:
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, track_matcher=track_matcher,
            weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses
        )
        self.criterion.to(self.device)

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        if self.training:
            # prepare images & gt.
            images = self.preprocess_image(batched_inputs)
            if isinstance(batched_inputs[0], tuple):
                gt_instances = list()
                for paired_inputs in batched_inputs:
                    paired_instances = [x["instances"].to(self.device) for x in paired_inputs]
                    gt_instances.append(paired_instances)
            else:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            # detection first.
            output = self.detr(copy.deepcopy(images))
            output, pre_embed = output
            if self.track_on:
                # generate targets for tracking.
                targets = self.prepare_targets_for_tracking(gt_instances)  # cur_targets / pre_targets.
                # compute loss for detection (pre frame) & generate indices.
                loss_det, indices_det = self.criterion(output, targets[1], track_on=False)

                # track.
                if self.track_aug:
                    track_embed = list()
                    track_indices = list()
                    for i, indices in enumerate(indices_det):
                        embedding = pre_embed[i]
                        indices_pos = indices[0]
                        size_embedding = len(embedding)
                        size_pos = len(indices_pos)
                        indices_cand = torch.ones(size_embedding)
                        indices_cand[indices_pos] = 0
                        indices_neg = indices_cand.nonzero().squeeze(1)
                        assert len(indices_pos) <= len(indices_neg)
                        indices_select = torch.randperm(len(indices_neg))[:size_pos]
                        indices_neg = indices_neg[indices_select]
                        # make a copy/aug.
                        embedding[indices_neg, :] = embedding[indices_pos, :]
                        track_embed.append(embedding.unsqueeze(0))
                        track_indices.append(tuple(list(indices) + [indices_neg]))  # pos / gt / neg.
                    track_embed = torch.cat(track_embed, 0).permute(1, 0, 2)
                else:
                    track_embed = pre_embed.permute(1, 0, 2)
                    track_indices = indices_det

                output = self.detr(images, pre_embed=track_embed)
                output, _ = output
                # compute loss for tracking (cur frame).
                loss_track, _ = self.criterion(output, targets, indices_track=track_indices, track_on=True)
                # aggregate losses.
                losses_dict = dict()
                weight_dict = self.criterion.weight_dict
                for loss_dict in [loss_det, loss_track]:
                    for k in loss_dict.keys():
                        if k in weight_dict:
                            if k not in losses_dict.keys():
                                losses_dict[k] = loss_dict[k] * weight_dict[k]
                            else:
                                losses_dict[k] = (losses_dict[k] + loss_dict[k] * weight_dict[k]) / 2
                return losses_dict
            else:
                raise NotImplementedError
                targets = self.prepare_targets(gt_instances)
                loss_dict = self.criterion(output, targets)
                weight_dict = self.criterion.weight_dict
                for k in loss_dict.keys():
                    if k in weight_dict:
                        loss_dict[k] *= weight_dict[k]
                return loss_dict
        else:
            assert len(batched_inputs) == 1, \
                print("Only support ONE image each time during inference, "
                      "while there are {} images now.".format(len(batched_inputs)))
            # prepare images.
            images = self.preprocess_image(batched_inputs)
            if isinstance(batched_inputs[0], tuple):
                cur_input = batched_inputs[0][0]
                pre_embed = cur_input.get("pre_embed", None)
            else:
                raise NotImplementedError

            # inference.
            output = self.detr(images, pre_embed=pre_embed)
            output, pre_embed = output
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.track_on:
                box_track = output["pred_tracks"]
                results = self.inference(box_cls, box_pred, images.image_sizes, box_track=box_track)
            else:
                raise NotImplementedError
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs[0], images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results, pre_embed.permute(1, 0, 2)

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets

    def prepare_targets_for_tracking(self, targets):
        cur_targets = []
        pre_targets = []
        for paired_targets in targets:
            for i, targets_per_image in enumerate(paired_targets):
                h, w = targets_per_image.image_size
                image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
                gt_classes = targets_per_image.gt_classes
                gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
                gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
                gt_tracks = targets_per_image.gt_tracks
                if i == 0:
                    cur_targets.append({"labels": gt_classes, "boxes": gt_boxes, "tracks": gt_tracks})
                elif i == 1:
                    pre_targets.append({"labels": gt_classes, "boxes": gt_boxes, "tracks": gt_tracks})
                else:
                    raise NotImplementedError
        return cur_targets, pre_targets

    def inference(self, box_cls, box_pred, image_sizes, **kwargs):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        batchSize = len(image_sizes)
        results = []
        if self.track_on:
            if kwargs is not None:
                box_track = kwargs["box_track"].sigmoid()
            else:
                raise NotImplementedError

        # For each box we assign the best class or the second best if the best on is `no_object`.
        scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for index, scores_per_image, labels_per_image, box_pred_per_image, image_size in zip(
            range(batchSize), scores, labels, box_pred, image_sizes
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])

            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            if self.track_on:
                result.pred_tracks = box_track[index].squeeze(1)
            results.append(result)
        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        if isinstance(batched_inputs[0], tuple):
            images = list()
            for paired_inputs in batched_inputs:
                paired_images = [self.normalizer(x["image"].to(self.device)) for x in paired_inputs]
                images.append(torch.cat(paired_images, dim=0))  # cur:pre.
        else:
            images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]

        images = ImageList.from_tensors(images)
        return images
