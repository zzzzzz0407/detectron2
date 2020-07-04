# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
from typing import List
import torch
import torch.nn.functional as F
from torch import nn
import fvcore.nn.weight_init as weight_init

from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.structures import Boxes, ImageList, Instances

from projects.SOD.sod.coordconv import CoordConv
from projects.SOD.sod.box_ops import box_xyxy_to_cxcywh
from projects.SOD.sod.matcher import HungarianMatcher
from projects.SOD.sod.utils import SetCriterion, PostProcess


__all__ = ["SOD"]

INF = 100000000


class SOD_1(nn.Module):
    """
    SOD model.
    Creates FPN backbone, \
    Instance branch for query, cls and reg, \
    Base branch for class-agnostic masks. \
    Calculates and applies proper losses to class and boxes.
    """

    def __init__(self, cfg):
        super().__init__()

        # get the device of the model.
        self.device = torch.device(cfg.MODEL.DEVICE)

        # fmt: off
        self.instance_in_features = cfg.MODEL.SOD.INSTANCE_IN_FEATURES
        self.num_classes = cfg.MODEL.SOD.NUM_CLASSES  # without background.
        self.use_base = cfg.MODEL.SOD.USE_BASE
        self.center_symmetry = cfg.MODEL.SOD.CENTER_SYMMETRY
        self.scale_variance = cfg.MODEL.SOD.SCALE_VARIANCE
        self.max_per_img = cfg.MODEL.SOD.MAX_PER_IMG
        # fmt: on

        # generate sizes of interest.
        soi = []
        prev_size = -1
        for s in cfg.MODEL.SOD.SIZES_OF_INTEREST:
            soi.append([prev_size, s])
            prev_size = s
        soi.append([prev_size, INF])
        self.sizes_of_interest = soi

        # build the backbone.
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()

        # build the base head.
        if self.use_base:
            base_shapes = [backbone_shape[f] for f in self.instance_in_features]
            self.base_head = SODBaseHead(cfg, base_shapes)

        # build the ins head.
        instance_shapes = [backbone_shape[f] for f in self.instance_in_features]
        self.ins_head = SODInsHead(cfg, instance_shapes)

        # loss function.
        losses = ["labels", "boxes"]

        self.weight_dict = {
            "loss_ce": cfg.MODEL.SOD.CE_LOSS_COEFF,
            "loss_bbox": cfg.MODEL.SOD.BBOX_LOSS_COEFF,
            "loss_giou": cfg.MODEL.SOD.GIOU_LOSS_COEFF,
        }

        matcher = HungarianMatcher(
            soi=self.sizes_of_interest,
            scale_variance=self.scale_variance,
            cost_class=cfg.MODEL.SOD.CE_LOSS_COEFF,
            cost_bbox=cfg.MODEL.SOD.BBOX_LOSS_COEFF,
            cost_giou=cfg.MODEL.SOD.GIOU_LOSS_COEFF,
        )

        self.criterion = SetCriterion(self.num_classes,
                                      matcher=matcher,
                                      weight_dict=self.weight_dict,
                                      eos_coef=cfg.MODEL.SOD.EOS_COEFF,
                                      losses=losses,
                                      )

        self.post_processors = {"bbox": PostProcess()}

        # image transform.
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DetectionTransform` .
                Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:
                image: Tensor, image in (C, H, W) format.
                instances: Instances
                Other information that's included in the original dicts, such as:
                    "height", "width" (int): the output resolution of the model, used in inference.
                        See :meth:`postprocess` for details.
         Returns:
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        images = self.preprocess_image(batched_inputs)

        # backbone.
        features = self.backbone(images.tensor)

        # base head.
        if self.use_base:
            base_features = [features[f] for f in self.instance_in_features]
            base_features = self.base_head(base_features)
        else:
            base_features = None

        # ins head.
        ins_features = [features[f] for f in self.instance_in_features]
        cate_pred, bbox_pred = self.ins_head(ins_features, base_features)
        if self.center_symmetry:
            bbox_pred[:, :, :2] = (bbox_pred[:, :, :2] + 1.) / 2.
        cate_pred = self.permute_to_B_KHW_C(cate_pred, len(self.instance_in_features))
        bbox_pred = self.permute_to_B_KHW_C(bbox_pred, len(self.instance_in_features))
        out = {'pred_logits': cate_pred, 'pred_boxes': bbox_pred}

        if self.training:
            # get_ground_truth.
            targets = self.get_ground_truth(batched_inputs)
            # compute loss.
            loss_dict = self.criterion(out, targets)
            # assign different loss coeff.
            for k, v in loss_dict.items():
                loss_dict[k] = v * self.weight_dict[k] if k in self.weight_dict else v
            return loss_dict
        else:
            # do inference for results.
            results = self.inference(out, batched_inputs)
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, batched_inputs):
        targets = []
        for bi in batched_inputs:
            target = {}
            h, w = bi["image"].shape[-2:]
            boxes = box_xyxy_to_cxcywh(
                bi["instances"].gt_boxes.tensor / torch.tensor([w, h, w, h], dtype=torch.float32))
            target["boxes"] = boxes.to(self.device)
            target["area"] = bi["instances"].gt_boxes.area().to(self.device)
            target["labels"] = bi["instances"].gt_classes.to(self.device)
            if hasattr(bi["instances"], "gt_masks"):
                target["masks"] = bi["instances"].gt_masks
            target["iscrowd"] = torch.zeros_like(target["labels"], device=self.device)
            target["orig_size"] = torch.tensor([bi["height"], bi["width"]], device=self.device)
            target["size"] = torch.tensor([h, w], device=self.device)
            target["image_id"] = torch.tensor(bi["image_id"], device=self.device)
            targets.append(target)
        return targets

    def permute_to_B_KHW_C(self, tensor, K):
        """
        Transpose/reshape a tensor from (BxK, HxW, C) to (B, KxHxW, C)
        """
        assert tensor.dim() == 3, tensor.shape
        N, S, C = tensor.shape
        B = N // K
        tensor = tensor.split(B, dim=0)
        tensor = torch.cat(tensor, dim=1)  # Size=(B,KHW,C)
        return tensor

    def inference(self, outs, images):
        """
        Arguments:
            outs: the predicted classes & boxes.
            images (ImageList): the input images

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        image_sizes = [
            (img["height"], img["width"]) for img in images]
        results = self.post_processors["bbox"](outs, image_sizes)

        # post process.
        processed_results = []
        for result_per_image, image_size in zip(results, image_sizes):
            result = Instances(image_size)
            boxes = result_per_image["boxes"].float()
            scores = result_per_image["scores"].float()
            labels = result_per_image["labels"].long()
            if len(scores) > self.max_per_img:
                sort_inds = torch.argsort(scores, descending=True)
                sort_inds = sort_inds[:self.max_per_img]
                boxes = boxes[sort_inds, :]
                scores = scores[sort_inds]
                labels = labels[sort_inds]
            result.pred_boxes = Boxes(boxes)
            result.scores = scores
            result.pred_classes = labels
            processed_results.append({"instances": result})

        return processed_results


class SODBaseHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOD Base Head.
        """
        super().__init__()
        # fmt: off
        self.instance_in_features = cfg.MODEL.SOD.INSTANCE_IN_FEATURES
        self.num_in_channels = cfg.MODEL.SOD.INSTANCE_IN_CHANNELS  # = fpn.
        self.num_channels = cfg.MODEL.SOD.BASE_CHANNELS
        self.num_conv = cfg.MODEL.SOD.NUM_BASE_CONVS
        self.norm = cfg.MODEL.SOD.NORM
        self.with_coord = cfg.MODEL.SOD.WITH_COORD
        self.num_levels = len(input_shape)
        assert self.num_levels == len(self.instance_in_features), \
            print("Input shape should match the features.")
        # fmt: on

        head_configs = {"base": (self.num_conv,
                                 self.with_coord,
                                 False),  # leave for DCN.
                        }

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == self.num_in_channels, \
            print("In channels should equal to tower in channels!")

        for head in head_configs:
            tower = []
            num_convs, use_coord, use_deformable = head_configs[head]
            for i in range(num_convs):
                # with coord or not.
                if i == 0:
                    if use_coord:
                        chn = self.num_in_channels + 2
                    else:
                        chn = self.num_in_channels
                else:
                    chn = self.num_channels
                # use deformable conv or not.
                if use_deformable and i == num_convs - 1:
                    raise NotImplementedError
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                    chn, self.num_channels,
                    kernel_size=3, stride=1,
                    padding=1, bias=self.norm is None
                ))
                if self.norm == "GN":
                    tower.append(nn.GroupNorm(32, self.num_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        # init.
        for l in self.base_tower:
            if isinstance(l, nn.Conv2d):
                nn.init.normal_(l.weight, std=0.01)
                if l.bias is not None:
                    nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.

        Returns:
            pass
        """
        assert len(features) == self.num_levels, \
            print("The number of input features should be equal to the supposed level.")

        base_feat = features[0]
        if self.with_coord:
            base_feat = CoordConv(base_feat)
        base_feat = self.base_tower(base_feat)
        return base_feat


class SODInsHead(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        SOD Instance Head.
        """
        super().__init__()
        # fmt: off
        self.num_classes = cfg.MODEL.SOD.NUM_CLASSES  # without background.
        self.num_kernels = cfg.MODEL.SOD.NUM_KERNELS
        self.instance_in_features = cfg.MODEL.SOD.INSTANCE_IN_FEATURES
        self.num_in_channels = cfg.MODEL.SOD.INSTANCE_IN_CHANNELS  # = fpn.
        self.num_channels = cfg.MODEL.SOD.INSTANCE_CHANNELS
        self.num_grids = cfg.MODEL.SOD.NUM_GRIDS
        self.strides = cfg.MODEL.SOD.FPN_INSTANCE_STRIDES
        self.num_conv_before = cfg.MODEL.SOD.NUM_INSTANCE_CONVS_BEFORE
        self.num_conv_after = cfg.MODEL.SOD.NUM_INSTANCE_CONVS_AFTER
        self.fc_dim = cfg.MODEL.SOD.FC_DIM
        self.with_coord = cfg.MODEL.SOD.WITH_COORD
        self.type_att = cfg.MODEL.SOD.TYPE_ATTENTION
        self.norm = cfg.MODEL.SOD.NORM
        self.center_symmetry = cfg.MODEL.SOD.CENTER_SYMMETRY
        self.use_base = cfg.MODEL.SOD.USE_BASE  # use dense2sparse or not.
        self.pe_on = cfg.MODEL.SOD.PE_ON  # use positional encoding or not.
        # Convolutions to use in the towers
        self.num_levels = len(self.instance_in_features)
        assert self.num_levels == len(self.strides), \
            print("Strides should match the features.")
        assert len(set(self.num_grids)) == 1, \
            print("The grid among different stages should be same.")
        # fmt: on
        if self.pe_on:
            num_ins = torch.tensor(self.num_grids).pow(2).sum()
            self.ins_embed = nn.Embedding(num_ins, self.num_in_channels)

        head_configs = {"ins_before": (self.num_conv_before,
                                       self.with_coord,
                                       False),  # leave for DCN.
                        "ins_after": (self.num_conv_after,
                                      self.with_coord,
                                      False)
                        }

        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, \
            print("Each level must have the same channel!")
        in_channels = in_channels[0]
        assert in_channels == self.num_in_channels, \
            print("In channels should equal to tower in channels!")

        # shared conv.
        for head in head_configs:
            tower = []
            num_convs, use_coord, use_deformable = head_configs[head]
            for i in range(num_convs):
                # with coord or not.
                if i == 0:
                    if use_coord:
                        chn = self.num_in_channels + 2
                    else:
                        chn = self.num_in_channels
                else:
                    chn = self.num_channels
                # use deformable conv or not.
                if use_deformable and i == num_convs - 1:
                    raise NotImplementedError
                else:
                    conv_func = nn.Conv2d
                tower.append(conv_func(
                        chn, self.num_channels,
                        kernel_size=3, stride=1,
                        padding=1, bias=self.norm is None
                ))
                if self.norm == "GN":
                    tower.append(nn.GroupNorm(32, self.num_channels))
                tower.append(nn.ReLU(inplace=True))
            self.add_module('{}_tower'.format(head),
                            nn.Sequential(*tower))

        # att conv.
        if self.use_base:
            self.base_att = nn.Conv2d(
                self.num_channels, self.num_kernels, kernel_size=1, stride=1, padding=0
            )
            self.ins_att = nn.Conv2d(
                self.num_channels, self.num_kernels, kernel_size=1, stride=1, padding=0
            )

        # individual fc.
        cls_tower = []
        bbox_tower = []
        self._output_size = self.num_channels
        for k, fc_dim in enumerate(self.fc_dim):
            cls_tower.append(nn.Linear(self._output_size, fc_dim))
            cls_tower.append(nn.ReLU(inplace=True))
            bbox_tower.append(nn.Linear(self._output_size, fc_dim))
            bbox_tower.append(nn.ReLU(inplace=True))
            self._output_size = fc_dim
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # pred layer.
        self.cls_pred = nn.Linear(self._output_size, self.num_classes + 1)
        self.bbox_pred = nn.Linear(self._output_size, 4)

        # init.
        conv_modules = [self.ins_before_tower, self.ins_after_tower]
        if self.use_base:
            conv_modules += [self.base_att, self.ins_att]
        for modules in conv_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    if l.bias is not None:
                        nn.init.constant_(l.bias, 0)

        for modules in [self.cls_tower, self.bbox_tower]:
            for l in modules.modules():
                if isinstance(l, nn.Linear):
                    weight_init.c2_xavier_fill(l)

        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_pred, self.bbox_pred]:
            if l.bias is not None:
                nn.init.constant_(l.bias, 0)

    def positional_encoding(self, feature, index):
        B, C, H, W = feature.shape
        N, E = self.ins_embed.weight.shape
        assert C == E, print("The dim of input feature and positional embedding should be same.")
        pe = self.ins_embed.weight[H*W*index:H*W*(index+1), :].view(1, H, W, C).permute(0, 3, 1, 2)
        pe = pe.repeat([B, 1, 1, 1])
        feature = feature + pe
        return feature

    def rescale_and_cat(self, features):
        ins_features = []
        for l, ins_feat in enumerate(features):
            dst_size = tuple([self.num_grids[l], self.num_grids[l]])
            if self.with_coord:
                ins_feat = CoordConv(ins_feat)
            ins_feat = F.interpolate(ins_feat, size=dst_size, mode='bilinear', align_corners=False)
            if self.pe_on:
                ins_feat = self.positional_encoding(ins_feat, l)
            ins_features.append(ins_feat)
        ins_features = torch.cat(ins_features, dim=0)
        return ins_features

    def attention_forward(self, ins_features, base_features, num_feats=5):
        bsz, ch, h, w = ins_features.size()
        # query.
        ins_query = self.ins_att(ins_features)
        ins_query = ins_query.flatten(2).permute(0, 2, 1)
        scaling = float(ins_query.shape[2]) ** -0.5
        ins_query = ins_query * scaling
        # key.
        base_key = self.base_att(base_features)
        base_key = base_key.repeat([num_feats, 1, 1, 1])
        base_key = base_key.flatten(2)
        attn = torch.bmm(ins_query, base_key)
        # attn type.
        if self.type_att == 'softmax':
            attn = attn.softmax(-1)
        elif self.type_att == 'sigmoid':
            attn = attn.sigmoid()
            # norming = torch.norm(attn, dim=2, keepdim=True) + 1e-5
            norming = torch.sum(attn, dim=2, keepdim=True) + 1e-5
            attn = attn / norming
        else:
            raise NotImplementedError
        # dense2sparse.
        base_features = base_features.flatten(2).permute(0, 2, 1)
        base_features = base_features.repeat([num_feats, 1, 1])
        ins_attn = torch.bmm(attn, base_features)
        ins_attn = ins_attn.permute(0, 2, 1)
        ins_attn = ins_attn.view(bsz, ch, h, w)
        return ins_attn

    def forward(self, ins_features, base_features=None):
        """
        Arguments:
            ins_features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
            base_features (Tensor): Encoded features.

        Returns:
            pass
        """

        # feature extraction.
        num_feats = len(ins_features)
        ins_features = self.rescale_and_cat(ins_features)
        ins_features = self.ins_before_tower(ins_features)

        # att.
        if self.use_base:
            ins_attn = self.attention_forward(ins_features, base_features, num_feats)
            ins_features = ins_features + ins_attn

        # feature refine.
        if self.with_coord:
            ins_features = CoordConv(ins_features)
        ins_features = self.ins_after_tower(ins_features)

        # fc.
        if ins_features.dim() > 3:
            ins_features = ins_features.flatten(2).permute(0, 2, 1)
            cate_features = self.cls_tower(ins_features)
            bbox_features = self.bbox_tower(ins_features)

        # pred.
        cate_pred = self.cls_pred(cate_features)
        bbox_pred = self.bbox_pred(bbox_features)
        if self.center_symmetry:
            centers = torch.tanh(bbox_pred[:, :, :2])
            sizes = torch.sigmoid(bbox_pred[:, :, 2:])
            bbox_pred = torch.cat([centers, sizes], dim=-1)
        else:
            bbox_pred = bbox_pred.sigmoid()

        return cate_pred, bbox_pred
