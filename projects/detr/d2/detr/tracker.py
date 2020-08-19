# coding: utf-8
import copy
import torch

from detectron2.data import MetadataCatalog


class Tracker(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_ins = cfg.MODEL.DETR.NUM_OBJECT_QUERIES
        self.score_thresh = cfg.MODEL.DETR.SCORE_THRESH
        self.track_thresh = cfg.MODEL.DETR.TRACK_THRESH

        category_mapper_o = MetadataCatalog.get(cfg.DATASETS.TEST[0]).thing_dataset_id_to_contiguous_id
        category_mapper = dict()
        for k, v in category_mapper_o.items():
            category_mapper[v] = k
        self.category_mapper = category_mapper

        self.id_count = 0  # 累计至目前的id号 (从1开始).
        self.track_ids = torch.zeros(self.num_ins)  # 上一帧的各位置的ID号.
        self.track_masks = torch.zeros(self.num_ins)  # 上一帧的目标位置 True/False.
        self.reset_all()

    def reset_all(self):
        self.id_count = 0
        self.track_ids = torch.zeros(self.num_ins)
        self.track_masks = torch.zeros(self.num_ins)

    def reset(self):
        self.track_ids = torch.zeros(self.num_ins)
        self.track_masks = torch.zeros(self.num_ins)

    def init_track(self, results):
        ret = list()

        # filter.
        scores = results.scores
        score_filter = scores >= self.score_thresh
        self.track_masks = score_filter.clone()

        scores = scores[score_filter]
        classes = results.pred_classes[score_filter]
        bboxes = results.pred_boxes.tensor[score_filter, :]
        num_obj = score_filter.sum()
        loc_obj = score_filter.nonzero()

        for idx in range(num_obj):
            # update property.
            self.id_count += 1
            self.track_ids[loc_obj[idx]] = self.id_count

            # add pred.
            obj = dict()
            obj["score"] = float(scores[idx])
            obj["class"] = self.category_mapper[int(classes[idx])]
            obj["bbox"] = bboxes[idx, :].cpu().numpy().tolist()
            obj["tracking_id"] = self.id_count
            ret.append(obj)

        return ret

    def step(self, results, public_det=None):
        ret = list()

        # filter.
        scores = results.scores
        tracks = results.pred_tracks
        score_filter = scores >= self.score_thresh
        track_filter = tracks >= self.track_thresh
        track_masks = self.track_masks
        match_filter = score_filter & track_filter & track_masks

        # matched items.
        scores_m = scores[match_filter]
        classes_m = results.pred_classes[match_filter]
        bboxes_m = results.pred_boxes.tensor[match_filter, :]
        num_obj_m = match_filter.sum()
        track_ids = self.track_ids[match_filter]

        for idx in range(num_obj_m):
            # add pred.
            obj = dict()
            obj["score"] = float(scores_m[idx])
            obj["class"] = self.category_mapper[int(classes_m[idx])]
            obj["bbox"] = bboxes_m[idx, :].cpu().numpy().tolist()
            obj["tracking_id"] = int(track_ids[idx])
            ret.append(obj)

        # update.
        self.track_ids *= match_filter.cpu()
        self.track_masks = score_filter.clone()

        # new items.
        new_filter = score_filter ^ match_filter
        scores_n = scores[new_filter]
        classes_n = results.pred_classes[new_filter]
        bboxes_n = results.pred_boxes.tensor[new_filter, :]
        num_obj_n = new_filter.sum()
        loc_obj_n = new_filter.nonzero()

        for idx in range(num_obj_n):
            # update property.
            self.id_count += 1
            self.track_ids[loc_obj_n[idx]] = self.id_count

            # add pred.
            obj = dict()
            obj["score"] = float(scores_n[idx])
            obj["class"] = self.category_mapper[int(classes_n[idx])]
            obj["bbox"] = bboxes_n[idx, :].cpu().numpy().tolist()
            obj["tracking_id"] = self.id_count
            ret.append(obj)
        assert (self.track_ids > 0).sum() == self.track_masks.sum()

        return ret
