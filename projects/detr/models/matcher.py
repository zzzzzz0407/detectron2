# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianMatcherTrack(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets, indices_track):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        indices_final = list()

        bs = len(indices_track)
        targets_cur = targets[0]
        targets_pre = targets[1]
        out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size, num_queries, num_classes]
        out_bbox = outputs["pred_boxes"]  # [batch_size, num_queries, 4]
        for i in range(bs):
            tmp_track_pre = targets_pre[i]["tracks"]
            tmp_track_pre = tmp_track_pre[indices_track[i][1]]  # pre track in detection order.
            tmp_track_cur = targets_cur[i]["tracks"]  # cur track in original order.
            # matching indices: pre_index / cur_index.
            pairs = (tmp_track_pre.unsqueeze(1) == tmp_track_cur).nonzero(as_tuple=True)
            # un-matching items in cur frame.
            target_track = torch.zeros_like(tmp_track_cur)
            target_track[pairs[1]] = 1
            targets[0][i]["binary_tracks"] = target_track.float()
            unmatched_cur = (target_track == 0).nonzero().squeeze(1)
            if len(unmatched_cur):
                # un-matching gt.
                target_cate = targets_cur[i]["labels"][unmatched_cur]
                target_bbox = targets_cur[i]["boxes"][unmatched_cur]
                # un-matching pred.
                pred_prob = out_prob[i]
                pred_bbox = out_bbox[i]
                flag_out = torch.ones_like(pred_prob[:, 0]).bool()  # [num_queries, 1]
                flag_out[indices_track[i][0][pairs[0]]] = False
                pred_prob = pred_prob[flag_out]
                pred_bbox = pred_bbox[flag_out]
                pred_queries = flag_out.nonzero().squeeze(1)
                # compute the cost matrix.
                cost_class = -pred_prob[:, target_cate]
                cost_bbox = torch.cdist(pred_bbox, target_bbox, p=1)
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(pred_bbox), box_cxcywh_to_xyxy(target_bbox))
                C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
                # match.
                unmatched_indices = linear_sum_assignment(C.cpu())
                matched_queries = torch.cat([indices_track[i][0][pairs[0]], pred_queries[unmatched_indices[0]].cpu()], dim=0)
                matched_gts = torch.cat([pairs[1], unmatched_cur[unmatched_indices[1]]], dim=0).cpu()
                indices_final.append((matched_queries, matched_gts, indices_track[i][2]))  # pos / gt / neg.
            else:
                indices_final.append((indices_track[i][0][pairs[0]], pairs[1].cpu(), indices_track[i][2]))

        return indices_final, targets[0]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)


def build_track_matcher(args):
    return HungarianMatcherTrack(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)