"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np

from .box_ops import box_cxcywh_to_xyxy, generalized_box_iou
from .coordconv import center_prior


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    def __init__(self,
                 soi,
                 strides,
                 loc_prior=False,
                 scale_variance=False,
                 cost_class: float = 1,
                 cost_bbox: float = 1,
                 cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.soi = soi
        self.strides = strides
        assert len(self.soi) == len(self.strides), print("The number of soi should equal to strides.")
        self.num_stage = len(self.soi)
        self.loc_prior = loc_prior
        self.scale_variance = scale_variance
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cannot be 0"

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
        # Concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if self.loc_prior:
            # Compute the loc prior.
            if self.scale_variance:
                num_loc = num_queries // self.num_stage
                out_loc = center_prior(num_loc, tgt_bbox.device)
                out_loc = out_loc.repeat(self.num_stage, 1)
            else:
                out_loc = center_prior(num_queries, tgt_bbox.device)

            # Compute the euclidean metric between boxes.
            cost_prior = torch.cdist(out_loc, tgt_bbox[:, :2], p=2)
            C = cost_prior.unsqueeze(0).repeat(bs, 1, 1).cpu()

        else:
            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(
                -1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(
                0, 1)  # [batch_size * num_queries, 4]

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                             box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        # assign different gts to different levels.
        if self.scale_variance:
            indices = []
            batched_costs = C.split(sizes, -1)
            num_grids = num_queries // self.num_stage
            for i, c in enumerate(batched_costs):
                match_pred = []
                match_gt = []
                costs = c[i].split(num_grids, 0)
                areas = torch.sqrt(targets[i]['area'])
                match_strides = tgt_ids.new_zeros(areas.shape)
                level_indices = tgt_ids.new_tensor([])
                for l, cost in enumerate(costs):
                    # choose the satisfied targets.
                    stride = self.strides[l]
                    scale_bound = areas.new_tensor(self.soi[l])
                    scale_bound = scale_bound.expand(len(areas), -1)
                    is_cared_in_the_level = \
                        (areas >= scale_bound[:, 0]) & (areas < scale_bound[:, 1])

                    # leave the redundant gt to next stage.
                    is_cared_in_the_level[level_indices] = True
                    num_redundant = is_cared_in_the_level.sum() - num_grids
                    if num_redundant > 0:
                        stage_areas = areas * is_cared_in_the_level
                        _, level_indices = stage_areas.topk(k=num_redundant)
                        is_cared_in_the_level[level_indices] = False
                    else:
                        level_indices = tgt_ids.new_tensor([])

                    cost = cost[:, is_cared_in_the_level]
                    indices_in_the_level = is_cared_in_the_level.nonzero().flatten()
                    num_ins = len(indices_in_the_level)
                    if num_ins == 0:
                        continue

                    # bipartite match and transfer.
                    match_src = linear_sum_assignment(cost)
                    match_pred += (match_src[0] + l * num_grids).tolist()
                    match_gt += (indices_in_the_level[match_src[1]].cpu().numpy()).tolist()
                    match_strides[indices_in_the_level[match_src[1]]] = stride

                # ensure that all gts are assigned to preds (one by one).
                assert len(set(match_gt)) == len(areas), \
                    print("There are some gts not assigned.")
                assert len(set(match_pred)) == len(set(match_gt)), \
                    print("Bipartite matching should be one by one.")
                indices.append(tuple([np.array(match_pred), np.array(match_gt)]))
                targets[i]['strides'] = match_strides

        else:
            indices = [
                linear_sum_assignment(c[i])
                for i, c in enumerate(C.split(sizes, -1))
            ]

        match_indices = [(torch.as_tensor(i, dtype=torch.int64),
                 torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        return match_indices, targets
