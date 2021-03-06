# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""
import os
import sys

# fmt: off
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on

import time
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict
import torch
import json

import detectron2.utils.comm as comm
from d2.detr import DetrDatasetMapper, add_detr_config, DetrTrackMapper
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader # MetadataCatalog #, build_detection_train_loader
from detectron2.engine import AutogradProfiler, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluator, print_csv_format # ,inference_on_dataset
from detectron2.solver.build import maybe_add_gradient_clipping
from fvcore.common.file_io import PathManager

from d2.detr import build_detection_train_loader
from d2.detr.inference import inference_on_dataset
from d2.detr.tracker import Tracker


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to DETR.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        self.clip_norm_val = 0.0
        if cfg.SOLVER.CLIP_GRADIENTS.ENABLED:
            if cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
                self.clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
        super().__init__(cfg)

    def run_step(self):
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        if self.clip_norm_val > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm_val)
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.META_ARCHITECTURE == "Detr":
            mapper = DetrDatasetMapper(cfg, True)
        elif cfg.MODEL.META_ARCHITECTURE == "DetrTrack":
            mapper = DetrTrackMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = torch.optim.SGD(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif optimizer_type == "ADAMW":
            optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.META_ARCHITECTURE == "DetrTrack":
            mapper = DetrTrackMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger("detectron2")
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name)
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            # inference.
            tracker = Tracker(cfg)
            results_i, res_tracks = inference_on_dataset(model, data_loader, tracker, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

                if res_tracks is not None:
                    file_path = os.path.join(cfg.OUTPUT_DIR, "inference", "track_results.json")
                    logger.info("Saving tracking results to {}".format(file_path))
                    with PathManager.open(file_path, "w") as f:
                        f.write(json.dumps(res_tracks))
                        f.flush()

        if len(results) == 1:
            results = list(results.values())[0]

        return results


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_detr_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    if args.eval_only:
        # It's important to set num_worker = 1 for tracking.
        cfg.DATALOADER.NUM_WORKERS = 1
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
