# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json

# ==== Predefined datasets and splits for COCO ==========
_PREDEFINED_SPLITS_COCO = dict()
_PREDEFINED_SPLITS_COCO["coco"] = {
    "instances_train2017_semi_0.1_min_200": ("coco/train2017",
                                             "coco/annotations/semi/instances_train2017_semi_0.1_min_200.json"),
    "instances_val2017_semi_0.1_min_200": ("coco/val2017",
                                           "coco/annotations/semi/instances_val2017_semi_0.1_min_200.json"),
    "instances_train2017_semi_0.2_min_200": ("coco/train2017",
                                             "coco/annotations/semi/instances_train2017_semi_0.2_min_200.json"),
    "instances_train2017_semi_0.5_min_200": ("coco/train2017",
                                             "coco/annotations/semi/instances_train2017_semi_0.5_min_200.json"),
    "instances_train2017_semi_0.05_min_200": ("coco/train2017",
                                              "coco/annotations/semi/instances_train2017_semi_0.05_min_200.json"),
    "instances_train2017_semi_0.01_min_100": ("coco/train2017",
                                              "coco/annotations/semi/instances_train2017_semi_0.01_min_100.json"),
    # categories.
    "instances_train2017_semi_0.1_min_200_person": ("coco/train2017",
                                                    "coco/annotations/semi/"
                                                    "instances_train2017_semi_0.1_min_200_person.json"),
    "instances_val2017_semi_0.1_min_200_person": ("coco/val2017",
                                                  "coco/annotations/semi/"
                                                  "instances_val2017_semi_0.1_min_200_person.json"),
}


__all__ = ["register_coco_instances"]


def register_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    # 1. register a function which returns dicts
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name,
                                                         extra_annotation_keys=["blind"]))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)

