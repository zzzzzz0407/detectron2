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
from .crowdhuman import load_crowdhuman_json
from .mot import load_mot_json

# CrowdHuman.
CROWDHUMAN_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
]

# ==== Predefined datasets and splits for COCO ==========
_PREDEFINED_SPLITS_CROWDHUAMN = dict()
_PREDEFINED_SPLITS_CROWDHUAMN["crowdhuman"] = {
    "CrowdHuman_train": ("crowdhuman/CrowdHuman_train",
                         "crowdhuman/annotations/CrowdHuman_train.json"),
    "CrowdHuman_val": ("crowdhuman/CrowdHuman_val",
                       "crowdhuman/annotations/CrowdHuman_val.json"),
}

# MOT.
MOT_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "pedestrian"},
]
_PREDEFINED_SPLITS_MOT = dict()
_PREDEFINED_SPLITS_MOT["mot"] = {
    "mot17_train_half": ("mot/mot17/train",
                         "mot/mot17/annotations/mot17_train_half.json"),
    "mot17_val_half": ("mot/mot17/train",
                       "mot/mot17/annotations/mot17_val_half.json"),
}


__all__ = ["register_crowdhuman_instances", "register_mot_instances"]


def _get_builtin_metadata(dataset_name):
    if dataset_name == "crowdhuman":
        thing_ids = [k["id"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
        thing_colors = [k["color"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
        assert len(thing_ids) == 1, len(thing_ids)
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [k["name"] for k in CROWDHUMAN_CATEGORIES if k["isthing"] == 1]
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
        }
        return ret
    elif dataset_name == "mot":
        thing_ids = [k["id"] for k in MOT_CATEGORIES if k["isthing"] == 1]
        thing_colors = [k["color"] for k in MOT_CATEGORIES if k["isthing"] == 1]
        assert len(thing_ids) == 1, len(thing_ids)
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
        thing_classes = [k["name"] for k in MOT_CATEGORIES if k["isthing"] == 1]
        ret = {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
        }
        return ret

    raise KeyError("No built-in metadata for dataset {}".format(dataset_name))


def register_crowdhuman_instances(name, metadata, json_file, image_root):
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
    DatasetCatalog.register(name, lambda: load_crowdhuman_json(json_file, image_root, name))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )


def register_all_crowdhuman(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_CROWDHUAMN.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_crowdhuman_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


def register_mot_instances(name, metadata, json_file, image_root):
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
    DatasetCatalog.register(name, lambda: load_mot_json(json_file, image_root, name,
                                                        extra_annotation_keys=["conf", "track_id"]))

    # 2. Optionally, add metadata about this dataset,
    # since they might be useful in evaluation, visualization or logging
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="coco", **metadata
    )  # todo: yan zheng dei chong xin xie.


def register_all_mot(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_MOT.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_mot_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


# Register them all under "./datasets"
_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_crowdhuman(_root)
register_all_mot(_root)

