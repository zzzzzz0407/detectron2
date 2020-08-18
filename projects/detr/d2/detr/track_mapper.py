# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging

import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.data import MetadataCatalog
from detectron2.data import get_detection_dataset_dicts
from .detection_utils import (annotations_to_instances,
                              transform_instance_annotations,
                              filter_empty_instances)

import cv2
import os
import zipfile

__all__ = ["DetrTrackMapper"]


_im_zfile = []


def zipimread(filename, flags=cv2.IMREAD_COLOR):
    global _im_zfile

    path_dir, path_file = os.path.split(filename)
    path_dir = path_dir.split("/")
    pre_dir, pos_dir = "/".join(path_dir[:-2]), "/".join(path_dir[-2:])
    filename = os.path.join(pre_dir + ".zip@", pos_dir, path_file)

    path = filename
    if flags == "unchanged":
        flags = cv2.IMREAD_UNCHANGED
    pos_at = path.index('@')
    if pos_at == -1:
        print("character '@' is not found from the given path '%s'"%(path))
        assert 0
    path_zip = path[0: pos_at]
    dataset_prefix = os.path.basename(path_zip).split('.')[0]
    path_img = dataset_prefix + path[pos_at + 1:]
    if not os.path.isfile(path_zip):
        print("zip file '%s' is not found" % path_zip)
        assert 0
    for i in range(len(_im_zfile)):
        if _im_zfile[i]['path'] == path_zip:
            data = _im_zfile[i]['zipfile'].read(path_img)
            return cv2.imdecode(np.frombuffer(data, np.uint8), flags)

    # print("read new image zip file '%s', '%s"%(path_zip, path_img))
    _im_zfile.append({
        'path': path_zip,
        'zipfile': zipfile.ZipFile(path_zip, 'r')
    })
    data = _im_zfile[-1]['zipfile'].read(path_img)

    return cv2.imdecode(np.frombuffer(data, np.uint8), flags)


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class DetrTrackMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by DETR.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        assert not cfg.MODEL.MASK_ON, "Mask is not supported"

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in pre-process: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        self.img_format = cfg.INPUT.FORMAT
        self.zip_read = cfg.INPUT.ZIP_READ
        self.is_train = is_train
        self.max_frame_dist = cfg.MODEL.DETR.MAX_FRAME_DIST

        # video2img --> 问题可能是出在这边，每次得释放掉 / 考虑放到builtin试试.
        if self.is_train:
            self.dataset_dicts = get_detection_dataset_dicts(
                cfg.DATASETS.TRAIN,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
                if cfg.MODEL.KEYPOINT_ON
                else 0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
            )
            assert len(cfg.DATASETS.TRAIN) == 1, logging.\
                info("Only support ONE dataset each time, however, "
                     "there are {} datasets now.".format(len(cfg.DATASETS.TRAIN)))
            self.video_to_images = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).video_to_images
            self.image_to_index = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).image_to_index
        else:
            self.dataset_dicts = get_detection_dataset_dicts(
                cfg.DATASETS.TEST,
                filter_empty=False,
                min_keypoints=0,
                proposal_files=None
            )
            assert len(cfg.DATASETS.TEST) == 1, logging. \
                info("Only support ONE dataset each time, however, "
                     "there are {} datasets now.".format(len(cfg.DATASETS.TEST)))
            self.video_to_images = MetadataCatalog.get(cfg.DATASETS.TEST[0]).video_to_images
            self.image_to_index = MetadataCatalog.get(cfg.DATASETS.TEST[0]).image_to_index

    def _load_pre_data(self, video_id, frame_id, sensor_id=1):
        img_infos = self.video_to_images[video_id]
        # If training, random sample nearby frames as the "previous" frame.
        # If testing, get the exact previous frame.
        if self.is_train:
            img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos
                       if abs(img_info['frame_id'] - frame_id) < self.max_frame_dist
                       and (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]

            frame_dist = 0
            while not frame_dist:   # not self.
                rand_id = np.random.choice(len(img_ids))
                img_id, pre_frame_id = img_ids[rand_id]
                frame_dist = abs(frame_id - pre_frame_id)

            pre_dict = self.dataset_dicts[self.image_to_index[img_id]]
            assert pre_dict['image_id'] == img_id
        else:
            if frame_id == 1:
                img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos
                           if abs(img_info['frame_id'] - frame_id) == 0
                           and (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
                assert img_ids[0][1] == frame_id
            else:
                img_ids = [(img_info['id'], img_info['frame_id']) for img_info in img_infos
                           if frame_id - img_info['frame_id'] == 1
                           and (not ('sensor_id' in img_info) or img_info['sensor_id'] == sensor_id)]
                assert img_ids[0][1] == frame_id - 1
            pre_dict = self.dataset_dicts[self.image_to_index[img_ids[0][0]]]
        return pre_dict

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # cur.
        cur_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # pre.
        pre_dict = self._load_pre_data(cur_dict["video_id"], cur_dict["frame_id"],
                                       cur_dict["sensor_id"] if "sensor_id" in cur_dict else 1)

        # we use zip loading.
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        image = zipimread(cur_dict["file_name"])
        pre_image = zipimread(pre_dict["file_name"])

        if self.zip_read:
            if self.img_format == "RGB":
                image = image[:, :, ::-1]
                pre_image = pre_image[:, :, ::-1]
        else:
            raise NotImplementedError

        utils.check_image_size(cur_dict, image)
        utils.check_image_size(pre_dict, pre_image)

        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        # apply the same transform to pre frame.
        pre_image = transforms.apply_image(pre_image)
        image_shape = image.shape[:2]  # h, w
        pre_image_shape = pre_image.shape[:2]
        assert image_shape == pre_image_shape

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        cur_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        pre_dict["image"] = torch.as_tensor(np.ascontiguousarray(pre_image.transpose(2, 0, 1)))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            cur_dict.pop("annotations", None)
            pre_dict.pop("annotations", None)
            return cur_dict, pre_dict

        if "annotations" in cur_dict and "annotations" in pre_dict:
            # cur.
            annos = [
                transform_instance_annotations(obj, transforms, image_shape)
                for obj in cur_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            instances = annotations_to_instances(annos, image_shape)
            cur_dict["instances"] = filter_empty_instances(instances)

            # pre.
            pre_annos = [
                transform_instance_annotations(obj, transforms, pre_image_shape)
                for obj in pre_dict["annotations"]
                if obj.get("iscrowd", 0) == 0
            ]
            pre_instances = annotations_to_instances(pre_annos, pre_image_shape)
            pre_dict["instances"] = filter_empty_instances(pre_instances)

        return cur_dict, pre_dict
