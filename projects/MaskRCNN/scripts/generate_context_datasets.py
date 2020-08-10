# coding: utf-8
import os
import argparse
import json
import cv2
import torch
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Context Dataset Generation.')
    parser.add_argument('--data-dir', type=str,
                        default='/data00/home/zhangrufeng1/datasets/coco',
                        help='the name of data root.')
    parser.add_argument('--data-type', type=str,
                        default='train2017',
                        help='the name of data type.')
    parser.add_argument('--out-dir', type=str,
                        default='context',
                        help='the name of out dir.')
    # pred.
    parser.add_argument('--pred-path', type=str, default="/data00/home/zhangrufeng1/projects/detectron2/projects/MaskRCNN/models/"
                                                         "faster_rcnn_R_50_FPN_1x_train/inference/instances_predictions.pth")
    parser.add_argument('--out-name', type=str, default="faster_rcnn_R_50_FPN_1x_train")
    parser.add_argument('--context-type', type=str, default='bbox')
    parser.add_argument('--thresh', type=float, default=0.)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # dataset.
    dataDir = args.data_dir
    dataType = args.data_type
    imgDir = os.path.join(dataDir, dataType)
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    with open(annFile, 'r') as f:
        info_ann = json.load(f)

    # other params.
    thresh = args.thresh
    context_type = args.context_type

    # output json.
    predFile = args.pred_path
    info_pred = torch.load(predFile)
    len_pred = len(info_pred)

    out_name = args.out_name + "_" + args.context_type + "_" + str(args.thresh)
    out_dir = os.path.join(dataDir, args.out_dir, out_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Statistics.
    dictionary = dict()
    for img_info in info_ann["images"]:
        dictionary[img_info['id']] = img_info['file_name']

    for index, pred in enumerate(info_pred):
        img_file = dictionary[pred['image_id']]
        img_path = os.path.join(imgDir, img_file)
        new_img_path = os.path.join(out_dir, img_file)
        bboxes = pred['instances']
        img = cv2.imread(img_path)
        size = img.shape
        mask = np.zeros(size).astype(np.uint8)
        for bbox in bboxes:
            if bbox['score'] < thresh:
                continue
            bbox = bbox['bbox']
            x_min, y_min = int(bbox[0]), int(bbox[1])
            x_max, y_max = int((bbox[0]+bbox[2]+1)), int((bbox[1]+bbox[3]+1))
            x_max = min(x_max, size[1])
            y_max = min(y_max, size[0])
            mask[y_min:y_max, x_min:x_max, :] = 1

        img = img * mask
        cv2.imwrite(new_img_path, img)
        print("Process {}/{}".format(index, len_pred))
