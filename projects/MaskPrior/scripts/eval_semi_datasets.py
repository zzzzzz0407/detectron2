# coding: utf-8
import os
import argparse
import json
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Semi Dataset Generation.')
    parser.add_argument('--data-dir', type=str,
                        default='/data00/home/zhangrufeng1/datasets/coco/annotations',
                        help='the name of data root.')
    parser.add_argument('--data-type', type=str,
                        default='train2017',
                        help='the name of data type.')
    parser.add_argument('--out-dir', type=str,
                        default='semi',
                        help='the name of out dir.')
    parser.add_argument('--ratio-semi', type=float, default=0.1)
    parser.add_argument('--num-min', type=int, default=200)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(666)
    args = parse_args()

    # dataset.
    dataDir = args.data_dir
    dataType = args.data_type
    ratio_semi = args.ratio_semi
    num_min = args.num_min
    out_dir = os.path.join(dataDir, args.out_dir)
    annFile_semi = os.path.join(out_dir, 'instances_{}_semi_{}_min_{}.json'.
                                format(dataType, ratio_semi, num_min))
    with open(annFile_semi, 'r') as f:
        json_original = json.load(f)

    # Filter out crowd annotations.
    categories = json_original['categories']
    annotations_ori = json_original['annotations']
    num_ori = len(annotations_ori)
    num_semi = 0
    for i, ann in enumerate(annotations_ori):
        if i % 100 == 0:
            print('Clean {}/{}'.format(i, num_ori))
        if ann['blind'] == 1:
            num_semi += 1

    print('num_ori: {}, num_semi: {}, final_ratio_semi: {}%'.format(num_semi, num_ori,
                                                                    num_semi/num_ori*100))





