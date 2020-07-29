# coding: utf-8
import os
import argparse
import json
import random


def parse_args():
    parser = argparse.ArgumentParser(description='Category Dataset Generation.')
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
    parser.add_argument('--category', type=str, default='person')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(666)
    args = parse_args()

    # semi dataset.
    dataDir = args.data_dir
    dataType = args.data_type
    ratio_semi = args.ratio_semi
    num_min = args.num_min
    cate_name = args.category
    out_dir = os.path.join(dataDir, args.out_dir)
    annFile = os.path.join(out_dir, 'instances_{}_semi_{}_min_{}.json'.
                           format(dataType, ratio_semi, num_min))
    cate_annFile = os.path.join(out_dir, 'instances_{}_semi_{}_min_{}_{}.json'.
                                format(dataType, ratio_semi, num_min, cate_name))
    with open(annFile, 'r') as f:
        json_semi = json.load(f)

    # Filter category.
    categories = json_semi['categories']
    for cate_info in categories:
        if cate_info['name'] == cate_name:
            category = [cate_info]
            cate_id = cate_info['id']
            break;

    # Filter annotations.
    annotations_semi = json_semi['annotations']
    num_semi = len(annotations_semi)
    annotations = list()
    for i, ann in enumerate(annotations_semi):
        if i % 100 == 0:
            print('Clean {}/{}'.format(i, num_semi))
        if ann.get('category_id') == cate_id:
            annotations.append(ann)
    num_ins = len(annotations)

    # generate json(semi).
    json_cate = dict()
    json_cate['info'] = json_semi['info']
    json_cate['licenses'] = json_semi['licenses']
    json_cate['images'] = json_semi['images']
    json_cate['categories'] = categories
    json_cate['annotations'] = annotations
    with open(cate_annFile, 'w') as f:
        json.dump(json_cate, f)
    print('Generate the annotations in {}'.format(cate_annFile))
    print('Filter for {} annotations: {}/{}'.format(cate_name, num_ins, num_semi))
    print('num_semi: {}, num_cate: {}'.format(num_semi, num_ins))





