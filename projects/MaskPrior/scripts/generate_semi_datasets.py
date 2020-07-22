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
    parser.add_argument('--ratio-semi', type=float, default=0.01)
    parser.add_argument('--num-min', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    random.seed(666)
    args = parse_args()

    # origin dataset.
    dataDir = args.data_dir
    dataType = args.data_type
    annFile = '{}/instances_{}.json'.format(dataDir, dataType)
    with open(annFile, 'r') as f:
        json_original = json.load(f)

    # semi dataset.
    ratio_semi = args.ratio_semi
    num_min = args.num_min
    out_dir = os.path.join(dataDir, args.out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    annFile_semi = os.path.join(out_dir, 'instances_{}_semi_{}_min_{}.json'.
                                format(dataType, ratio_semi, num_min))

    # Filter out crowd annotations.
    categories = json_original['categories']
    annotations_ori = json_original['annotations']
    num_ori = len(annotations_ori)
    annotations = list()
    for i, ann in enumerate(annotations_ori):
        if i % 100 == 0:
            print('Clean {}/{}'.format(i, num_ori))
        if ann.get('iscrowd', 0) == 0:
            annotations.append(ann)
    num_ins = len(annotations)

    # Statistics.
    ids = []
    for cate in categories:
        ids.append(cate['id'])
    counts = [0 for _ in range(max(ids)+1)]
    cate_indices = [[] for _ in range(max(ids)+1)]
    for i, ann in enumerate(annotations):
        if i % 100 == 0:
            print('Statistics {}/{}'.format(i, num_ins))
        cate_indices[ann['category_id']].append(i)
        counts[ann['category_id']] += 1
    num_before = sum(counts)

    # Semi.
    num_blind = []
    for count in counts:
        if count > num_min:
            num_blind.append(max(num_min, int(count*ratio_semi)))
        else:
            num_blind.append(count)
    num_after = sum(num_blind)

    blind_indices = []
    for i, nb in enumerate(num_blind):
        if nb < num_min:
            blind_indices.append(cate_indices[i])
        else:
            blind_indices.append(random.sample(cate_indices[i], nb))

    annotations_semi = list()
    for i, ann in enumerate(annotations):
        if i % 100 == 0:
            print('Semi {}/{}'.format(i, num_ins))
        if i in blind_indices[ann['category_id']]:
            ann['blind'] = 1
        else:
            ann['blind'] = 0
        annotations_semi.append(ann)

    # generate json(semi).
    json_semi = dict()
    json_semi['info'] = json_original['info']
    json_semi['licenses'] = json_original['licenses']
    json_semi['images'] = json_original['images']
    json_semi['categories'] = json_original['categories']
    json_semi['annotations'] = annotations_semi
    with open(annFile_semi, 'w') as f:
        json.dump(json_semi, f)
    print('Generate the annotations in {}'.format(annFile_semi))
    print('Filter out crowd annotations: {}/{}'.format(num_ins, num_ori))
    print('num_ori: {}, num_semi: {}, final_ratio_semi: {}%'.format(num_before, num_after,
                                                                    num_after/num_before*100))





