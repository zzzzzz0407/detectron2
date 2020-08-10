# coding:utf-8
import json
import os

if __name__ == '__main__':
    jsonFile = '/data00/home/zhangrufeng1/datasets/crowdhuman/annotations/train.json'
    dataPath, nameFile = os.path.split(jsonFile)
    jsonFile_filter = os.path.join(dataPath, 'CrowdHuman_{}'.format(nameFile))
    with open(jsonFile, 'r') as f:
        infos = json.load(f)

    count_dict = dict()
    for ann in infos['annotations']:
        if ann["image_id"] not in count_dict.keys() and ann["iscrowd"] == 0:
            count_dict[ann["image_id"]] = 1
        elif ann["image_id"] in count_dict.keys() and ann["iscrowd"] == 0:
            count_dict[ann["image_id"]] += 1

    ids_freq = list()
    for key, value in count_dict.items():
        if value > 100:
            ids_freq.append(key)

    anns_filter = list()
    for ann in infos['annotations']:
        if ann["image_id"] in ids_freq:
            continue
        anns_filter.append(ann)

    ori_size = len(infos['annotations'])
    filter_size = len(anns_filter)
    print("{}/{}".format(filter_size, ori_size))
    infos['annotations'] = anns_filter
    with open(jsonFile_filter, 'w') as f:
        json.dump(infos, f)
