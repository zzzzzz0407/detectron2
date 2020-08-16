import json

if __name__ == '__main__':
    jsonFile = '/data00/home/zhangrufeng1/projects/detectron2/projects/detr/datasets/mot/mot17/annotations/mot17_train_half.json'
    with open(jsonFile, 'r') as f:
        infos = json.load(f)

    count_dict = dict()
    for info in infos["images"]:
        if info["file_name"] in ["MOT17-02-FRCNN/img1/000091.jpg"]:



    for ann in infos['annotations']:
        if ann["image_id"] not in count_dict.keys() and ann["iscrowd"] == 0 and ann["bbox"][2] >= 1e-5 and ann["bbox"][3] >= 1e-5:
            count_dict[ann["image_id"]] = 1
        elif ann["image_id"] in count_dict.keys() and ann["iscrowd"] == 0:
            count_dict[ann["image_id"]] += 1

    max_count = 0
    min_count = 999
    num_freq = 0
    for key, value in count_dict.items():
        max_count = max(max_count, value)
        min_count = min(min_count, value)
        if value > 100:
            num_freq += 1

    print("max_count: {}".format(max_count))
    print("min_count: {}".format(min_count))
    print("num_freq: {}".format(num_freq))
