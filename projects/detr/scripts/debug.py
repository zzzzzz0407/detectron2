import json

if __name__ == '__main__':
    jsonFile = '/data00/home/zhangrufeng1/datasets/crowdhuman/annotations/CrowdHuman_val.json'
    with open(jsonFile, 'r') as f:
        infos = json.load(f)

    debug_file = '/data00/home/zhangrufeng1/datasets/coco/annotations/instances_val2017.json'
    with open(debug_file, 'r') as f:
        infos_debug = json.load(f)
    aa = 1