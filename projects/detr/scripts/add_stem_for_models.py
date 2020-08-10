# coding:utf-8

import os
import pickle


if __name__ == "__main__":
    oriModels = "/data00/home/zhangrufeng1/pretrained/detectron2/torchvision-R-50.pkl"
    rootDir, nameFile = os.path.split(oriModels)
    nameFile, extFile = os.path.splitext(nameFile)
    newModels = os.path.join(rootDir, "{}_double_stems{}".format(nameFile, extFile))

    with open(oriModels, 'rb') as f:
        checkpoints = pickle.load(f, encoding="latin1")

    models = checkpoints['model']
    new_models = dict()
    for key, value in models.items():
        if "stem" in key:
            new_key = key.split('.')
            new_key[0] = "stem_pre"
            new_key = ".".join(new_key)
            new_models[new_key] = value
        new_models[key] = value

    checkpoints['model'] = new_models
    with open(newModels, 'wb') as f:
        pickle.dump(checkpoints, f)
