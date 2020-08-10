# coding: utf-8
import os
import pandas as pd

if __name__ == '__main__':
    # Params.
    src_file = '/data00/home/zhangrufeng1/input2.txt'
    size = 5
    flag = 'FF'

    file_path, file_ext = os.path.splitext(src_file)
    dst_file = file_path + '_filter' + file_ext
    dst_writer = open(dst_file, 'w')

    # Load data.
    lines = pd.read_csv(src_file, encoding='unicode_escape').values

    for index in range(len(lines)):
        line = lines[index]
        if not isinstance(line, str):
            line = line.tolist()[0]

        start_id = line.find(flag)
        line = line[start_id:].split()
        count = len(line) // size
        if count < 1:
            continue
        else:
            i = 0
            for i in range(count):
                dst_writer.write(" ".join(line[i*size:(i+1)*size]) + "\n")
    dst_writer.close()
