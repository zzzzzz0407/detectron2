# coding:utf-8
import os
import argparse
import zipfile


def parse_args():
    parser = argparse.ArgumentParser(description='Code for zip uploading.')
    parser.add_argument("--local-dir", type=str, help="the name of local dir.")
    parser.add_argument("--dst-dir", type=str, help="the name of hdfs dir.")
    args = parser.parse_args()
    return args


def zip_uploading(local_dir, dst_dir):
    # zip.
    zip_path = local_dir + '.zip'
    z = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    for path, dirnames, filenames in os.walk(local_dir):
        this_path = os.path.abspath('.')
        fpath = path.replace(this_path, '')
        for filename in filenames:
            z.write(os.path.join(path, filename), os.path.join(fpath, filename))
    z.close()

    # upload.
    cmd = "hadoop fs -copyFromLocal -f " + zip_path + " " + dst_dir
    os.system(cmd)


if __name__ == "__main__":
    args = parse_args()
    zip_uploading(args.local_dir, args.dst_dir)


