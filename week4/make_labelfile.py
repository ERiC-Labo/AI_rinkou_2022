from distutils.file_util import write_file
import os, sys
import argparse
from yaml import parse 

parser = argparse.ArgumentParser(description="fdf")

parser.add_argument("--folder")
args = parser.parse_args()
path = os.path.join(args.folder, "labels")

files = os.listdir(path)


for fi in files:
    filepath = os.path.join(path, fi)
    writedir = os.path.join(args.folder, "labels_kai")
    if not os.path.exists(writedir):
        os.mkdir(writedir)
    writefilepath = os.path.join(writedir, fi)

    f = open(filepath, "r")
    fw = open(writefilepath, "w")
    datalist = f.readlines()
    for data in datalist:
        wakeru = data.split()
        wakeru[1] = float(wakeru[1])
        wakeru[2] = float(wakeru[2])
        wakeru[3] = float(wakeru[3])
        wakeru[4] = float(wakeru[4])
        x1 = 2064*(wakeru[1]*2-wakeru[3])/2
        x2 = 2064*(wakeru[1]*2+wakeru[3])/2
        y1 = 1544*(wakeru[2]*2-wakeru[4])/2
        y2 = 1544*(wakeru[2]*2+wakeru[4])/2
        x1 = str(round(x1, 3))
        x2 = str(round(x2, 3))
        y1 = str(round(y1, 3))
        y2 = str(round(y2, 3))
        # print(wakeru[0], " ", x1, " ", y1, " ", x2, " ", y2)
        fw.write(wakeru[0] + " " + x1 + " " + y1 + " " + x2 + " " + y2 + "\n")