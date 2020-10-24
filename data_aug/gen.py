import os
import glob

path = '/tmp1/home/dw/audio/baseline/class2/evaluation_setup/train_full.txt'
des = '/tmp1/home/dw/audio/data/features_train/feature/*.logmel'
des_f = open('/tmp1/home/dw/audio/baseline/class2/evaluation_setup/train.txt', 'w')
files = glob.glob(des)
files1 = []
files2 = []
for f in files:
    files1.append(f.split('/')[-1].split('.')[0]) 
lines = open(path, 'r').readlines()
for line in lines:
    files2.append(line.split(' ')[0].split('.')[0])
for i in range(len(files2)):
    if files2[i] in files1:
        des_f.write(lines[i])