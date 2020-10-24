import os
import sys
import glob

src_path = '/data/dengwang/audio/baseline/cnn_for_class2/train_val/val_full_B.txt'
des_path = '/data/dengwang/audio/baseline/cnn_for_class2/train_val/val_full_clean.txt'
fil = 'del.txt'

names = open(fil, 'r').readlines()
fil_names = []


for name in names:
    fil_names.append(name.split('.')[0])


des = open(des_path, 'w')
lines = open(src_path, 'r').readlines()
for line in lines:
    if line.split('_')[0] not in fil_names or 'pitch' not in line.split(' ')[0]:
        des.write(line)