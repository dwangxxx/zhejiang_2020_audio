import os
import glob

file_path = '/data/dengwang/audio/data/train/train.txt'
des_path = '/data/dengwang/audio/data/split_train/train.txt'
lines = open(file_path, 'r').readlines()
des = open(des_path, 'w')
for line in lines[1:]:
    for i in range(20):
        cur_name = line.split(' ')[0].split('.')[0] + '_' + str(i) + '.wav'
        des_str = cur_name + ' ' + line.split(' ')[1] + ' ' + line.split(' ')[2]
        des.write(des_str)