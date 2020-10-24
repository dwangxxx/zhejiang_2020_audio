import os
import sys

path = '/data/dengwang/audio/data/split_train/train.txt'

# 生成50分类的训练文件
des_path = 'audio_50.txt'
with open(path, 'r') as f1:
    lines = f1.readlines()
    f2 = open(des_path, 'w')
    for line in lines:
        name = line.strip()
        name = name.split(' ')
        if (name[1] == '0'):
            des = name[0] + ' ' + name[2] + '\n'
            f2.write(des)

# 生成二分类的训练文件
des_path = 'audio_2.txt'
with open(path, 'r') as f1:
    lines = f1.readlines()
    f2 = open(des_path, 'w')
    for line in lines:
        name = line.strip()
        name = name.split(' ')
        des = name[0] + ' ' + name[1] + '\n'
        f2.write(des)

des_path = ['audio_50_noise.txt', 'audio_50_pitch.txt', 'audio_50_time.txt', 'audio_50_add.txt']
des = ['noise', 'pitch', 'time', 'add']
with open(path, 'r') as f1:
    lines = f1.readlines()
    for i in range(4):
        f2 = open(des_path[i], 'w')
        for line in lines:
            name = line.strip()
            name = name.split(' ')
            if (name[1] == '0'):
                _des = name[0].split('.')[0] + '_' + des[i] + '.wav' + ' ' + name[2] + '\n'
                f2.write(_des)

des_path = ['audio_2_noise.txt', 'audio_2_pitch.txt', 'audio_2_time.txt', 'audio_2_add.txt']
des = ['noise', 'pitch', 'time', 'add']
with open(path, 'r') as f1:
    lines = f1.readlines()
    for i in range(4):
        f2 = open(des_path[i], 'w')
        for line in lines:
            name = line.strip()
            name = name.split(' ')
            _des = name[0].split('.')[0] + '_' + des[i] + '.wav' + ' ' + name[1] + '\n'
            f2.write(_des)