import numpy as np 
import os
import sys

class2_train = '../cnn_for_class2/train_val/train_full_B.txt'
class2_evaluate = '../cnn_for_class2/train_val/val_full_B.txt'
class2_list = ['audio_2.txt', 'audio_2_noise.txt', 'audio_2_pitch.txt', 'audio_2_time.txt', 'audio_2_add.txt']

class50_train = '../cnn_for_class50/train_val/train_full_B.txt'
class50_evaluate = '../cnn_for_class50/train_val/val_full_B.txt'
class50_list = ['audio_50.txt', 'audio_50_noise.txt', 'audio_50_pitch.txt', 'audio_50_time.txt', 'audio_50_add.txt']

train = open(class2_train, 'w')
val = open(class2_evaluate, 'w')
# 生成二分类的训练数据和验证数据
for i in range(5):
    with open(class2_list[i], 'r') as f:
        lines = f.readlines()
        sel = range(0, len(lines), 10)
        #sel = np.random.randint(len(lines), size = int(0.1 * len(lines)))
        #sel = list(set(sel))
        print(len(sel))
        for i in range(len(lines)):
            if i in sel:
                val.write(lines[i])
            else:
                train.write(lines[i])


train = open(class50_train, 'w')
val = open(class50_evaluate, 'w')
# 生成多分类的训练数据和验证数据
for i in range(5):
    with open(class50_list[i], 'r') as f:
        lines = f.readlines()
        sel = range(0, len(lines), 10)
        #sel = np.random.randint(len(lines), size = int(0.1 * len(lines)))
        #sel = list(set(sel))
        print(len(sel))
        for i in range(len(lines)):
            if i in sel:
                val.write(lines[i])
            else:
                train.write(lines[i])