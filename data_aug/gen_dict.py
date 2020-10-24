import os
import sys

# 生成字典
path = '/data/dengwang/audio/data/train/train.txt'
name_list = []

with open(path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        name = line.strip()
        name = name.split(' ')
        if (name[1] == '0'):
            name_list.append(name[2])
name_list = list(set(name_list))
name_dict = {}
i = 0
for name in name_list:
    name_dict[name] = i
    i += 1
print(name_dict)