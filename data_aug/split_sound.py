import soundfile as sf
import os
from scipy.io import wavfile
import numpy as np
from numpy.random import seed, randint
import librosa

# 将训练数据分割成10s
path = '/data/dengwang/audio/data/testB/test.txt'
data_path = '/data/dengwang/audio/data/testB/'
des_path = '/data/dengwang/audio/data/test_split/'

# 切割训练数据
def split_data(files, des_file):
    sr = 44100
    duration = 150
    f = open(files, 'r')
    lines = f.readlines()
    lines = lines[:]
    sample_num = len(lines)
    seg_num = 3
    # 分割成10s
    seg_len = 10    
    print(len(lines))
    split_txt = open(os.path.join(des_path, 'test.txt'), 'w')
    for i in range(len(lines)):
        line = lines[i].strip()
        # line = lines[i]
        sound_file = line.split(' ')[0]
        print(i, sound_file)
        # sr默认源数据的sr
        #wav_data, sr = sf.read(os.path.join(data_path, sound_file))
        wav_data, sr = librosa.load(os.path.join(data_path, sound_file), sr = sr)

        # 10s片段
        length = sr * seg_len 
        range_high = len(wav_data) - length
        seed(1)
        # 随机生成截取开始位置
        random_start = randint(range_high, size = seg_num)
        
        # 截取数据
        for j in range(seg_num):
            cur_wav = wav_data[random_start[j]: random_start[j] + length]
            # 保存截取的数据
            cur_name = sound_file.split('.')[0] + '_' + str(j) + '.wav'
            des_name = os.path.join(des_file, cur_name)
            sf.write(des_name, cur_wav, sr)
            # 同时重新写入标签
            des_str = cur_name + ' ' + line.split(' ')[1] + ' ' + line.split(' ')[2] + '\n'
            split_txt.write(des_str)

split_data(path, des_path)