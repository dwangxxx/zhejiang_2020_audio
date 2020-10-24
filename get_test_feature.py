import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from numpy.random import seed, randint

file_path = '/data/dengwang/audio/data/final_test_round2/'
txt_file = '/data/dengwang/audio/data/final_test_round2/test.txt'
output_path = '/data/dengwang/audio/data/feature_round2'
feature_type = 'logmel'

sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1

if not os.path.exists(output_path):
    os.makedirs(output_path)

wavpath = []
with open(txt_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        wavpath.append(line.split(' ')[0])


print(len(wavpath))
mode = 'test'


# 对源数据进行处理，生成源数据特征
for i in range(len(wavpath)):
    # 随机截取10s数据进行测试
    if mode == 'test':
        stereo, sr = librosa.load(file_path + wavpath[i].strip(), sr = sr)
        length = sr * duration 
        range_high = len(stereo) - length
        seed(1)
        random_start = randint(range_high, size = 1)
        stereo = stereo[random_start[0]:random_start[0] + length]
    else:
        stereo, sr = librosa.load(file_path + wavpath[i].strip(), sr = sr, offset = 0.0, duration = duration)
    # logmel特征的维度
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr=sr, n_fft=num_fft, 
                    hop_length = hop_length, n_mels = num_freq_bin, fmin = 0.0, fmax = sr / 2, htk = True, norm = None)

    logmel_data = np.log(logmel_data + 1e-8)
    
    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + '/' + wavpath[i][:-4] + feature_type
    print(i, cur_file_name, feat_data.shape)
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)