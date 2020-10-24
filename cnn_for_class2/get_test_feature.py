import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from numpy.random import seed, randint

overwrite = True

file_path = '/data/dengwang/audio/data/test/'
csv_file = 'test.txt'
output_path = '/data/dengwang/audio/data/features_test_pitch/'
feature_type = 'logmel'

# pitch
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
with open(csv_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        wavpath.append(line.split(' ')[0])

mode = 'test'

for i in range(len(wavpath)):
    if mode == 'test':
        stereo, sr = librosa.load(file_path + wavpath[i].strip(), sr = sr)
        length = sr * duration 
        range_high = len(stereo) - length
        seed(1)
        random_start = randint(range_high, size = 1)
        stereo = stereo[random_start[0]:random_start[0] + length]
    n_step = np.random.uniform(-4, 4)
    y_pitched = librosa.effects.pitch_shift(stereo, sr, n_steps = n_step)
    length = len(stereo)
    stereo = y_pitched
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:, :, 0]= librosa.feature.melspectrogram(stereo[:], sr = sr, n_fft = num_fft, \
                    hop_length = hop_length, n_mels = num_freq_bin, fmin = 0.0, fmax = sr / 2, htk = True, norm = None)

    logmel_data = np.log(logmel_data + 1e-8)

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))

    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + wavpath[i][:-4] + feature_type
    print(i, cur_file_name)
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)