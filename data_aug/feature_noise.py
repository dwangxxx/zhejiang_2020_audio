import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool

overwrite = True

file_path = '/data/dengwang/audio/data/train_no_noise/'
csv_file = 'audio_2.txt'
output_path = '/data/dengwang/audio/data/train_feature_clean'
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
with open(csv_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        wavpath.append(line.split(' ')[0])

print(len(wavpath))

for i in range(len(wavpath)):
    stereo, sr = librosa.load(file_path + wavpath[i], sr = sr, duration = duration)
    # stereo, fs = sound.read(file_path + wavpath[i], stop = duration * sr)
    noise = np.random.normal(0, 1, len(stereo))
    augmented_data = np.where(stereo != 0.0, stereo.astype('float64') + 0.01 * noise, 0.0).astype(np.float32)
    stereo = augmented_data 
    logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
    logmel_data[:, :, 0] = librosa.feature.melspectrogram(stereo[:], sr = sr, n_fft = num_fft, hop_length = hop_length, n_mels = num_freq_bin, fmin = 0.0, fmax = sr / 2, htk = True, norm = None)

    logmel_data = np.log(logmel_data + 1e-8)

    feat_data = logmel_data
    feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))

    feature_data = {'feat_data': feat_data,}

    cur_file_name = output_path + '/' + wavpath[i][:-4] + '_noise.' + feature_type
    print(i, cur_file_name)
    pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol = pickle.HIGHEST_PROTOCOL)