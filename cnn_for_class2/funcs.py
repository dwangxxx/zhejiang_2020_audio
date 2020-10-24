import numpy as np
import pickle
import random
import pandas as pd

def generate_train_txt(train_txt, feat_path, experiments):
    lines = open(train_txt, 'r').readlines()
    # 生成logmel文件信息
    for idx, elem in enumerate(lines):
        lines[idx] = lines[idx].split()
        lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

    temp_aug = feat_path.split('/')
    name_aug = temp_aug[-1]
    if name_aug == '':
        name_aug = temp_aug[-2]

    fw_txt = experiments + '/' + name_aug + '.txt'
    fw = open(fw_txt, 'w')
    for i in range(len(lines)):
        fw.write(feat_path + '/' + lines[i][0] + '.logmel')
        fw.write(' ' + lines[i][1] + '\n')

    fw.close()

    return fw_txt
        

def frequency_masking(mel_spectrogram, frequency_masking_para = 13, frequency_mask_num = 1):
    fbank_size = mel_spectrogram.shape

    for i in range(frequency_mask_num):
        f = random.randrange(0, frequency_masking_para)
        f0 = random.randrange(0, fbank_size[0] - f)

        if (f0 == f0 + f):
            continue

        mel_spectrogram[f0:(f0+f),:] = 0
    return mel_spectrogram

   
def time_masking(mel_spectrogram, time_masking_para = 40, time_mask_num = 1):
    fbank_size = mel_spectrogram.shape

    for i in range(time_mask_num):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, fbank_size[1] - t)

        if (t0 == t0 + t):
            continue

        mel_spectrogram[:, t0:(t0+t)] = 0
    return mel_spectrogram


def cmvn(data):
    shape = data.shape
    eps = 2**-30
    for i in range(shape[0]):
        utt = data[i].squeeze().T
        mean = np.mean(utt, axis=0)
        utt = utt - mean
        std = np.std(utt, axis=0)
        utt = utt / (std + eps)
        utt = utt.T
        data[i] = utt.reshape((utt.shape[0], utt.shape[1], 1))
    return data


def frequency_label(num_sample, num_frequencybins, num_timebins):

    data = np.arange(num_frequencybins, dtype='float32').reshape(num_frequencybins, 1) / num_frequencybins
    data = np.broadcast_to(data, (num_frequencybins, num_timebins))
    data = np.broadcast_to(data, (num_sample, num_frequencybins, num_timebins))
    data = np.expand_dims(data, -1)
    
    return data