import os
import numpy as np
import scipy.io
import pandas as pd
import librosa
import pickle
import soundfile as sound
from multiprocessing import Pool
from itertools import islice
import random

overwrite = True

txt_file = 'audio_add.txt'
output_path = '/data/dengwang/audio/data/train_feature_clean'
feature_type = 'logmel'
folder_name = "/data/dengwang/audio/data/split_train_clean/"

sr = 44100
duration = 10
num_freq_bin = 128
num_fft = 2048
hop_length = int(num_fft / 2)
num_time_bin = int(np.ceil(duration * sr / hop_length))
num_channel = 1


if not os.path.exists(output_path):
    os.makedirs(output_path)


# label_dict = dict(airport = 0, bus = 1, metro = 2, metro_station = 3, park = 4, public_square = 5, shopping_mall = 6, street_pedestrian = 7, street_traffic = 8, tram = 9)
# label_dict = {'cq_7461': 0, 'znn_3014': 1, 'wlq_6714': 2, 'zzg_6647': 3, 'xx_0148': 4, 'zzw_6479': 5, 'zxd_3476': 6, 'whl_1010': 7, 'wb_4678': 8, 'yxj_6671': 9, 'ws_0314': 10, 'lww_1346': 11, 'gll_6412': 12, 'wyy_7741': 13, 'wjx_7974': 14, 'ylg_1435': 15, 'zym_9745': 16, 'jcx_0316': 17, 'cl_4738': 18, 'zy_3167': 19, 'gyf_3014': 20, 'zyl_6677': 21, 'zym_0137': 22, 'hml_3467': 23, 'ht_0145': 24, 'cjh_6794': 25, 'lqx_9746': 26, 'lmx_9714': 27, 'zwq_4476': 28, 'wyh_1973': 29, 'czh_6014': 30, 'rjn_0346': 31, 'wzy_3121': 32, 'cy_3658': 33, 'cl_4732': 34, 'ypp_3746': 35, 'gyx_3105': 36, 'zfj_6741': 37, 'zq_9742': 38, 'zjp_7843': 39, 'cmk_4613': 40, 'gzx_0348': 41, 'lh_1034': 42, 'wll_3679': 43, 'wc_3014': 44, 'jyd_1031': 45, 'wyj_9746': 46, 'wx_3476': 47, 'hxr_3014': 48, 'ctj_6713': 49, '1': 50}
label_dict = {'cej_2304': 0, 'zy_2371': 1, 'yge_6705': 2, 'ctf_0172': 3, 'dgl_3846': 4, 'dsw_3607': 5, 'sj_0269': 6, 'rdk_0724': 7, 'rq_7342': 8, 'sl_9052': 9, 'abk_1049': 10, 'qf_4356': 11, 'da_6092': 12, 'kjh_7219': 13, 'bz_3269': 14, 'pcy_6375': 15, 'cbe_9478': 16, 'af_8572': 17, 'yl_7619': 18, 'ha_5413': 19, 'jta_0836': 20, 'zc_1804': 21, 'wk_7215': 22, 'eqa_3472': 23, 'pbs_0752': 24, 'eqg_3205': 25, 'glp_0974': 26, 'pyh_0001': 27, 'bfh_8971': 28, 'zsq_0001': 29, 'kyd_0153': 30, 'hsd_0642': 31, 'wsc_4963': 32, 'dq_1348': 33, 'zd_1689': 34, 'rs_5932': 35, 'ts_8039': 36, 'bly_1354': 37, 'bq_8369': 38, 'lta_2934': 39, 'ah_4729': 40, 'wrb_1378': 41, 'hcs_4783': 42, 'aq_3792': 43, 'tr_9178': 44, 'czf_7398': 45, 'yz_2984': 46, 'kr_6257': 47, 'jf_4635': 48, 'kpl_2536': 49, 'qs_8612': 50, 'kp_9837': 51, '1': 52}


def class_sort():
    class_list = []
    for i in range(len(label_dict)):
        ap = []
        class_list.append(ap)
    with open(txt_file, 'r') as txt_r:
        lines = txt_r.readlines()
        for line in lines[:]:
            file_name = line.strip().split(' ')[0]
            label = line.strip().split(' ')[1]
            class_list[label_dict[label]].append(file_name)
    return class_list


def data_add():
    i = 0
    sample_rate = 44100
    class_list = class_sort()
    for label in class_list:
        length = len(label)
        for file in label:
            y, sr = librosa.load(folder_name + file, mono = True, sr = sample_rate)
            num = random.randint(0, length - 1)
            while file == label[num]:
                num = random.randint(0, length - 1)
            f1, f2 = random.uniform(0.5, 1), random.uniform(0.5, 1)
            y2, _ = librosa.load(folder_name + label[num], mono = True, sr = sample_rate)
            stereo = y * f1 + y2 * f2

            logmel_data = np.zeros((num_freq_bin, num_time_bin, num_channel), 'float32')
            logmel_data[:,:,0]= librosa.feature.melspectrogram(stereo[:], sr = sr, n_fft = num_fft, \
                            hop_length = hop_length, n_mels = num_freq_bin, fmin = 0.0, fmax = sr / 2, htk = True, norm = None)

            logmel_data = np.log(logmel_data + 1e-8)

            feat_data = logmel_data
            feat_data = (feat_data - np.min(feat_data)) / (np.max(feat_data) - np.min(feat_data))
    
            feature_data = {'feat_data': feat_data,}

            cur_file_name = output_path + '/' + file.split('.')[0] + '_add.' + feature_type
            print(i, cur_file_name)
            i += 1

            pickle.dump(feature_data, open(cur_file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    data_add()