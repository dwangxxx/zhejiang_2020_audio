import os

import numpy as np
import h5py
import scipy.io
import pandas as pd

import librosa
import soundfile as sound
import keras
import tensorflow

from .utils import *
from .funcs import *

from sklearn.metrics import log_loss
import os
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

def get_result_2(txt_file):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config = config)

    test_txt = txt_file
    des_path = '/data/dengwang/audio/baseline/result/class50.txt'
    feat_path = '/data/dengwang/audio/data/feature_round2'

    # 模型文件
    """ model_path = '/data/dengwang/audio/baseline/models/class2/cnn_1/model_cnn_1_full/model-50-1.0000.hdf5'
    model_path2 = '/data/dengwang/audio/baseline/models/class2/cnn_2/model_cnn_2_full/model-25-1.0000.hdf5'
    model_path3 = '/data/dengwang/audio/baseline/models/class2/cnn_3/model_cnn_3_full/model-50-1.0000.hdf5' """
    #model_path4 = '/data/dengwang/audio/baseline/class2/mobnet/exp_mobnet_2class/model-20-1.0000.hdf5'

    model_path = '/data/dengwang/audio/baseline/models/class2/cnn_1/model_cnn_1_clean_no_add_B/model-30-1.0000.hdf5'
    model_path2 = '/data/dengwang/audio/baseline/models/class2/cnn_2/model_cnn_2_clean_no_add_B/model-30-1.0000.hdf5'
    model_path3 = '/data/dengwang/audio/baseline/models/class2/cnn_3/model_cnn_3_clean_no_add_B/model-60-0.9998.hdf5'
    model_path4 = '/data/dengwang/audio/baseline/models/class2/cnn_1/model_cnn_1_no_pitch_B/model-40-1.0000.hdf5'
    model_path5 = '/data/dengwang/audio/baseline/models/class2/cnn_2/model_cnn_2_no_pitch_B/model-20-1.0000.hdf5'
    model_path6 = '/data/dengwang/audio/baseline/models/class2/cnn_2/model_cnn_2_clean_no_add_B/model-09-0.9998.hdf5'

    num_freq_bin = 128
    num_classes = 2

    # 载入测试数据
    data_val = load_data_test(feat_path, test_txt, 'logmel')
    # 计算一阶差分deltas
    data_deltas_val = deltas(data_val)
    # 计算二阶差分deltas-deltas
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:, :, 4:-4, :], data_deltas_val[:, :, 2:-2, :], data_deltas_deltas_val), axis = -1)

    # 推理
    best_model = keras.models.load_model(model_path)
    best_model2 = keras.models.load_model(model_path2)
    #best_model3 = keras.models.load_model(model_path3)
    best_model4 = keras.models.load_model(model_path4)
    best_model5 = keras.models.load_model(model_path5)
    #best_model6 = keras.models.load_model(model_path6)

    preds = best_model.predict(data_val)
    preds2 = best_model2.predict(data_val)
    #preds3 = best_model3.predict(data_val)
    preds4 = best_model4.predict(data_val)
    preds5 = best_model5.predict(data_val)
    #preds6 = best_model6.predict(data_val)

    preds = np.add(preds ,preds2)
    #preds = np.add(preds, preds3)
    preds = np.add(preds, preds4)
    preds = np.add(preds, preds5)
    #preds = np.add(preds, preds6)
    y_pre = np.argmax(preds, axis = 1)
    names = open(test_txt, 'r').readlines()
    wavpath = []
    for n in names:
        name = n.strip()
        wavpath.append(name)
    res_f = open(des_path, 'w')
    # 如果测试结果中是非伪造数据，则送入50分类模型进行测试
    for i in range(len(y_pre)):
        # 非伪造数据
        if y_pre[i] == 0:
            res_f.write(wavpath[i] + '\n')
    np.savetxt("/data/dengwang/audio/baseline/result/class2_result.txt", preds, fmt = '%.7f')