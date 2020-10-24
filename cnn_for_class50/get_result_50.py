import os
import numpy as np
import h5py
import scipy.io
import pandas as pd
import librosa
import soundfile as sound
import keras
import tensorflow
from sklearn.metrics import confusion_matrix
from .utils import *
from .funcs import *
from sklearn.metrics import log_loss
import os
from tensorflow import ConfigProto
from tensorflow import InteractiveSession

def get_result_50(txt_file):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config = config)

    test_txt = txt_file
    feat_path = '/data/dengwang/audio/data/feature_round2'
 
    model_path = '/data/dengwang/audio/baseline/models/class50/cnn_1/model_cnn_1_clean_B/model-82-1.0000.hdf5'
    model_path2 = '/data/dengwang/audio/baseline/models/class50/cnn_2/model_cnn_2_clean_B/model-50-1.0000.hdf5'
    model_path3 = '/data/dengwang/audio/baseline/models/class50/cnn_3/model_cnn_3_no_pitch_B/model-30-1.0000.hdf5'
    model_path4 = '/data/dengwang/audio/baseline/models/class50/cnn_1/model_cnn_1_full_clean_B/model-30-1.0000.hdf5'
    model_path5 = '/data/dengwang/audio/baseline/models/class50/cnn_2/model_cnn_2_clean_B/model-60-1.0000.hdf5'
    model_path6 = '/data/dengwang/audio/baseline/models/class50/cnn_2/model_cnn_2_clean_B/model-40-1.0000.hdf5'
    model_path7 = '/data/dengwang/audio/baseline/models/class50/cnn_1/model_cnn_1_full_clean_B/model-25-1.0000.hdf5'
    model_path8 = '/data/dengwang/audio/baseline/models/class50/cnn_2/model_cnn_2_clean_B/model-66-1.0000.hdf5'

    num_freq_bin = 128

    data_val = load_data_test(feat_path, test_txt, 'logmel')
    data_deltas_val = deltas(data_val)
    data_deltas_deltas_val = deltas(data_deltas_val)
    data_val = np.concatenate((data_val[:, :, 4:-4, :], data_deltas_val[:, :, 2:-2, :], data_deltas_deltas_val), axis = -1)

    best_model = keras.models.load_model(model_path)
    best_model2 = keras.models.load_model(model_path2)
    best_model3 = keras.models.load_model(model_path3)
    best_model4 = keras.models.load_model(model_path4)
    best_model5 = keras.models.load_model(model_path5)
    best_model6 = keras.models.load_model(model_path6)
    #best_model7 = keras.models.load_model(model_path7)
    best_model8 = keras.models.load_model(model_path8)

    preds = best_model.predict(data_val)
    preds2 = best_model2.predict(data_val)
    preds3 = best_model3.predict(data_val)
    preds4 = best_model4.predict(data_val)
    preds5 = best_model5.predict(data_val)
    preds6 = best_model6.predict(data_val)
    #preds7 = best_model7.predict(data_val)
    preds8 = best_model8.predict(data_val)

    preds = np.add(preds ,preds2)
    preds = np.add(preds, preds3)
    preds = np.add(preds, preds4)
    preds = np.add(preds, preds5)
    preds = np.add(preds, preds6)
    #preds = np.add(preds, preds7)
    #preds = np.add(preds, preds8)

    np.savetxt("/data/dengwang/audio/baseline/result/class50_result.txt", preds, fmt = '%.7f')