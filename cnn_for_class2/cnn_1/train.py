import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from network import model_fcnn
from training_functions import *

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

train_txt = '../train_val/train_full_B.txt'
val_txt = '../train_val/val_full_B.txt'

feat_path = '/data/dengwang/audio/data/train_feature_clean'

experiments = '/data/dengwang/audio/baseline/models/class2/cnn_1/model_cnn_1_full_clean_B'

if not os.path.exists(experiments):
    os.makedirs(experiments)

train_all = generate_train_txt(train_txt, feat_path, experiments)
num_audio_channels = 1
num_freq_bin = 128
# 合成数据和非合成数据
num_classes = 2
max_lr = 0.1
batch_size = 32
num_epochs = 100
mixup_alpha = 0.4
crop_length = 400
# 获取采样数
sample_num = len(open(train_all, 'r').readlines())

# 获取验证集的数据特征
data_val, y_val = load_data_val(feat_path, val_txt, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:, :, 4:-4, :],data_deltas_val[:, :, 2:-2, :],data_deltas_deltas_val), axis = -1)
print(y_val)
y_val = keras.utils.to_categorical(y_val, num_classes)
print(y_val)

model = model_fcnn(num_classes, input_shape = [num_freq_bin, None, 3 * num_audio_channels], num_filters = [48, 96, 192], wd = 0)

model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr = max_lr, decay = 1e-6, momentum = 0.9, nesterov = False),
              metrics=['accuracy'])

model.summary()

# warmRestart，在epoch_restart处使用初始学习率
cos_scheduler = WarmRestart(nbatch = np.ceil(sample_num / batch_size), Tmult = 2,
                              initial_lr = max_lr, min_lr = max_lr * 1e-4,
                              epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0]) 
save_path = experiments + "/model-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor = 'val_acc', verbose = 1, save_best_only = False, mode = 'max')
callbacks = [cos_scheduler, checkpoint]

train_data_generator = Data_Generator(feat_path, train_all, num_freq_bin,
                              batch_size = batch_size,
                              alpha = mixup_alpha,
                              crop_length = crop_length, splitted_num = 15)()


history = model.fit_generator(train_data_generator,
                              validation_data = (data_val, y_val),
                              epochs = num_epochs, 
                              verbose = 1, 
                              workers = 4,
                              max_queue_size = 100,
                              callbacks = callbacks,
                              steps_per_epoch = np.ceil(sample_num / batch_size)
                              )