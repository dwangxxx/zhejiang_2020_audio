import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np
import keras
import tensorflow
from keras.optimizers import SGD

import sys
sys.path.append("..")
from utils import *
from funcs import *

from network import model_resnet
from training_functions import *

from tensorflow import ConfigProto
from tensorflow import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config = config)

train_txt = '../train_val/train_full_B.txt'
val_txt = '../train_val/val_full_B.txt'

feat_path = '/data/dengwang/audio/data/train_feature_clean'

experiments = '/data/dengwang/audio/baseline/models/class50/cnn_3/model_cnn_3_full_clean_B'

if not os.path.exists(experiments):
    os.makedirs(experiments)

train_all = generate_train_txt(train_txt, feat_path, experiments)

num_audio_channels = 1
num_freq_bin = 128
num_classes = 52
max_lr = 0.1
batch_size = 16
num_epochs = 100
mixup_alpha = 0.4
crop_length = 400
sample_num = len(open(train_all, 'r').readlines())


data_val, y_val = load_data_val(feat_path, val_txt, num_freq_bin, 'logmel')
data_deltas_val = deltas(data_val)
data_deltas_deltas_val = deltas(data_deltas_val)
data_val = np.concatenate((data_val[:, :, 4:-4, :], data_deltas_val[:, :, 2:-2, :], data_deltas_deltas_val), axis = -1)
y_val = keras.utils.to_categorical(y_val, num_classes)

model = model_resnet(num_classes,
                     input_shape = [num_freq_bin, None, 3 * num_audio_channels], 
                     num_filters = 48,
                     wd = 1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer = SGD(lr = max_lr,decay = 1e-6, momentum = 0.9, nesterov = False),
              metrics = ['accuracy'])

model.summary()

cos_scheduler = WarmRestart(nbatch = np.ceil(sample_num / batch_size), Tmult = 2,
                            initial_lr = max_lr, min_lr = max_lr * 1e-3,
                            epochs_restart = [3.0, 7.0, 15.0, 31.0, 63.0, 127.0, 255.0, 511.0]) 

save_path = experiments + "/model-{epoch:02d}-{val_acc:.4f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor = 'val_acc', verbose = 1, save_best_only = False, mode = 'max')
callbacks = [cos_scheduler, checkpoint]

# Due to the memory limitation, in the training stage we split the training data
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
