import numpy as np
import pandas as pd
import pickle
import os

def load_data_val(feat_path, txt_path, feat_dim, file_type):
    with open(txt_path, 'r') as text_file:
        lines = text_file.readlines()
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].strip().split(' ')
            lines[idx][0] = lines[idx][0].split('.')[0]

        lines = [elem for elem in lines if elem != ['']]
        labels = []
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
            labels.append(int(lines[idx][-1]))
        label_info = np.array(lines)

        feat_mtx = []
        # 获取logmel data
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.logmel' 
            with open(filepath,'rb') as f:
                temp = pickle.load(f, encoding = 'latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels


def load_data_train(feat_path, csv_path, feat_dim, idxlines, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.readlines()
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].strip().split(' ')
            lines[idx][0] = lines[idx][0].split('.')[0]

        lines = [lines[i] for i in idxlines]
        lines = [elem for elem in lines if elem != ['']]
        labels = []
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
            labels.append(lines[idx][-1])
        label_info = np.array(lines)
        
        feat_mtx = []
        for [filename, labnel] in label_info:
            # filepath = feat_path + '/' + filename + '.logmel'
            filepath = filename + '.logmel'
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding='latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels

# 获取测试数据，只返回特征
def load_data_test(feat_path, csv_path, file_type):
    with open(csv_path, 'r') as text_file:
        lines = text_file.readlines()
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].strip().split(' ')
            lines[idx][0] = lines[idx][0].split('.')[0]

        lines = [elem for elem in lines if elem != ['']]
        label_info = np.array(lines)
        
        feat_mtx = []
        for filename in label_info:
            # filepath = feat_path + '/' + filename + '.logmel'
            filepath = feat_path + '/' + filename[0] + '.logmel'
            with open(filepath,'rb') as f:
                temp = pickle.load(f, encoding = 'latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx


def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out
