import numpy as np
import pandas as pd
import pickle
import os

# 载入验证数据
def load_data_val(feat_path, csv_path, feat_dim, file_type):
    # label_dict = {'cq_7461': 0, 'znn_3014': 1, 'wlq_6714': 2, 'zzg_6647': 3, 'xx_0148': 4, 'zzw_6479': 5, 'zxd_3476': 6, 'whl_1010': 7, 'wb_4678': 8, 'yxj_6671': 9, 'ws_0314': 10, 'lww_1346': 11, 'gll_6412': 12, 'wyy_7741': 13, 'wjx_7974': 14, 'ylg_1435': 15, 'zym_9745': 16, 'jcx_0316': 17, 'cl_4738': 18, 'zy_3167': 19, 'gyf_3014': 20, 'zyl_6677': 21, 'zym_0137': 22, 'hml_3467': 23, 'ht_0145': 24, 'cjh_6794': 25, 'lqx_9746': 26, 'lmx_9714': 27, 'zwq_4476': 28, 'wyh_1973': 29, 'czh_6014': 30, 'rjn_0346': 31, 'wzy_3121': 32, 'cy_3658': 33, 'cl_4732': 34, 'ypp_3746': 35, 'gyx_3105': 36, 'zfj_6741': 37, 'zq_9742': 38, 'zjp_7843': 39, 'cmk_4613': 40, 'gzx_0348': 41, 'lh_1034': 42, 'wll_3679': 43, 'wc_3014': 44, 'jyd_1031': 45, 'wyj_9746': 46, 'wx_3476': 47, 'hxr_3014': 48, 'ctj_6713': 49, '0': 50}
    label_dict = {'cej_2304': 0, 'zy_2371': 1, 'yge_6705': 2, 'ctf_0172': 3, 'dgl_3846': 4, 'dsw_3607': 5, 'sj_0269': 6, 'rdk_0724': 7, 'rq_7342': 8, 'sl_9052': 9, 'abk_1049': 10, 'qf_4356': 11, 'da_6092': 12, 'kjh_7219': 13, 'bz_3269': 14, 'pcy_6375': 15, 'cbe_9478': 16, 'af_8572': 17, 'yl_7619': 18, 'ha_5413': 19, 'jta_0836': 20, 'zc_1804': 21, 'wk_7215': 22, 'eqa_3472': 23, 'pbs_0752': 24, 'eqg_3205': 25, 'glp_0974': 26, 'pyh_0001': 27, 'bfh_8971': 28, 'zsq_0001': 29, 'kyd_0153': 30, 'hsd_0642': 31, 'wsc_4963': 32, 'dq_1348': 33, 'zd_1689': 34, 'rs_5932': 35, 'ts_8039': 36, 'bly_1354': 37, 'bq_8369': 38, 'lta_2934': 39, 'ah_4729': 40, 'wrb_1378': 41, 'hcs_4783': 42, 'aq_3792': 43, 'tr_9178': 44, 'czf_7398': 45, 'yz_2984': 46, 'kr_6257': 47, 'jf_4635': 48, 'kpl_2536': 49, 'qs_8612': 50, 'kp_9837': 51, '1': 52}
    with open(csv_path, 'r') as text_file:
        lines = text_file.readlines()
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].strip().split(' ')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        labels = []
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
            labels.append(label_dict[lines[idx][-1]])
        label_info = np.array(lines)

        feat_mtx = []
        for [filename, labnel] in label_info:
            filepath = feat_path + '/' + filename + '.logmel' 
            with open(filepath,'rb') as f:
                temp=pickle.load(f, encoding = 'latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx, labels


# 载入训练数据
def load_data_train(feat_path, csv_path, feat_dim, idxlines, file_type):
    # label_dict = {'cq_7461': 0, 'znn_3014': 1, 'wlq_6714': 2, 'zzg_6647': 3, 'xx_0148': 4, 'zzw_6479': 5, 'zxd_3476': 6, 'whl_1010': 7, 'wb_4678': 8, 'yxj_6671': 9, 'ws_0314': 10, 'lww_1346': 11, 'gll_6412': 12, 'wyy_7741': 13, 'wjx_7974': 14, 'ylg_1435': 15, 'zym_9745': 16, 'jcx_0316': 17, 'cl_4738': 18, 'zy_3167': 19, 'gyf_3014': 20, 'zyl_6677': 21, 'zym_0137': 22, 'hml_3467': 23, 'ht_0145': 24, 'cjh_6794': 25, 'lqx_9746': 26, 'lmx_9714': 27, 'zwq_4476': 28, 'wyh_1973': 29, 'czh_6014': 30, 'rjn_0346': 31, 'wzy_3121': 32, 'cy_3658': 33, 'cl_4732': 34, 'ypp_3746': 35, 'gyx_3105': 36, 'zfj_6741': 37, 'zq_9742': 38, 'zjp_7843': 39, 'cmk_4613': 40, 'gzx_0348': 41, 'lh_1034': 42, 'wll_3679': 43, 'wc_3014': 44, 'jyd_1031': 45, 'wyj_9746': 46, 'wx_3476': 47, 'hxr_3014': 48, 'ctj_6713': 49, '0': 50}
    label_dict = {'cej_2304': 0, 'zy_2371': 1, 'yge_6705': 2, 'ctf_0172': 3, 'dgl_3846': 4, 'dsw_3607': 5, 'sj_0269': 6, 'rdk_0724': 7, 'rq_7342': 8, 'sl_9052': 9, 'abk_1049': 10, 'qf_4356': 11, 'da_6092': 12, 'kjh_7219': 13, 'bz_3269': 14, 'pcy_6375': 15, 'cbe_9478': 16, 'af_8572': 17, 'yl_7619': 18, 'ha_5413': 19, 'jta_0836': 20, 'zc_1804': 21, 'wk_7215': 22, 'eqa_3472': 23, 'pbs_0752': 24, 'eqg_3205': 25, 'glp_0974': 26, 'pyh_0001': 27, 'bfh_8971': 28, 'zsq_0001': 29, 'kyd_0153': 30, 'hsd_0642': 31, 'wsc_4963': 32, 'dq_1348': 33, 'zd_1689': 34, 'rs_5932': 35, 'ts_8039': 36, 'bly_1354': 37, 'bq_8369': 38, 'lta_2934': 39, 'ah_4729': 40, 'wrb_1378': 41, 'hcs_4783': 42, 'aq_3792': 43, 'tr_9178': 44, 'czf_7398': 45, 'yz_2984': 46, 'kr_6257': 47, 'jf_4635': 48, 'kpl_2536': 49, 'qs_8612': 50, 'kp_9837': 51, '1': 52}
    with open(csv_path, 'r') as text_file:
        lines = text_file.readlines()
        for idx, elem in enumerate(lines):
            lines[idx] = lines[idx].strip().split(' ')
            lines[idx][0] = lines[idx][0].split('/')[-1].split('.')[0]

        labels = []
        lines = [lines[i] for i in idxlines]
        lines = [elem for elem in lines if elem != ['']]
        for idx, elem in enumerate(lines):
            lines[idx][-1] = lines[idx][-1].split('\r')[0]
            labels.append(label_dict[lines[idx][-1]])
        label_info = np.array(lines)
    

        feat_mtx = []
        for [filename, label] in label_info:
            filepath = feat_path + '/' + filename + '.' + 'logmel'
            # filepath = filename + '.' + 'logmel'
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
        for filename in lines:
            # filepath = feat_path + '/' + filename + '.logmel'
            filepath = feat_path + '/' + filename[0] + '.logmel'
            with open(filepath,'rb') as f:
                temp = pickle.load(f, encoding = 'latin1')
                feat_mtx.append(temp['feat_data'])

        feat_mtx = np.array(feat_mtx)

        return feat_mtx


# 计算梯度
def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out