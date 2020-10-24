from cnn_for_class2 import get_result_2
from cnn_for_class50 import get_result_50
import numpy as np 
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

test_txt_2 = '/data/dengwang/audio/data/test_split/test.txt'
# 2分类
get_result_2.get_result_2(test_txt_2)

test_txt_50 = '/data/dengwang/audio/baseline/result/class50.txt'
# 50分类
get_result_50.get_result_50(test_txt_50)

filename = open(test_txt_2, 'r').readlines()
wavfiles_2 = []
for file in filename:
    name = file.strip()
    wavfiles_2.append(name)

filename = open(test_txt_50, 'r').readlines()
wavfiles_50 = []
for file in filename:
    name = file.strip()
    wavfiles_50.append(name)

result = open('result/result.txt', 'w')
result_2 = np.loadtxt('result/class2_result.txt')
result_50_arr = np.loadtxt('result/class50_result.txt')
result_50_sec = []

result_2 = np.argmax(result_2, axis = 1)
result_50 = np.argmax(result_50_arr, axis = 1)

for i in range(len(result_50_arr)):
    temp = []
    for j in range(len(result_50_arr[i])):
        if j != result_50[i]:
            temp.append(result_50_arr[i][j])
    result_50_sec.append(temp)
result_50_sec_max = np.argmax(np.array(result_50_sec), axis = 1)

# label_dict = {'cq_7461': 0, 'znn_3014': 1, 'wlq_6714': 2, 'zzg_6647': 3, 'xx_0148': 4, 'zzw_6479': 5, 'zxd_3476': 6, 'whl_1010': 7, 'wb_4678': 8, 'yxj_6671': 9, 'ws_0314': 10, 'lww_1346': 11, 'gll_6412': 12, 'wyy_7741': 13, 'wjx_7974': 14, 'ylg_1435': 15, 'zym_9745': 16, 'jcx_0316': 17, 'cl_4738': 18, 'zy_3167': 19, 'gyf_3014': 20, 'zyl_6677': 21, 'zym_0137': 22, 'hml_3467': 23, 'ht_0145': 24, 'cjh_6794': 25, 'lqx_9746': 26, 'lmx_9714': 27, 'zwq_4476': 28, 'wyh_1973': 29, 'czh_6014': 30, 'rjn_0346': 31, 'wzy_3121': 32, 'cy_3658': 33, 'cl_4732': 34, 'ypp_3746': 35, 'gyx_3105': 36, 'zfj_6741': 37, 'zq_9742': 38, 'zjp_7843': 39, 'cmk_4613': 40, 'gzx_0348': 41, 'lh_1034': 42, 'wll_3679': 43, 'wc_3014': 44, 'jyd_1031': 45, 'wyj_9746': 46, 'wx_3476': 47, 'hxr_3014': 48, 'ctj_6713': 49}
label_dict = {'cej_2304': 0, 'zy_2371': 1, 'yge_6705': 2, 'ctf_0172': 3, 'dgl_3846': 4, 'dsw_3607': 5, 'sj_0269': 6, 'rdk_0724': 7, 'rq_7342': 8, 'sl_9052': 9, 'abk_1049': 10, 'qf_4356': 11, 'da_6092': 12, 'kjh_7219': 13, 'bz_3269': 14, 'pcy_6375': 15, 'cbe_9478': 16, 'af_8572': 17, 'yl_7619': 18, 'ha_5413': 19, 'jta_0836': 20, 'zc_1804': 21, 'wk_7215': 22, 'eqa_3472': 23, 'pbs_0752': 24, 'eqg_3205': 25, 'glp_0974': 26, 'pyh_0001': 27, 'bfh_8971': 28, 'zsq_0001': 29, 'kyd_0153': 30, 'hsd_0642': 31, 'wsc_4963': 32, 'dq_1348': 33, 'zd_1689': 34, 'rs_5932': 35, 'ts_8039': 36, 'bly_1354': 37, 'bq_8369': 38, 'lta_2934': 39, 'ah_4729': 40, 'wrb_1378': 41, 'hcs_4783': 42, 'aq_3792': 43, 'tr_9178': 44, 'czf_7398': 45, 'yz_2984': 46, 'kr_6257': 47, 'jf_4635': 48, 'kpl_2536': 49, 'qs_8612': 50, 'kp_9837': 51, '1': 52}
label = {}
for l in label_dict:
    label[label_dict[l]] = l


# 生成提交文件
for i in range(len(result_2)):
    # 伪造数据
    if result_2[i] == 1:
        result.write(wavfiles_2[i] + ' ')
        result.write('1')
        result.write(' ' + '0' + '\n')


# 非伪造数据
for i in range(len(result_50)):
    # 如果置信度不高，则将其辨别为伪造数据
    if result_50_arr[i][result_50[i]] < 0.08:
        result.write(wavfiles_50[i] + ' ')
        result.write('1')
        result.write(' ' + '0' + '\n')
    else:
        result.write(wavfiles_50[i] + ' ')
        result.write('0')
        result.write(' ' + label[result_50[i]] + '\n')