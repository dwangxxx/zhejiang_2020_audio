src_path = '/data/dengwang/audio/baseline/result/result.txt'
test_txt = '/data/dengwang/audio/data/testB/test.txt'
des_path = '/data/dengwang/audio/baseline/result/result_merge.txt'

test_names = open(test_txt, 'r').readlines()
src_res = open(src_path, 'r').readlines()
des_res = open(src_path, 'w')
for test_name in test_names:
    res_dict = []
    for 