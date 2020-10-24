import glob

test_path = '/data/dengwang/audio/data/final_test_round2/*.wav'
des_path = '/data/dengwang/audio/data/final_test_round2/test.txt'
''' test_path = '/data/dengwang/audio/data/test_f/*.wav'
des_path = '/data/dengwang/audio/data/test_f/test.txt' '''

file_names = glob.glob(test_path)
des_f = open(des_path, 'w')
for name in file_names:
    des_f.write(name.split('/')[-1] + '\n')