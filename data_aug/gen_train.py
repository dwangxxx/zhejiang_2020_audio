import glob

des_path = '/tmp1/home/dw/audio/baseline/class2/evaluation_setup/half_val.txt'
des_f = open(des_path, 'w')
with open('/tmp1/home/dw/audio/baseline/class2/evaluation_setup/val.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        names = line.split(' ')[0].split('_')
        if (len(names) == 3):
            if (int(names[1]) < 10):
                des_f.write(line)
        else:
            if (int(names[1].split('.')[0]) < 10):
                des_f.write(line)

"""
path = '/media/dengwang/225A6D42D4FA828F/audio/data/features_train/features_train/logmel128_scaled_full/*.logmel'
files = glob.glob(path)
for f in files:
    des_f.write(f.split('/')[-1].split('.')[0] + '.wav' + ' 0\n') """