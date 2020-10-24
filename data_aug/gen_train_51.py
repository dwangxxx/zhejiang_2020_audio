import glob

des_path = '/tmp1/home/dw/audio/baseline/class50/evaluation_setup/val_51_full.txt'
des_f = open(des_path, 'w')
with open('/tmp1/home/dw/audio/baseline/class2/evaluation_setup/val.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        names = line.split(' ')[0].split('_')
        if line.split(' ')[1] == '0\n':
            des_f.write(line)