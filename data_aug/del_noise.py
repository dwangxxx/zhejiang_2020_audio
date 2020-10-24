import os
import glob
import sys
import subprocess

file_path = '/data/dengwang/audio/data/testB/*.wav'
des_path = '/data/dengwang/audio/data/testB_clean/'
names = glob.glob(file_path)

cnt = 1
for name in names:
    print(cnt, name)
    cnt += 1
    cmd1 = 'sox ' + name + ' -n noiseprof noise.prof'
    cmd2 = 'sox ' + name + ' ' + os.path.join(des_path, name.split('/')[-1])  + ' noisered noise.prof 0.21'
    os.system(cmd1)
    os.system(cmd2)
    #subprocess.Popen(cmd1, shell=True)
    #subprocess.Popen(cmd2, shell=True)