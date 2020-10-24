import os
import glob

files = glob.glob('/tmp1/home/dw/audio/data/features_train/feature/*.logmel')
for f in files:
    if os.path.getsize(f) <= 0:
        print(f)