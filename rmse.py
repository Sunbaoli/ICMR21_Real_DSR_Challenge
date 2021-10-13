from __future__ import print_function
import argparse
import os
import scipy.io as sio
import time
import cv2
from collections import OrderedDict

###mad-rmse
from glob import glob
import numpy as np
import math
from PIL import Image
from imageio import imread

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--output_dir', default='Results_EDSRx2/', help='The location of predictions')
opt = parser.parse_args()

print('<-------------- Calculating RMSE of {} -------------->'.format(opt.output_dir))

output=glob(opt.output_dir+'*.png')
output = sorted(output)
print(len(output))

rmse=0

for i in range(len(output)):
    oo=imread(output[i]).astype(np.float32)
    print(oo.shape)
    name = output[i].split('/')[-1].split('.')[0].split('_')[0]
    gg=imread('../data/test/' + name + '/' + name + '_HR_gt.png').astype(np.float32)
    
    gg = gg[6:-6, 6:-6]
    oo = oo[6:-6, 6:-6]
    gg = gg/10.0
    oo = oo/10.0

    res=np.sqrt(np.mean(np.power(gg - oo, 2)))
    rmse+=res
    # print(GT[i],output[i],res)
print(rmse/len(output))