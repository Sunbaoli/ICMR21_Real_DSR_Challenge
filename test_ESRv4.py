from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_eval_set
from functools import reduce
import scipy.io as sio
import time
import cv2
from collections import OrderedDict

###mad-rmse
from glob import glob
import numpy as np
import math
from PIL import Image


"""
Process of modification

1 Type of model
import Net,model_type,if model_type,model(),opt.upscale_factor

2 Weights
opt.weights,opt.gpus

3 Dataset
opt.input_dir,opt.test_dataset

4 Output
opt.output,prediction
"""

# from dbpn8 import Net as DBPN8
# from ddcnet_x8 import Net as DDCX8
# from dbpn8 import Net as DBPN8
# from dbpn8_parallel import Net as DBPN8_P
# from ddcnet_pretrainedEn import Net as DDCE
# from pmpanet_x2 import Net as PMBAX2

from edsrx2_v3 import Net as edsrx2_v3
from edsrx2_v4 import Net as edsrx2_v4

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# Calculate
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--gpus', default=2, type=float, help='number of gpu')
parser.add_argument('--gpu', type=int, default=0, help='the number of using gpu')
# Networks
parser.add_argument('--model_type', type=str, default='edsrx2_v4')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--weights', default="./ubunEDSRx2v4edsrx2_v4__epoch_79.pth", help='sr pretrained base model')
# Data
parser.add_argument('--data_dir', type=str, default='/home/ubuntu/users/sunbaoli/data/', help='root of data')
parser.add_argument('--test_dataset', type=str, default='validate20/', help='')
parser.add_argument('--output_dir', default='results_v4_1_n/', help='Location to save the prediction')

parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

def eval():
    model.eval()
    torch.set_grad_enabled(False)
    
    with torch.no_grad():
        for batch in testing_data_loader:
            input_rgb, input, name, minmax = Variable(batch[0]),Variable(batch[1]), batch[2], batch[3]
            
            if cuda:
                input = input.cuda(gpus_list[opt.gpu])
                input_rgb = input_rgb.cuda(gpus_list[opt.gpu])

            "Output"
            t0 = time.time()
            prediction = model(input, input_rgb)[1]
            t1 = time.time()

            # print(result.shape, 'rrrrrrrrrrrrr')
            # print(prediction)

            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))

            save_img(prediction.cpu().numpy(), name[0], minmax)

def save_img(img, img_name, minmax):
    # save_img = img.squeeze().clamp(0, 1).numpy()
    # img = img.squeeze().numpy()
    
    "remove normalization"
    maxx = minmax[1].numpy()
    minn = minmax[0].numpy()
    img_normal = img[0, 0]

    output = (img_normal*(maxx-minn)+minn).astype(np.uint16)

    "remove very high value"
    temp = output > 60000
    output[temp] = img_normal[temp]
    save_dir=opt.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = './'+save_dir + img_name.split('/')[-1]
    # print(save_fn, 'sssssss')
    cv2.imwrite(save_fn,output)
    # print('iiiiiiiiiiiiii')

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


print('===> Loading datasets')
test_set = get_eval_set(opt.data_dir, opt.test_dataset, opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model')
if opt.model_type == 'edsrx2_v3':
    model = edsrx2_v3(num_channels=1, base_filter=64, scale_factor=opt.upscale_factor)
elif opt.model_type == 'edsrx2_v4':
    model = edsrx2_v4(num_channels=1, base_filter=64, scale_factor=opt.upscale_factor)
###
flag=1 # for loading SR model

"Single GPU testing"
if cuda:
    model = model.cuda(gpus_list[opt.gpu])

if os.path.exists(opt.weights):
    model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
    flag=0
    print('<-------------- Pre-trained SR model is loaded. -------------->')

"Multi-GPUs Testing"
# if cuda:
#     model = nn.DataParallel(model).cuda(gpus_list[opt.gpu]) #use parallel

# if os.path.exists(opt.weights):
#     #model= torch.load(opt.model, map_location=lambda storage, loc: storage)
#     # print("11111111111111111111")
#     model.load_state_dict(torch.load(opt.weights, map_location=lambda storage, loc: storage))
#     flag=0
#     print('<--------------Pre-trained SR model is loaded.-------------->')

# if flag == 1:
#     print('!-------------- Cannot load pre-trained model! --------------!')



##Eval Start!!!!
eval()

print('<-------------- Writing results to {} -------------->'.format(opt.output_dir))