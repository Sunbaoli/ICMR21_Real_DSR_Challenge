from __future__ import print_function
import argparse
from math import log10
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import pdb
import socket
import time
import numpy as np
# from PIL import Image

# from ddcnet_nodense_x8 import Net as DDCNDX8
# from ddcnet_x8 import Net as DDCX8
# from dbpn8 import Net as DBPN8
# from ddcnet_parallel import Net as DDCP
# from ddcnet_edgedecouple_v2 import Net as DDCE
# from pmpanet_x4 import Net as pmpanet_x4
# from hrdsrnet_x8v2 import Net as PMBAX4
# from edsrx2_v2 import Net as EDSRx2v2
# from edsrx2_v3 import Net as EDSRx2v3
from models.edsrx2_v4 import Net as EDSRx2v4


# Training Settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
# Device
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--gpus', default=2, type=float, help='number of gpu')
parser.add_argument('--gpu', type=int, default=0, help='the number of using gpu')
# Networks
parser.add_argument('--model_type', type=str, default='EDSRx2v4')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--prefix', default='edsrx2_v4_', help='description of the model for weights')
# Pretrained Model
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--pretrained_sr', default="edsrc2_v3train/ubunEDSRx2v3edsrc2_v3_epoch_109.pth", help='sr pretrained base model')
# Training Strategy
parser.add_argument('--nEpochs', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')

# Data Path
parser.add_argument('--data_dir', type=str, default='/home/ubuntu/users/sunbaoli/data/new_train_patch/', help='root of data')
parser.add_argument('--train_dataset', type=str, default='new_train_patch/', help='1.down-sampled images for input  2.also for the dataset type in DatasetFolder')
parser.add_argument('--test_dir', type=str, default='new_validate40/', help='test data')
# Loading Settings
parser.add_argument('--threads', type=int, default=0, help='number of extra threads for data loader to use')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--patch_size', type=int, default=128, help='Size of cropped HR image')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
# Output
parser.add_argument('--snapshots', type=int, default=2, help='Snapshots for checkpoint and rmse computing')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--out_rmse', default="./rmse_v4.txt", help='path for rmse')

opt = parser.parse_args()
print(opt)


"Obtain device info"
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())


def train(epoch):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        input_rgb, input, target = Variable(batch[0], requires_grad=True), Variable(batch[1], requires_grad=True), Variable(batch[2], requires_grad=True)
        if cuda:
            input_rgb = input_rgb.cuda(gpus_list[opt.gpu])
            input = input.cuda(gpus_list[opt.gpu])
            target = target.cuda(gpus_list[opt.gpu])

        optimizer.zero_grad()
        # print(target.size())

        t0 = time.time()
        result1, result2, result3, result4 = model(input, input_rgb)
        t1 = time.time()

        label_fft1 = torch.rfft(target, signal_ndim=2, normalized=False, onesided=False)
        pred_fft1 = torch.rfft(result3, signal_ndim=2, normalized=False, onesided=False)

        loss = criterion(result1, target) + criterion(result2, target) + criterion(result3, target) + 0.5*criterion(result4, input) + 0.5*criterion(pred_fft1, label_fft1)
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print("===> Epoch[{}]({}/{}): Loss: {:.4f} || Timer: {:.4f} sec.".format(epoch, iteration, len(training_data_loader), loss.item(), (t1 - t0)))

    avg_loss = epoch_loss / len(training_data_loader)
    loss_list.append(avg_loss)
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))


def validate(epoch):
    model.eval()
    # torch.set_grad_enabled(False)
    rmse = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input_rgb, input, target, name, minmax = Variable(batch[0]),Variable(batch[1]), Variable(batch[2]), batch[3], batch[4]
            if cuda:
                input = input.cuda(gpus_list[opt.gpu])
                input_rgb = input_rgb.cuda(gpus_list[opt.gpu])
                gt = target.cuda(gpus_list[opt.gpu])

            "Output"
            t0 = time.time()
            prediction = model(input, input_rgb)[2]

            t1 = time.time()
            # print(result.shape, 'rrrrrrrrrrrrr')
            # print(prediction)
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            rmse += calc_rmse(gt[0,0].cpu().numpy(), prediction[0,0].cpu().numpy(), minmax)
    avg_rmse = rmse / len(testing_data_loader)
    return avg_rmse

def calc_rmse(a, b, minmax):
    maxx = minmax[1].numpy()
    minn = minmax[0].numpy()

    a = a[6:-6, 6:-6]
    b = b[6:-6, 6:-6]
    
    # a = a*(maxx-minn) + minn
    b = b*(maxx-minn) + minn
    a = a/10.0
    b = b/10.0

    return np.sqrt(np.mean(np.power(a-b,2)))

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def checkpoint(epoch):
    path_1 = opt.save_folder+opt.prefix+opt.train_dataset
    model_out_path = opt.save_folder+opt.prefix+opt.train_dataset+hostname+opt.model_type+opt.prefix+"_epoch_{}.pth".format(epoch)
    if not os.path.exists(path_1):
        os.makedirs(path_1)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


"Check cuda"
cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

"Choose Seed"
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)


"Loading Dataset"
print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.train_dataset, opt.patch_size, opt.upscale_factor, opt.data_augmentation)
test_set = get_test_set(opt.data_dir, opt.test_dir, opt.upscale_factor)

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

"Building Net"
print('===> Building model ', opt.model_type)
if opt.model_type == 'EDSRx2v4':
    model = EDSRx2v4(num_channels=1, base_filter=64, scale_factor=opt.upscale_factor)

"DataParallel"
# model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion = nn.L1Loss()
# bce_loss = nn.BCELoss(size_average=True)
criterion_ce = nn.CrossEntropyLoss()
print('---------- Networks architecture -------------')
print_network(model)
print('----------------------------------------------')


"Loading Pre-trained weights"
if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        pretrained_dict = torch.load(model_name, map_location=lambda storage, loc: storage)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        #model= torch.load(model_name, map_location=lambda storage, loc: storage)
        # model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print("************************************Pre-trained SR weights:{} is loaded.************************************".format(opt.pretrained_sr))


"To cuda"
if cuda:
    model = model.cuda(gpus_list[opt.gpu])
    criterion = criterion.cuda(gpus_list[opt.gpu])


"Optimizer"
optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)

###
# img=Image.open('./MSG-TrainData/Edge_grad100f/1.png')
# r,g,b=img.split()
# print(len(img.split()))

# for param_group in optimizer.param_groups:
#     param_group['lr'] /= 100.0
# print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))          

"Training"
loss_list = []
rmse_list = []
for epoch in range(1, opt.nEpochs + 1):
    # avg_rmse = validate(epoch)
    train(epoch)

    if (epoch+1) % 2 == 0:
        plt.figure(1)
        plt.plot(loss_list, linewidth=5)
        plt.title('Loss Table', fontsize=24)
        plt.xlabel('epoch', fontsize=14)
        plt.ylabel('loss', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        path_fig = opt.save_folder+opt.prefix+opt.train_dataset
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        plt.savefig('{}log_loss_lr1e-4_batch8.png'.format(path_fig))

    if (epoch+1) == 40:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) == 70:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) == 140:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 5.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if (epoch+1) % (opt.snapshots) == 0:
        checkpoint(epoch)
        avg_rmse = validate(epoch)

        with open(opt.out_rmse,"a+") as f:
            f.write("epoch: {}, rmse: {} \r\n".format(epoch, avg_rmse))
        rmse_list.append(avg_rmse)

        plt.figure(2)
        plt.plot(rmse_list, linewidth=5)
        plt.title('RMSE Table', fontsize=24)
        plt.xlabel('checkpoint', fontsize=14)
        plt.ylabel('rmse', fontsize=14)
        plt.tick_params(axis='both', labelsize=14)
        path_fig = opt.save_folder+opt.prefix+opt.train_dataset
        if not os.path.exists(path_fig):
            os.makedirs(path_fig)
        plt.savefig('{}log_rmse_lr1e-4_batch8.png'.format(path_fig))

        # print("开始验证")
        # for index, (input, GT) in enumerate(dataloader, 0):

