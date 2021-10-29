import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageFilter
import random
from random import randrange
import glob
import cv2
import imageio

"Return all filenames of images in folder"
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


"Define the image loading"
def load_img(filepath, type):
    # img = Image.open(filepath).convert('RGB')
    ##############
    # if type == "depth":
    #     img = Image.open(filepath, 0)
    # else:
    #     img = Image.open(filepath)

    if type == "depth":
        img = imageio.imread(filepath)
    else:
        img = cv2.imread(filepath)
    # y, _, _ = img.split()
    return img

"loading high-frequncy part of the image "
# def load_hf(filepath):
#     # img = Image.open(filepath).convert('RGB')
#     ##############
#     img = Image.open(filepath)
#     hf = img.filter(ImageFilter.FIND_EDGES)
#     # img = Image.open(filepath)
#     # y, _, _ = img.split()
#     return hf

def get_patch(img_in, img_rgb, img_tar, patch_size, ix=-1, iy=-1):
    # print(img_in.shape)
    # print(patch_size)
    (c, ih, iw) = img_in.shape
    # print('input:', ih, iw)
    # (th, tw) = (scale * ih, scale * iw)
    scale = 4
    ip = int(patch_size / scale)
    tp = patch_size

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
    # print(ix, "xxxxxxxxx")
    # print(iy, "yyyyyyyyyyyy")
    # print(ip,"pppppppppppp")
    (tx, ty) = (scale * ix, scale * iy)
    img_in = img_in[:, iy:iy + ip, ix:ix + ip]
    # print('get_patch', img_tar.size(), ty, ty+tp, tx, tx+tp)
    img_rgb = img_rgb[:, ty:ty + tp, tx:tx + tp]
    img_tar = img_tar[:, ty:ty + tp, tx:tx + tp]

    info_patch = {
        'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
    # print(info_patch)
    # print('after', img_in.shape)
    # print('after', img_rgb.shape)
    # print('after', img_tar.shape)

    return img_in, img_rgb, img_tar, info_patch


"Define the data augmentation"
def augment(img_in, img_rgb, img_tar, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    # print(img_edge.size(), 'eeeeeeeeeeeeeeeeeeee')
    if random.random() < 0.5 and flip_h:
        ####print('<-------------->', img_tar.size())
        img_in = torch.from_numpy(img_in[:, :, ::-1].copy())
        img_rgb = torch.from_numpy(img_rgb[:, :, ::-1].copy())
        img_tar = torch.from_numpy(img_tar[:, :, ::-1].copy())
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            img_in = torch.from_numpy(img_in[:, ::-1, :].copy())
            img_rgb = torch.from_numpy(img_rgb[:, ::-1, :].copy())
            img_tar = torch.from_numpy(img_tar[:, ::-1, :].copy())
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            img_in = torch.FloatTensor(np.transpose(img_in, (0, 2, 1)))
            img_rgb = torch.FloatTensor(np.transpose(img_rgb, (0, 2, 1)))
            img_tar = torch.FloatTensor(np.transpose(img_tar, (0, 2, 1)))
            info_aug['trans'] = True
    # print(img_edge.size(), 'ssssssssss')
    return img_in, img_rgb, img_tar, info_aug


"Read the data from folder in training"
class DatasetFromFolder(data.Dataset):
    def __init__(self, dataset, patch_size, data_augmentation,
                 input_transform=None, input_rgb_transform=None,target_transform=None):
        super(DatasetFromFolder, self).__init__()
        # self.image_filepaths = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.dataset = dataset
        self.patch_size = patch_size
        self.dataset = dataset
        self.all_dirs = sorted(glob.glob(os.path.join(self.dataset, '*')))

        self.input_transform = input_transform
        self.input_rgb_transform = input_rgb_transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation


    def __getitem__(self, index):
        i_dir = self.all_dirs[index]
        # print(i_dir)
        "Get the filename with postfix"
        # _, file = os.path.split(i_dir)
        ##### print('<==============>', self.dataset)

        "Load target data"
        # target = load_img(self.image_filepaths[index])
        # print(self.image_filenames[index], target.size)
        # print(self.edge_dir)
        # print(self.rgb_dir)

        "Load the data by determining the type of the dataset"
        # print(i_dir.split('/')[-1])
        if self.dataset.split('/')[-2] == 'train':
            # print(os.path.join(self.rgb_dir, os.path.splitext(file)[0] + '.png'))
            target = load_img(os.path.join(i_dir, i_dir.split('/')[-1] + '_HR_gt.png'), "depth")
            input_rgb = load_img(os.path.join(i_dir, i_dir.split('/')[-1] + '_RGB.jpg'), "rgb")
            lr = load_img(os.path.join(i_dir, i_dir.split('/')[-1] + '_LR_fill_depth.png'), "depth").astype('float32')
            # input = input.resize((512,384),Image.BICUBIC)
            # input = cv2.resize(input, (512,384))
            input = Image.fromarray(lr).resize((128,96),Image.BICUBIC)

        "normalization"
        input_rgb = np.transpose(input_rgb, (2, 0, 1)) / 255.0
        # print(input)

        input = np.array(input)
        maxx=np.max(input)
        minn=np.min(input)
        input_norm=(input-minn)/(maxx-minn)
        input = np.expand_dims(input_norm, 0)

        # print(input)
        target=target.astype('float32')
        maxx=np.max(target)
        minn=np.min(target)
        target_norm=(target-minn)/(maxx-minn)
        target = np.expand_dims(target_norm, 0)

        # print(target)

        # print(input, '2222222222222222')

        "get_patch"
        input, input_rgb, target, _ = get_patch(input, input_rgb, target, self.patch_size)

        # if self.input_rgb_transform:
        #     input_rgb = self.input_rgb_transform(input_rgb).float()
        #     # print('input_edge_tttttttttttttttttt:', input_edge.size())
        # if self.input_transform:
        #     input = self.input_transform(input).float()
        # if self.target_transform:
        #     target = self.target_transform(target).float()
        # print('target:', target.size())
        # print('target:', target.size())

        "Augment the image data"
        # if self.data_augmentation:
        #     input, input_rgb, target, _ = augment(input, input_rgb, target)

        "Transform the image data"
        input = torch.from_numpy(input).float()
        input_rgb = torch.from_numpy(input_rgb).float()
        target = torch.from_numpy(target).float()

        # print('input_rgb:', input_rgb.size())
        # print('input:', input.size())
        # print('target:', target.size())

        return input_rgb, input, target

    def __len__(self):
        return len(self.all_dirs)


"Read the data from folder in evaluation"
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, dataset,scale,
                 input_transform=None, input_rgb_transform=None):
        super(DatasetFromFolderEval, self).__init__()
        # self.image_filepaths = [join(hr_dir, x) for x in listdir(hr_dir) if is_image_file(x)]
        self.dataset = dataset
        self.all_dirs = sorted(glob.glob(os.path.join(self.dataset, '*')))

        self.input_transform = input_transform
        self.input_rgb_transform = input_rgb_transform

        self.LR = 'LR_FILL_20/'
        self.RGB = 'RGB_20/'
        self.all_files = sorted(glob.glob(os.path.join(self.dataset, self.LR + '*')))

        #print(len(self.all_files), 'aaaaaaaaaaaa')

    def __getitem__(self, index):
        
        # print(i_dir)
        "Load the data by determining the type of the dataset"
        # print(i_dir.split('/')[-1])
        if self.dataset.split('/')[-2] == 'validate20':
            i_dir = self.all_dirs[index]
            # print(os.path.join(self.rgb_dir, os.path.splitext(file)[0] + '.png'))
            input_rgb = load_img(os.path.join(i_dir, i_dir.split('/')[-1] + '_RGB.jpg'), "rgb")
            lr = load_img(os.path.join(i_dir, i_dir.split('/')[-1] + '_LR_fill_depth.png'), "depth").astype('float32')
            input = Image.fromarray(lr).resize((256,192),Image.BICUBIC)
        elif self.dataset.split('/')[-2] == 'Stage1_new':
            i_file = self.all_files[index]
            # print(self.all_files)
            # print(i_file)
            input_rgb = load_img(os.path.join(self.dataset + self.RGB, i_file.split('/')[-1].split('_')[0] + '_RGB.jpg'), "rgb")
            lr = load_img(os.path.join(self.dataset + self.LR, i_file.split('/')[-1].split('_')[0] + '_LR_fill_depth.png'), "depth").astype('float32')
            # input = input.resize((512,384),Image.BICUBIC)
            # input = cv2.resize(input, (512,384))
            input = Image.fromarray(lr).resize((256,192),Image.BICUBIC)
            i_dir = i_file.split('/')[-1].split('_')[0]

        "normalization"
        input_rgb = np.transpose(input_rgb, (2, 0, 1)) / 255.0

        input = np.array(input)
        maxx=np.max(input)
        minn=np.min(input)
        minmax = (minn, maxx)
        # print(minmax)

        input_norm=(input-minn)/(maxx-minn)
        input = np.expand_dims(input_norm, 0)


        "Transform the image data"
        input = torch.from_numpy(input).float()
        input_rgb = torch.from_numpy(input_rgb).float()

        # print('input_rgb:', input_rgb.size())
        # print('input:', input.size())
        # print('target:', target.size())
        return input_rgb, input, i_dir+'.png', minmax

    def __len__(self):
        return len(self.all_files)
