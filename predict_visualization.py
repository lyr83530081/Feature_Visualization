# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 22:54:55 2021

@author: User
"""
import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import torch.nn as nn
import skimage.data
import skimage.io
import math
import skimage.transform
import matplotlib.pyplot as plt
from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

def get_picture(picture_dir, transform):
    '''
    该算法实现了读取图片，并将其类型转化为Tensor
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    img256 = np.asarray(img256)
    img256 = img256.astype(np.float32)

    return transform(img256)

def get_picture_rgb(picture_dir):
    '''
    该函数实现了显示图片的RGB三通道颜色
    '''
    img = skimage.io.imread(picture_dir)
    img256 = skimage.transform.resize(img, (256, 256))
    skimage.io.imsave('new4.jpg', img256)

    # 取单一通道值显示
    # for i in range(3):
    #     img = img256[:,:,i]
    #     ax = plt.subplot(1, 3, i + 1)
    #     ax.set_title('Feature {}'.format(i))
    #     ax.axis('off')
    #     plt.imshow(img)

    # r = img256.copy()
    # r[:,:,0:2]=0
    # ax = plt.subplot(1, 4, 1)
    # ax.set_title('B Channel')
    # # ax.axis('off')
    # plt.imshow(r)

    # g = img256.copy()
    # g[:,:,0]=0
    # g[:,:,2]=0
    # ax = plt.subplot(1, 4, 2)
    # ax.set_title('G Channel')
    # # ax.axis('off')
    # plt.imshow(g)

    # b = img256.copy()
    # b[:,:,1:3]=0
    # ax = plt.subplot(1, 4, 3)
    # ax.set_title('R Channel')
    # # ax.axis('off')
    # plt.imshow(b)

    # img = img256.copy()
    # ax = plt.subplot(1, 4, 4)
    # ax.set_title('image')
    # # ax.axis('off')
    # plt.imshow(img)

    img = img256.copy()
    ax = plt.subplot()
    ax.set_title('new-image')
    # ax.axis('off')
    plt.imshow(img)

    plt.show()

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        print('---------',self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            if "fc" in name:
                x = x.view(x.size(0), -1)
            print(module)
            x = module(x)
            print('name', name)
            if name in self.extracted_layers:
                outputs.append(x)
        return outputs





if __name__ == '__main__':
    weights_file = 'BLAH_BLAH/epoch_378.pth'
    image_file = 'data/butterfly_GT.bmp'
    scale = 2
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)
    
    exact_list = ['conv2']
    
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
            
    model.eval()
    myexactor = FeatureExtractor(model, exact_list) 
    image = pil_image.open(image_file).convert('RGB')

    image_width = (image.width // scale) * scale
    image_height = (image.height // scale) * scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // scale, image.height // scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.BICUBIC)
    #image.save(image_file.replace('.', '_bicubic_x{}.'.format(scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)
    
    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        maps = myexactor(y)#.clamp(0.0, 1.0)
        print(maps[0].shape[1])
    
    col_num = math.ceil(math.sqrt(maps[0].shape[1]))
    for i in range(maps[0].shape[1]):  # 可视化了32通道
        ax = plt.subplot(col_num, col_num, i + 1)
        #ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        #ax.set_title('new—conv1-image')

        plt.imshow(maps[0].data.cpu().numpy()[0,i,:,:],cmap='gray')

    plt.show() 


