# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 07:04:29 2021

@author: user
"""

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

def show_kernal(model):
    # 可视化卷积核
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            in_channels = param.size()[1]
            out_channels = param.size()[0]  # 输出通道，表示卷积核的个数
 
            k_w, k_h = param.size()[3], param.size()[2]  # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
    return kernel_all


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
            
    
    kernel_all = show_kernal(model)
    col_num = math.ceil(math.sqrt(kernel_all.shape[0]))
    for i in range(kernel_all.shape[0]):  # 可视化了32通道
        ax = plt.subplot(col_num, col_num, i + 1)
        #ax.set_title('Feature {}'.format(i))
        ax.axis('off')
        #ax.set_title('new—conv1-image')

        plt.imshow(kernel_all.data.cpu().numpy()[i,0,:,:],cmap='gray')

    plt.show() 
    
    