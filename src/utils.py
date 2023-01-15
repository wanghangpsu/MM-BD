
import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def pattern_craft(im_size):

    pattern = torch.zeros(im_size)
    cx = torch.randint(low=3, high=im_size[-1] - 6, size=(1,))
    cy = torch.randint(low=3, high=im_size[-1] - 6, size=(1,))
    value = torch.randint(low=0, high=255, size=(9, 3)).float() / 255
    for i in range(3):
        for j in range(3):
            pattern[:, cx[0]+i, cy[0]+j] = value[i*3+j, :]

    return pattern


def mask_craft(pattern):

    mask = (pattern > 0.0).float()

    return mask


def embed_backdoor(image, pattern, mask):
    image = image * (1-mask) + pattern * mask
    image.clamp(0, 1)
    
    return image


def im_process(images, process_type, process_parameter):
    NoI = images.size(0)
    im_size = images.size()[1:]

    if process_type == 'noise':
        images += torch.randint(-int(process_parameter), int(process_parameter+1), images.size()).to(device)/255.0
        images.clamp(0, 1)
    if process_type == 'avg_filter':
        if ~int(process_parameter)%2:
            temp = torch.zeros((images.size(0), images.size(1), images.size(2)+1, images.size(3)+1)).to(device)
            temp[:, :, :images.size(2), :images.size(3)] = images
            images = temp
        padding_size = int((process_parameter-1)/2)
        weight = torch.zeros(images.size(1), images.size(1), int(process_parameter), int(process_parameter)).to(device)
        for c in range(images.size(1)):
            weight[c, c, :, :] = torch.ones((int(process_parameter), int(process_parameter)))/(process_parameter**2)
        images = F.conv2d(images, weight, padding=padding_size)
    if process_type == 'med_filter':
        if ~int(process_parameter)%2:
            temp = torch.zeros((images.size(0), images.size(1), images.size(2)+1, images.size(3)+1)).to(device)
            temp[:, :, :images.size(2), :images.size(3)] = images
            images = temp
        padding_size = int((process_parameter-1)/2)
        temp = torch.zeros((images.size(0), images.size(1), images.size(2)+2*padding_size, images.size(3)+2*padding_size)).to(device)
        temp[:, :, padding_size:images.size(2)+padding_size, padding_size:images.size(3)+padding_size] = images
        images = temp
        images_unfolded = images.unfold(2, int(process_parameter), 1).unfold(3, int(process_parameter), 1)
        images_unfolded = torch.reshape(images_unfolded, (images_unfolded.size(0), images_unfolded.size(1), images_unfolded.size(2), images_unfolded.size(3), -1))
        images = torch.median(images_unfolded, -1)[0]
    if process_type == 'quant':
        l = 2**process_parameter
        images = (torch.round(images*l+0.5)-0.5)/l
    
    return images
