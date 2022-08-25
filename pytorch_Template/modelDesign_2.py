# -*- coding: utf-8 -*-
# torch==1.2.0
# 归一化
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.signal import butter, lfilter, freqz
# def butter_lowpass(cutoff, fs, order=5):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return b, a
# def butter_lowpass_filter(data, cutoff, fs, order=5, axis = -1):
#     b, a = butter_lowpass(cutoff, fs, order=order)
#     y = lfilter(b, a, data, axis = -1)
#     return y
from torchvision.models import resnet34
# class Model_2(nn.Module):
#     def __init__(self):
#         super(Model_2, self).__init__()
#         self.net = nn.Sequential(
            
#             nn.Conv2d(256, 256, kernel_size = 2, stride = 1, padding= 1),  
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
#             nn.Conv2d(256, 512, kernel_size = 2, stride = 1, padding= 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
#             nn.Conv2d(512, 768, kernel_size = 2, stride = 1, padding= 1),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
#             nn.Flatten(),
#             nn.Linear(768*9*5,2),
#             # nn.Dropout(0.2),
#             # nn.Linear(10,2)
#         )
#         # 256,73,3
#         # 256,36,3

#         # 512,37,4
#         # 512,18,4

#         # 728,19,5
#         # 728,9,5


#     def forward(self, x, data_format='channels_last'):
#         # x.shape ([bs, 256, 72, 2])
        
#         out = self.net(x)
#         out[:,0][out[:,0]>120]=120
#         out[:,0][out[:,0]<0]=0
#         out[:,1][out[:,1]>60]=60
#         out[:,1][out[:,1]<0]=0
#         return out
class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # self.net = nn.Sequential(
        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

        #     nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=1, padding=0),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=0),

        #     nn.Flatten(),
        #     nn.Linear(256*8,2),
            
            
        # )
        resnet = resnet34(pretrained=True,)
        # resnet.fc = nn.Sequential(nn.Dropout(p=0.05), torch.nn.Linear(512,2))
        self.net = nn.Sequential(
            resnet,
            nn.Dropout(p=0.3),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1000,2)
        )
        # self.eval()



    def forward(self, x, data_format='channels_last'):
        # x.shape ([bs, 256, 72, 2])
        # with torch.no_grad():
        x = x.norm(dim=-1)  # ([bs, 256, 72])
        x = x.unsqueeze(1)
        x = x.repeat(1,3,1,1)
        out = self.net(x)
        out[:,0][out[:,0]>120]=120
        out[:,0][out[:,0]<0]=0
        out[:,1][out[:,1]>60]=60
        out[:,1][out[:,1]<0]=0
        return out
