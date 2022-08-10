# -*- coding: utf-8 -*-
# torch==1.2.0
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        self.net = nn.Sequential(
            
            nn.Conv2d(256, 256, kernel_size = 2, stride = 1, padding= 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
            nn.Conv2d(256, 512, kernel_size = 2, stride = 1, padding= 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
            nn.Conv2d(512, 768, kernel_size = 2, stride = 1, padding= 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
            nn.Flatten(),
            nn.Linear(768*9*5,2),
            # nn.Dropout(0.2),
            # nn.Linear(10,2)
        )


    def forward(self, x, data_format='channels_last'):
        # x.shape ([bs, 256, 72, 2])
        
        out = self.net(x)
        out[:,0][out[:,0]>120]=120
        out[:,0][out[:,0]<0]=0
        out[:,1][out[:,1]>60]=60
        out[:,1][out[:,1]<0]=0
        return out