# -*- coding: utf-8 -*-
# torch==1.2.0
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        
        self.relu   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        self.conv1 = nn.Conv2d(256, 256, kernel_size = 2, stride = 1, padding= 1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size = 2, stride = 1, padding= 1)
        self.conv3 = nn.Conv2d(512, 768, kernel_size = 2, stride = 1, padding= 1)
        
        self.pool  = nn.MaxPool2d(kernel_size = (2, 1), stride = (2, 1), padding = 0)
        self.Flatten = nn.Flatten()
        
        self.fc_1  = nn.Linear(768*9*5,2)
        
    def forward(self, x, data_format='channels_last'):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.Flatten(x)
        
        out = self.fc_1(x)

        return out