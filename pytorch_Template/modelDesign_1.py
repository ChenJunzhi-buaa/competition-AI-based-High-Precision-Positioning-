# -*- coding: utf-8 -*-
# torch==1.2.0
"""官方"""
# import h5py
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision.models import resnet34, vgg11_bn

# class Model_1(nn.Module):
#     def __init__(self):
#         super(Model_1, self).__init__()
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
#             nn.Linear(768*9*4,1000),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Linear(1000, 2)
#             # nn.Dropout(0.2),
#             # nn.Linear(10,2)
#         )


#         # self.net = nn.Sequential(
#         #     nn.Conv1d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=0,),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
#         #     nn.Conv1d(in_channels=512, out_channels=768, kernel_size=4, stride=1, padding=0),
#         #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         #     nn.MaxPool1d(kernel_size=3, stride=3, padding=0),
#         #     nn.Flatten(),
#         #     nn.Linear(768*6,2),
#         # )
        
#         # vgg = vgg11_bn(pretrained=True,)
#         # vgg.fc = torch.nn.Linear(512,2)
#         # self.net = vgg
#         # self.eval()



#     def forward(self, x, data_format='channels_last'):
#         x[:,:,4:20,:]=0
#         x[:,:,24:48,:]=0
#         x[:,:,52:68,:]=0
#         # x.shape ([bs, 256, 72, 2])
#         # with torch.no_grad():
#         x = x.norm(dim=-1)
#         x = x.unsqueeze(3)
#         # x = x.repeat(1,3,1,1)

#         out = self.net(x)
#         # out[:,0][out[:,0]>120]=120
#         # out[:,0][out[:,0]<0]=0
#         # out[:,1][out[:,1]>60]=60
#         # out[:,1][out[:,1]<0]=0
#         return out


"""resnet"""
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
    #     self.net = nn.Sequential(
            
    #         nn.Conv2d(256, 256, kernel_size = 2, stride = 1, padding= 1),
    #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #         nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
    #         nn.Conv2d(256, 512, kernel_size = 2, stride = 1, padding= 1),
    #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #         nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
    #         nn.Conv2d(512, 768, kernel_size = 2, stride = 1, padding= 1),
    #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
    #         nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0),
    #         nn.Flatten(),
    #         nn.Linear(768*9*5,2),
    #         # nn.Dropout(0.2),
    #         # nn.Linear(10,2)
    #     )


    # def forward(self, x, data_format='channels_last'):
    #     x[:,:,4:20,:]=0
    #     x[:,:,24:48,:]=0
    #     x[:,:,52:68,:]=0
    #     # x.shape ([bs, 256, 72, 2])
        
    #     out = self.net(x)
    #     out[:,0][out[:,0]>120]=120
    #     out[:,0][out[:,0]<0]=0
    #     out[:,1][out[:,1]>60]=60
    #     out[:,1][out[:,1]<0]=0
    #     return out
        resnet = resnet18(pretrained=True,)
        # resnet.fc = nn.Sequential(nn.Dropout(p=0.05), torch.nn.Linear(512,2))
        resnet.fc = nn.Linear(512,2)
        self.net = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            resnet
            )
        # self.eval()



    def forward(self, x, data_format='channels_last'):
        # with torch.no_grad():
        """方式1：设为0"""
        # x[:,:,4:20,:]=0
        # x[:,:,24:48,:]=0
        # x[:,:,52:68,:]=0
        # x.shape ([bs, 256, 72, 2])
        """方式2：去掉不用通道"""
        x = torch.concat((x[:,:,0:4,:], x[:,:,20:24,:], x[:,:,48:52,:], x[:,:,68:,:]), dim=2)

        x = x.norm(dim=-1)
        x = x.unsqueeze(1)
        x = x.repeat(1,3,1,1)
        

        out = self.net(x)
        return out