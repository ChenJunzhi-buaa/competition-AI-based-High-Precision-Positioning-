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
from torchvision.models import resnet18,resnet34
class Model_1(nn.Module):
    def __init__(self, no_grad=True, infer_batchsize=256):
        super(Model_1, self).__init__()
        self.no_grad = no_grad
        self.infer_batchsize = infer_batchsize
        if self.no_grad == True:
            resnet = resnet34(pretrained=False,)
        else:
            resnet = resnet34(pretrained=True,)
        # resnet.fc = nn.Sequential(nn.Dropout(p=0.05), torch.nn.Linear(512,2))
        resnet.fc = nn.Linear(512,2)
        self.net = nn.Sequential(
            # nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            resnet
            )
        # self.eval()


    def _forward(self, x, data_format='channels_last'):
        """方式1：设为0"""
        # x[:,:,4:20,:]=0
        # x[:,:,24:48,:]=0
        # x[:,:,52:68,:]=0
        # x.shape ([bs, 256, 72, 2])
        """方式2：去掉不用通道"""
        # x = torch.concat((x[:,:,0:4,:], x[:,:,20:24,:], x[:,:,48:52,:], x[:,:,68:,:]), dim=2)
        x = torch.cat((x[:,:,0:4,:], x[:,:,20:24,:], x[:,:,48:52,:], x[:,:,68:,:]), dim=2)

        x = x.norm(dim=-1)
        x = x.unsqueeze(1)
        x = x.repeat(1,3,1,1)
        

        return self.net(x)

    def forward(self, x, data_format='channels_last'):
        if self.no_grad == True:
            self.eval()
           
            with torch.no_grad():
                _out = []
                for i in range(0,x.shape[0],self.infer_batchsize):
                    if i+self.infer_batchsize <= x.shape[0]:
                        batch_out = x[i:i+self.infer_batchsize]
                    else:
                        batch_out = x[i:]
                    _out.append(batch_out)
                out = torch.cat(_out, axis=0)
        else:
            out = self._forward(x)
        
        return out

"""transformer"""
# import h5py
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision.models import resnet18

# class Model_1(nn.Module):
#     def __init__(self):
#         super(Model_1, self).__init__()
  
#         encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2,)
#         transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6,)
#         self.net = nn.Sequential(
#             transformer_encoder,
#             nn.Flatten(),
#             nn.Linear(256*16,2)
#         )
#         # self.eval()



#     def forward(self, x, data_format='channels_last'):
#         # with torch.no_grad():
#         """方式1：设为0"""
#         # x[:,:,4:20,:]=0
#         # x[:,:,24:48,:]=0
#         # x[:,:,52:68,:]=0
#         # x.shape ([bs, 256, 72, 2])
#         """方式2：去掉不用通道"""
#         # x = torch.concat((x[:,:,0:4,:], x[:,:,20:24,:], x[:,:,48:52,:], x[:,:,68:,:]), dim=2)
#         x = torch.cat((x[:,:,0:4,:], x[:,:,20:24,:], x[:,:,48:52,:], x[:,:,68:,:]), dim=2)

#         x = x.norm(dim=-1)
        

#         out = self.net(x)
#         return out