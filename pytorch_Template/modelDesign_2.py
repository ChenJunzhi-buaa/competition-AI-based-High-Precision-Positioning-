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
from torchvision.models import resnet18,resnet34,resnet50,resnet101,resnext50_32x4d,resnext101_32x8d
import random,os
import copy
class Model_2(nn.Module):
    def __init__(self, no_grad=True, infer_batchsize=256):
        super(Model_2, self).__init__()
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
            resnet,
            )
        # self.eval()


    def _forward(self, x, data_format='channels_last'):
        
        """方式1：幅值复制三份"""
        # x = x.norm(dim=-1)
        # x = x.unsqueeze(1)
        # x = x.repeat(1,3,1,1)
        """方式2：实部虚部幅值"""
        x_norm = x.norm(dim=-1)
        x_norm = x_norm.unsqueeze(3)
        x = torch.cat((x,x_norm),dim=3)
        x = x.permute(0,3,1,2)
        

        return self.net(x)
    def _tta_forward(self, x, num=5):
        # def seed_everything(seed_value):
        #     random.seed(seed_value)
        #     np.random.seed(seed_value)
        #     torch.manual_seed(seed_value)
        #     os.environ['PYTHONHASHSEED'] = str(seed_value)
            
        #     if torch.cuda.is_available(): 
        #         torch.cuda.manual_seed(seed_value)
        #         torch.cuda.manual_seed_all(seed_value)
        #         torch.backends.cudnn.deterministic = True
        #         torch.backends.cudnn.benchmark = False
        #         torch.backends.cudnn.enabled = False
        # seed_everything(seed_value=42)
        # out = self._forward(x)
        # for i in range(num-1):
        #     delete_num = np.random.choice(range(1,15),1)
        #     mask = np.random.choice(18,delete_num,replace=False)
        #     mask = np.concatenate((mask*4, mask*4+1, mask*4+2, mask*4+3))
        #     x_copy = copy.deepcopy(x)
        #     x_copy[:, :, mask, : ] = 0
        #     out = out + self._forward(x_copy)
        # out = out/num
        out = self._forward(x)
        aug_times = 10
        x_aug = self.data_aug(x, aug_times=aug_times)
        for i in range(1, aug_times):
            out = out + self._forward(x_aug[i*(x.shape[0]):(i+1)*(x.shape[0])])
        out = out / aug_times
        return out

    def forward(self, x, data_format='channels_last'):
        if self.no_grad == True:
            self.eval()
           
            with torch.no_grad():
                _out = []
                for i in range(0,x.shape[0],self.infer_batchsize):
                    if i+self.infer_batchsize <= x.shape[0]:
                        batch_out = self._forward(x[i:i+self.infer_batchsize])
                    else:
                        batch_out = self._forward(x[i:])
                    _out.append(batch_out)
                out = torch.cat(_out, axis=0)
        else:
            out = self._forward(x)
        
        return out
    def data_aug(self, x, aug_times=10, y=None):
        # TODO 1、mask掉时间维度，2、mask数量随机
        """固定mask掉一半的基站"""
        # x.shape = bs,256,72,2
        x_aug = copy.deepcopy(x)
        if y is not None:
            y_aug = copy.deepcopy(y)
        for j in range(aug_times - 1):
            x_copy = copy.deepcopy(x)
            for i in range(x.shape[0]):
                delete_num  = int( x.shape[2] / 4 / 2 )
                base_mask = np.random.choice(18,delete_num,replace=False)
                mask = np.concatenate((base_mask*4, base_mask*4+1, base_mask*4+2, base_mask*4+3))
                x_copy[i,:,mask,:] = 0
            x_aug = torch.cat((x_aug, x_copy), axis = 0)
            if y is not None:
                y_aug = torch.cat((y_aug, y), axis = 0)
        if y is not None:
            return x_aug, y_aug 
        else:
            return x_aug

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