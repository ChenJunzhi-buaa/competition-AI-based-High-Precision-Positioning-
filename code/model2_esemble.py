#--coding: utf-8--
"""
model2的模型集成
"""
from modelDesign_2 import Model_2
import torch
import os
from shutil import copyfile
import numpy as np
from score import test
ese = Model_2(method_id=0)

state_dict = dict()

ckpts = [1]*100
ckpts[4] = torch.load('/data/cjz/location/submit/59-4/submit_pt/modelSubmit_2.pth')
ckpts[6] = torch.load('/data/cjz/location/submit/59-6/submit_pt/modelSubmit_2.pth')
ckpts[7] = torch.load('/data/cjz/location/submit/59-7/submit_pt/modelSubmit_2.pth')
ckpts[25] = torch.load('/data/cjz/location/submit/59-25-531epochs/submit_pt/modelSubmit_2.pth')
ckpts[35] = torch.load('/data/cjz/location/submit/59-25-25-25-25-25-25_/modelSubmit_2_1600epochs.pth')
ckpts[36] = torch.load('/data/cjz/location/submit/65-6/modelSubmit_2_2000epochs.pth')

ckpts[41] = torch.load('/data/cjz/location/submit/58-4/submit_pt/modelSubmit_2.pth')
ckpts[42] = torch.load('/data/cjz/location/submit/58-6/submit_pt/modelSubmit_2.pth')
ckpts[43] = torch.load('/data/cjz/location/submit/58-12/submit_pt/modelSubmit_2.pth')
ckpts[44] = torch.load('/data/cjz/location/submit/62-4/submit_pt/modelSubmit_2.pth')
ckpts[45] = torch.load('/data/cjz/location/submit/62-5/submit_pt/modelSubmit_2.pth')
ckpts[46] = torch.load('/data/cjz/location/submit/62-25/submit_pt/modelSubmit_2.pth')

ckpts[51] = torch.load('/data/cjz/location/submit/65-4/modelSubmit_2_2600epochs.pth')
ckpts[52] = torch.load('/data/cjz/location/submit/65-6/modelSubmit_2_2800epochs.pth')
ckpts[53] = torch.load('/data/cjz/location/submit/65-7/modelSubmit_2_2000epochs.pth')
ckpts[54] = torch.load('/data/cjz/location/submit/65-9/modelSubmit_2_3600epochs.pth')
ckpts[55] = torch.load('/data/cjz/location/submit/65-12/modelSubmit_2_1800epochs.pth')
ckpts[56] = torch.load('/data/cjz/location/submit/65-25/modelSubmit_2_6400epochs.pth')

for i in range(1000):
    if hasattr(ese, f'net{i}'):# if ckpts[i] is not 1:
        print(f'已用子网络net{i}')
        ckpt = ckpts[i]
        for key in list(ckpt.keys()):
            new_key = f'net{i}.' + key 
            state_dict[new_key] = ckpt[key]


ese.load_state_dict(state_dict)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--submit_id', type=int, required=True)
parser.add_argument('--float16', default=False, action = 'store_true' )

args = parser.parse_args()
i = args.submit_id
model_save = '/data/cjz/location/submit/_ese2'
if not os.path.exists(os.path.join(model_save, str(i))):
    os.mkdir(os.path.join(model_save, str(i)))



if args.float16 == True:
    ese.half()
torch.save(ese.state_dict(), os.path.join(model_save, str(i),  f'modelSubmit_2.pth'))
copyfile('/data/cjz/location/pytorch_Template/modelDesign_2.py', os.path.join(model_save, str(i), 'modelDesign_2.py'))
copyfile(__file__, os.path.join(model_save, str(i), 'model2_esemble.py'))



# 集成模型##
model2 = Model_2(method_id=0)
model2.load_state_dict(torch.load(f'/data/cjz/location/submit/_ese2/{args.submit_id}/modelSubmit_2.pth'))
file_name1 = 'data/Case_1_2_Training.npy'
# logging.info('The current dataset is : %s'%(file_name1))
CIR = np.load(file_name1)
trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
file_name2 = 'data/Case_1_2_Training_Label.npy'
# logging.info('The current dataset is : %s'%(file_name2))
POS = np.load(file_name2)
trainY = POS.transpose((1,0)) #[none, 2]


testX = torch.tensor(trainX, dtype=torch.float32)
testY = torch.tensor(trainY, dtype=torch.float32)
test(testX, testY, model2)

# test(testX, testY, *model1s)
print(1)