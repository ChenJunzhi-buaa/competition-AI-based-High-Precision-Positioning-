#--coding: utf-8--
"""
model1 的模型集成，
集成模型的模型文件包含各个子模型，
可以按照字典的key将子模型的模型参数赋值给集成模型,
给--float16 的parser指令, 则将模型的参数从32位精度降低到16精度
"""
from modelDesign_1 import Model_1
import torch
import os
from shutil import copyfile
import numpy as np
from score import test
ese = Model_1(method_id=0)

state_dict = dict()
ckpts = [1] * 200

ckpts[1] = torch.load('/data/cjz/location/submit/61-5-2-2-2-2-7796epochs/submit_pt/modelSubmit_1.pth')
ckpts[2]= torch.load('/data/cjz/location/submit/61-3-4/submit_pt/modelSubmit_1.pth')
ckpts[3] = torch.load('/data/cjz/location/submit/61-6/submit_pt/modelSubmit_1.pth')
ckpts[4] = torch.load('/data/cjz/location/submit/61-5-5-5-5-5-5-5/modelSubmit_2_2000epochs.pth')
ckpts[5] = torch.load('/data/cjz/location/submit/61-5-2-2-2-2961epochs/modelSubmit_2_min_testloss.pth')
ckpts[6] = torch.load('/data/cjz/location/submit/61-5-3423epoch/submit_pt/modelSubmit_1.pth')
ckpts[7] = torch.load('/data/cjz/location/submit/111/modelSubmit_1.pth')
# ckpt7 = torch.load('/data/cjz/location/submit/43/submit_pt/modelSubmit_1.pth')
# ckpts = [ckpt1, ckpt2, ckpt3, ckpt4, ckpt5, ckpt6]
for i in range(1000):
    if hasattr(ese, f'net{i}'):
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
model_save = '/data/cjz/location/submit/_ese'
if not os.path.exists(os.path.join(model_save, str(i))):
    os.mkdir(os.path.join(model_save, str(i)))



if args.float16 == True:
    ese.half()
torch.save(ese.state_dict(), os.path.join(model_save, str(i),  f'modelSubmit_1.pth'), )
copyfile('/data/cjz/location/pytorch_Template/modelDesign_1.py', os.path.join(model_save, str(i), 'modelDesign_1.py'))
copyfile(__file__, os.path.join(model_save, str(i), 'model1_esemble.py'))


file_name1 = 'data/Case_1_2_Training.npy'
CIR = np.load(file_name1)
trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
file_name2 = 'data/Case_1_2_Training_Label.npy'
POS = np.load(file_name2)
trainY = POS.transpose((1,0)) #[none, 2]


testX = torch.tensor(trainX, dtype=torch.float32)
testY = torch.tensor(trainY, dtype=torch.float32)
score_max = 0
thres_best = 0
thres = 99.7
# for thres in np.arange(99.5,99.9,0.01):
print(f'################## thres:{thres} ########################')
model1 = Model_1(method_id=0, thres = thres)
model1.load_state_dict(torch.load(f'/data/cjz/location/submit/_ese/{args.submit_id}/modelSubmit_1.pth'))
score = test(testX, testY, model1)
if score > score_max:
    score_max = score
    thres_best = thres
    print(f'score_max:{score_max}, 此时的thres{thres_best}')
print(f'score_max:{score_max}, 此时的thres{thres_best}')

# torch.save(thres_best,  os.path.join(model_save, str(i), 'thres_best.pth') )
# torch.save(score_max,  os.path.join(model_save, str(i), 'score_max.pth') )