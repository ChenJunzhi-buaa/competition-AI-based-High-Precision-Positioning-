from modelDesign_1 import Model_1
import torch
import os
from shutil import copyfile
ese = Model_1(method_id=0)

state_dict = dict()

ckpt1 = torch.load('/data/cjz/location/submit/61-5-2-2-2-2-7796epochs/submit_pt/modelSubmit_1.pth')
ckpt2 = torch.load('/data/cjz/location/submit/61-3-4/submit_pt/modelSubmit_1.pth')
ckpt3 = torch.load('/data/cjz/location/submit/61-6/submit_pt/modelSubmit_1.pth')
ckpts = [ckpt1, ckpt2, ckpt3]
for i in [1,2,3]:
    ckpt = ckpts[i-1]
    for key in list(ckpt.keys()):
        new_key = f'net{i}.' + key 
        state_dict[new_key] = ckpt[key]

# for key in list(ckpt2.keys()):
#     new_key = 'net2.' + key 
#     state_dict[new_key] = ckpt2[key]
# for key in list(ckpt3.keys()):
#     new_key = 'net3.' + key 
#     state_dict[new_key] = ckpt3[key]
# ckpt2 ...
ese.load_state_dict(state_dict)

i = 1
model_save = '/data/cjz/location/submit/_ese'
if not os.path.exists(os.path.join(model_save, str(i))):
    os.mkdir(os.path.join(model_save, str(i)))
torch.save(ese.state_dict(), os.path.join(model_save, str(i),  f'modelSubmit_1.pth'))
copyfile('/data/cjz/location/pytorch_Template/modelDesign_1.py', os.path.join(model_save, str(i), 'modelDesign_1.py'))
copyfile(__file__, os.path.join(model_save, str(i), 'model1_esemble.py'))
print(1)