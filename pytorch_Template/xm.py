# from netrc import netrc
# from typing_extensions import Self
import numpy as np
import h5py
import torch
import torch.nn as nn

from modelDesign_1 import Model_1
from modelDesign_2 import Model_2, Helpnet

DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:1")

def test(*models):
        def score(y_test):
                Diff   = np.sqrt(np.sum((np.square(y_test - trainY)),1)) # Euclidean distance
                Order = np.sort(Diff)
                Score_Case3 =  Order[int(Num*0.9)]
                print('The score of Case 3 is '  + np.str(100 - Score_Case3))
        y_test_avg = 0
        for model in models:
                y_test = model(trainX).cpu().data.numpy()
                y_test_avg = y_test_avg + y_test
                score(y_test)
        y_test_avg = y_test_avg/len(models)
        score(y_test_avg)




##########################
# Case 3
file_name1 = 'data/case3/Case_3_Training_test.npy'
trainX_labeled = np.load(file_name1)
Num = trainX_labeled.shape[0]  #Number of samples


file_name2 = 'data/case3/Case_3_Training_test_label.npy'
trainY = (np.load(file_name2))[:,:2]
trainX = torch.from_numpy(trainX_labeled.astype(np.float32)).to(DEVICE)


##########################
# Model loading
# model_address = './submit/53-1/submit_pt/modelSubmit_2.pth'
# model = Model_2()
# model.load_state_dict(torch.load(model_address, map_location=DEVICE))
# model_1 =model.to(DEVICE)

# model_address = './submit/53-2/submit_pt/modelSubmit_2.pth'
# model = Model_2()
# model.load_state_dict(torch.load(model_address, map_location=DEVICE))
# model_2 =model.to(DEVICE)

# model_address = './submit/51/submit_pt/modelSubmit_2.pth'
# model = Helpnet()
# model.load_state_dict(torch.load(model_address, map_location=DEVICE))
# model_3 =model.to(DEVICE)

# model_address = './submit/54_5/submit_pt/modelSubmit_2.pth'
# model = Model_2()
# model.load_state_dict(torch.load(model_address, map_location=DEVICE))
# model_4 =model.to(DEVICE)

# for i in range(1,13):
        # model_address = './submit/53-1/submit_pt/modelSubmit_2.pth'
        # model = Model_2()
        # model.load_state_dict(torch.load(model_address, map_location=DEVICE))
        # model_1 =model.to(DEVICE)


# test(model_4)


# class EsembleNet():
#         self.net1 = Model_2()
#         self.net2 = Helpnet()


# ese = EsembleNet()

# state_dict = dict()

# ckpt1 = torch.load()
# ckpt2 = torch.load()

# for key in list(ckpt1.keys()):
#         new_key = 'net1.' + key 
#         state_dict[new_key] = ckpt1[key]


# ckpt2 ...

# ese.load_state_dict(state_dict)

"""model1"""