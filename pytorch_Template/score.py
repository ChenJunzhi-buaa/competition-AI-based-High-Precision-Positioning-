from netrc import netrc
from typing_extensions import Self
import numpy as np
import h5py
import torch
import torch.nn as nn


from utils import get_900
DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:2")

def test(testX, testY, *models, weight_thres=98.05):
    testY = np.array(testY)
    Num = len(testY)
    testX = testX.to(DEVICE)
    weights = []
    def score(y_test):
        Diff   = np.sqrt(np.sum((np.square(y_test - testY)),1)) # Euclidean distance
        Order = np.sort(Diff)
        Score_Case3 =  Order[int(Num*0.9)]
        print('The score of Case 3 is '  + np.str(100 - Score_Case3))
        return 100 - Score_Case3
    y_test_avg = 0
    for model in models:
        with torch.no_grad():
            model.eval()
            model = model.to(DEVICE)
            y_test = model(testX).cpu().data.numpy()
            score_ = score(y_test)
            weight = score_ - weight_thres if score_>weight_thres else 0
            weights.append(weight)
            y_test_avg = y_test_avg + y_test * weight
    y_test_avg = y_test_avg/sum(weights)
    score(y_test_avg)

if __name__ == '__main__':
    from modelDesign_1 import Model_1
    from modelDesign_2 import Model_2, Helpnet
    """model2"""
    # ##########################
    # # Case 3
    # # file_name1 = 'data/case3/Case_3_Training_test.npy'
    # # trainX_labeled = np.load(file_name1)
    # # Num = trainX_labeled.shape[0]  #Number of samples


    # # file_name2 = 'data/case3/Case_3_Training_test_label.npy'
    # # trainY = (np.load(file_name2))[:,:2]
    # # trainX = torch.from_numpy(trainX_labeled.astype(np.float32)).to(DEVICE)


    # ##########################
    # # Model loading
    # # model_address = './submit/53-1/submit_pt/modelSubmit_2.pth'
    # # model = Model_2()
    # # model.load_state_dict(torch.load(model_address, map_location=DEVICE))
    # # model_1 =model.to(DEVICE)

    # # model_address = './submit/53-2/submit_pt/modelSubmit_2.pth'
    # # model = Model_2()
    # # model.load_state_dict(torch.load(model_address, map_location=DEVICE))
    # # model_2 =model.to(DEVICE)

    # # model_address = './submit/51/submit_pt/modelSubmit_2.pth'
    # # model = Helpnet()
    # # model.load_state_dict(torch.load(model_address, map_location=DEVICE))
    # # model_3 =model.to(DEVICE)

    # # model_address = './submit/54_5/submit_pt/modelSubmit_2.pth'
    # # model = Model_2()
    # # model.load_state_dict(torch.load(model_address, map_location=DEVICE))
    # # model_4 =model.to(DEVICE)
    # trainX_unlabeled, trainX_labeled, trainY_labeled, testX_labeled, testY_labeled = get_900(k=0)
    # models = []
    # for i in range(1,13):
    #     # if i in [3,6,7,8,9,10,11,12]:
    #     #     continue
    #     model_address = f'./submit/58-{i}/submit_pt/modelSubmit_2.pth'
    #     model = Model_2(method_id=i)
    #     model.load_state_dict(torch.load(model_address, map_location=DEVICE))
    #     # model_1 =model.to(DEVICE)
    #     models.append(model)

    # test(testX_labeled, testY_labeled, *models)

    # # class EsembleNet():
    # #         self.net1 = Model_2()
    # #         self.net2 = Helpnet()



    # # ese = EsembleNet()

    # # state_dict = dict()

    # # ckpt1 = torch.load()
    # # ckpt2 = torch.load()

    # # for key in list(ckpt1.keys()):
    # #         new_key = 'net1.' + key 
    # #         state_dict[new_key] = ckpt1[key]


    # # ckpt2 ...

    # # ese.load_state_dict(state_dict)

    """model1"""
    #子模型#
    ckpt1 = torch.load('/data/cjz/location/submit/61-5-2-2-2-2-7796epochs/submit_pt/modelSubmit_1.pth')
    ckpt2 = torch.load('/data/cjz/location/submit/61-3-4/submit_pt/modelSubmit_1.pth')
    ckpt3 = torch.load('/data/cjz/location/submit/61-6/submit_pt/modelSubmit_1.pth')
    model1s = []
    model = Model_1(method_id=1)
    model.load_state_dict(ckpt1)
    model1s.append(model)

    model = Model_1(method_id=4)
    model.load_state_dict(ckpt2)
    model1s.append(model)

    model = Model_1(method_id=5)
    model.load_state_dict(ckpt3)
    model1s.append(model)


    # 集成模型##
    model1 = Model_1(method_id=0)
    model1.load_state_dict(torch.load('/data/cjz/location/submit/_ese/1/modelSubmit_1.pth'))
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
    test(testX, testY, model1)

    test(testX, testY, *model1s)