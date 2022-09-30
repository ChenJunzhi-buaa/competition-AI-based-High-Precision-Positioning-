#--coding: utf-8--
"""评测脚本"""
import numpy as np
import torch
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
    
    return score(y_test_avg)

if __name__ == '__main__':
    from modelDesign_1 import Model_1



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