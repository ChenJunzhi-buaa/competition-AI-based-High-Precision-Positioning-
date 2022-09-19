#--coding: utf-8--

# TODO
# seed随机数种子的选择可能挺有用
# 虚拟标签的应该在那个120，60的范围内
# 转化为分类问题
# 数据归一化
# BN层
from email.policy import default
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from modelDesign_2 import Model_2
import logging
import os
from shutil import copyfile
import argparse
from utils import *
from datetime import datetime
import copy
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import math
# class MyDataset(Dataset):
#     def __init__(self, trainX,trainY,split_ratio):
#         N = trainX.shape[0]
       
#         TrainNum = int((N*(1-split_ratio)))
#         self.x = trainX[:TrainNum].float()
#         self.y = trainY[:TrainNum].float()

#         self.len = len(self.y)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):     
#         x = self.x[idx]
#         y = self.y[idx]
        
#         return (x, y)

# class MyTestset(Dataset):
#     def __init__(self, trainX,trainY,split_ratio):
#         N = trainX.shape[0]
       
#         TrainNum = int((N*(1-split_ratio)))
#         self.x = trainX[TrainNum:].float()
#         self.y = trainY[TrainNum:].float()

#         self.len = len(self.y)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):     
#         x = self.x[idx]
#         y = self.y[idx]
        
#         return (x, y)
 



if __name__ == '__main__':
    # """命令行参数"""
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--submit_id', type=str, required=True)
    # parser.add_argument('--cuda', default=0)
    # parser.add_argument('--bs', type=int, default=32)
    # parser.add_argument('--big_bs', type=int, default=256)
    # parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--weight_decay', type=float, default=1e-4)
    # parser.add_argument('--rlrp', default=False, action='store_true' )
    # parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    # parser.add_argument('--seed', default=1, type=int )
    # parser.add_argument('--classifier', default=True, action='store_false' )
    # parser.add_argument('--epochs', default=2000, type=int)
    # parser.add_argument('--begin_epochs', default=10000, type=int)
    # parser.add_argument('--no_seed', default=False, action = 'store_true' )
    # parser.add_argument('--change_learning_rate_epochs', default=100, type=int)
    # args = parser.parse_args()
    # TOTAL_EPOCHS = args.epochs
    # """注意评测设备只有一块gpu"""
    # split_ratio = args.sr
    # """保存好要提交的文件、训练代码、训练日志"""
    # id_path = os.path.join('submit',str(args.submit_id))
    # if not os.path.exists(id_path):
    #     os.mkdir(id_path)
    # submit_path = os.path.join(id_path, 'submit_pt')
    # if not os.path.exists(submit_path):
    #     os.mkdir(submit_path)
    # logging.basicConfig(filename=os.path.join(id_path,"model2_log.txt"), filemode='w', level=logging.DEBUG)
    # logging.info(datetime.now())
    # logging.info(args)
    # model_save = os.path.join(submit_path,'modelSubmit_2.pth')
    # copyfile('pytorch_Template/modelDesign_2.py', os.path.join(submit_path, 'modelDesign_2.py'))
    # copyfile('pytorch_Template/utils.py', os.path.join(id_path, 'utils.py'))
    # copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    # """设置随机数种子"""
    # if args.no_seed == False:
    #     seed_value = args.seed
    #     seed_everything(seed_value=seed_value)
    #     logging.info(f'seed_value:{seed_value}')
    # else:
    #     logging.info(f'不设定可复现')
    # """加载数据"""
    # # file_name1 = 'data/Case_3_Training.npy'
    # # logging.info('The current dataset is : %s'%(file_name1))
    # # CIR = np.load(file_name1)
    # # trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    # # trainX_labeled = trainX[:1000,:,:,:]
    # # trainX_unlabeled = trainX[1000:,:,:,:]
    # # file_name2 = 'data/Case_3_Training_Label.npy'
    # # logging.info('The current dataset is : %s'%(file_name2))
    # # POS = np.load(file_name2)
    # # trainY_labeled = POS.transpose((1,0)) #[none, 2]

    
    

    # # if args.classifier == True:
    # #     trainY_class_labeled = get_class_label(trainY_labeled)
    # #     trainY_labeled = np.concatenate((trainY_labeled, trainY_class_labeled), axis=1)
    # # """转化为tensor"""
    # # trainX_labeled = torch.tensor(trainX_labeled)
    # # trainY_labeled = torch.tensor(trainY_labeled)
    # # trainX_unlabeled = torch.tensor(trainX_unlabeled)


    # # """打乱数据顺序"""
    # # index_L = np.arange(len(trainX_labeled))
    # # np.random.shuffle(index_L)
    # # trainX_labeled = trainX_labeled[index_L]
    # # trainY_labeled = trainY_labeled[index_L]
    # # index_U = np.arange(len(trainX_unlabeled))
    # # np.random.shuffle(index_U)
    # # trainX_unlabeled = trainX_unlabeled[index_U]


    
    # # """分出测试集"""
    # # test_trainX_labeled = trainX_labeled[0:int(split_ratio*len(trainX_labeled))]
    # # test_trainY_labeled = trainY_labeled[0:int(split_ratio*len(trainY_labeled))]

    # # """分出训练集"""
    # # trainX_labeled = trainX_labeled[int(split_ratio*len(trainX_labeled)):]
    # # trainY_labeled = trainY_labeled[int(split_ratio*len(trainY_labeled)):]
    # trainX_labeled = torch.tensor(np.load('data/case3/Case_3_Training_train.npy'), dtype=torch.float)
    # trainY_labeled = torch.tensor(np.load('data/case3/Case_3_Training_train_label.npy'), dtype=torch.float)
    # test_trainX_labeled = torch.tensor(np.load('data/case3/Case_3_Training_test.npy'), dtype=torch.float)
    # test_trainY_labeled = torch.tensor(np.load('data/case3/Case_3_Training_test_label.npy'), dtype=torch.float)
    # trainX_unlabeled = torch.tensor(np.load('data/case3/Case_3_unlabeled.npy'), dtype = torch.float)

    args, id_path, submit_path,  trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled, model_save = pre2()
    """训练集数据扩增"""
    # trainX_labeled, trainY_labeled = Model_2().data_aug(x = trainX_labeled, y = trainY_labeled)
    """测试集数据扩增"""
    # test_trainX_labeled, test_trainY_labeled = Model_2().data_aug(x = test_trainX_labeled, y = test_trainY_labeled)
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序

    test_avg_min = 100000
    """1000个数据训练好"""

    model = Model_2(no_grad=False, if_classifier=args.classifier)
    model = model.to(2)  
    model_save_ = os.path.join(id_path,'modelSubmit_2_1000.pth')
    test_avg_min = train(args, model , test_avg_min, args.begin_epochs, train_loader, test_loader, model_save_, True)
    """加载较好的训练模型"""
    # model = Model_2(no_grad=False, if_classifier=args.classifier)
    # model_path = 'submit/47-1/submit_pt/modelSubmit_2.pth'
    # model.load_state_dict(torch.load(model_path))
    # model = model.to(torch.device(f"cuda:{args.cuda}"))   
    
    """"训练NUMBER轮"""
    NUMBER =  12
    args.rlrp = True
    for i in range(NUMBER):
        """打标签 """
        logging.info("############## 打标签 ###############################################")
        good_num = (i+1)*1000
        with torch.no_grad():
            X_pselabeled_good, Y_pselabeled_good = label(args, trainX_unlabeled, model, args.bs, good_num=good_num, )
        logging.info(f"找到的好标签数目{X_pselabeled_good.shape[0]}")
        """新旧数据一起训练"""
        logging.info(f"############## 混合数据第{i}次一起训练 ###############################################")
        
        train_dataset = MyDataset(torch.concat((trainX_labeled, X_pselabeled_good), axis=0), torch.concat((trainY_labeled, Y_pselabeled_good), axis=0), split_ratio=0)
        train_loader = DataLoader(dataset=train_dataset,batch_size=math.ceil(256*(i+1)/NUMBER/32)*32,shuffle=True)  # shuffle 标识要打乱顺序
        model = Model_2(no_grad=False, if_classifier=args.classifier)

        model = model.to(torch.device(f"cuda:{args.cuda}"))
        test_avg_min = train(args, model, test_avg_min, args.epochs, train_loader, test_loader, model_save )
