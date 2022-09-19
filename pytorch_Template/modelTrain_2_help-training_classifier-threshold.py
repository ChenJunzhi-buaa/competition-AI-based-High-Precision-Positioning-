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
from modelDesign_2 import *
import logging
import os
from shutil import copyfile
import argparse
from utils import *
from datetime import datetime
import copy
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import math


if __name__ == '__main__':

    
    args, id_path, submit_path,  trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled, model_save = pre2()
    """训练集数据扩增"""
    # trainX_labeled, trainY_labeled = Model_2().data_aug(x = trainX_labeled, y = trainY_labeled)
    """测试集数据扩增"""
    # test_trainX_labeled, test_trainY_labeled = Model_2().data_aug(x = test_trainX_labeled, y = test_trainY_labeled)
    # train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    # train_loader = DataLoader(dataset=train_dataset,
    #                                            batch_size=args.bs,
    #                                            shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序

    test_avg_min = 100000


    model = Model_2(no_grad=False, if_classifier=args.classifier)
    logging.info(model)
    model = model.to(torch.device(f"cuda:{args.cuda}"))  
    # model_save_ = os.path.join(id_path,'modelSubmit_2_1000.pth')
    # test_avg_min = train(args, model , test_avg_min, args.begin_epochs, train_loader, test_loader, model_save_, True)
    """加载较强的训练模型"""
    # model = Model_2(no_grad=False, if_classifier=args.classifier)
    # model_path = 'submit/47-1/submit_pt/modelSubmit_2.pth'
    # model.load_state_dict(torch.load(model_path))
    # model = model.to(torch.device(f"cuda:{args.cuda}"))   
    good_model = Helpnet(no_grad=False, if_classifier=args.classifier)
    logging.info(good_model)
    model_path = 'submit/51/submit_pt/modelSubmit_2.pth'
    good_model.load_state_dict(torch.load(model_path))
    good_model = good_model.to(torch.device(f"cuda:{args.cuda}"))
    """"强模型打标签"""
    logging.info(f"############## 强模型打标签 ###############################################")
    good_num = 14000
    X_pselabeled_good, Y_pselabeled_good = label(args, trainX_unlabeled, good_model, args.bs, good_num=good_num, )
 
    args.rlrp = True
    """新旧数据一起训练"""
    logging.info(f"############## 混合数据一起训练 ###############################################")
    
    train_dataset = MyDataset(torch.concat((trainX_labeled, X_pselabeled_good), axis=0), torch.concat((trainY_labeled, Y_pselabeled_good), axis=0), split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,batch_size=args.big_bs,shuffle=True)  # shuffle 标识要打乱顺序
    logging.info(f"混合数据总数{len(train_dataset)}")
    test_avg_min = train(args, model, test_avg_min, args.epochs, train_loader, test_loader, model_save )
    logging.info(datetime.now())