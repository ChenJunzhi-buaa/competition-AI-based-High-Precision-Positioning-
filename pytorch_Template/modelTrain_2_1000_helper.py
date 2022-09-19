#--coding: utf-8--

# TODO
# seed随机数种子的选择可能挺有用
# 虚拟标签的应该在那个120，60的范围内
# 转化为分类问题
# 数据归一化
# BN层
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


if __name__ == '__main__':

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
    
    criterion = nn.MSELoss().to(torch.device(f"cuda:{args.cuda}"))
    if args.classifier == True:
        criterion_classifier = nn.CrossEntropyLoss().to(torch.device(f"cuda:{args.cuda}"))
    """加载模型"""
    model = Helpnet(no_grad=False, if_classifier=args.classifier)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    logging.info(model)
   
    train(args, model, 10000, args.begin_epochs, train_loader,  model_save, test_loader=test_loader, save=True)
    logging.info(datetime.now())


