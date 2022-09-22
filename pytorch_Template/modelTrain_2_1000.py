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
from modelDesign_2 import Model_2
import logging
import os
from shutil import copyfile
import argparse
from utils import *
from datetime import datetime
import copy
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau


if __name__ == '__main__':
    """带测试集"""
    args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    trainX_unlabeled, trainX_labeled, trainY_labeled, testX_labeled, testY_labeled = get_900(k=args.k)
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(testX_labeled,testY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    model = Model_2(no_grad=False, if_classifier=args.classifier, method_id=args.method_id)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    logging.info(model)
   
    train(args, model, 10000, args.begin_epochs, train_loader, model_save,test_loader=test_loader,save=True, testX = testX_labeled, testY= testY_labeled)
    logging.info(datetime.now())
    """不带测试集"""
    # args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    # copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    # trainX_unlabeled, trainX_labeled, trainY_labeled = get_1000()
    # train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    # train_loader = DataLoader(dataset=train_dataset,
    #                                            batch_size=args.bs,
    #                                            shuffle=True)  # shuffle 标识要打乱顺序
    # criterion = nn.MSELoss().to(torch.device(f"cuda:{args.cuda}"))
    # model = Model_2(no_grad=False, if_classifier=args.classifier)
    # model = model.to(torch.device(f"cuda:{args.cuda}"))
    # logging.info(model)
   
    # train(args, model, 10000, args.begin_epochs, train_loader, model_save,save=True,)
    # logging.info(datetime.now())