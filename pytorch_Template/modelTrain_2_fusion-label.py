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
from multiprocessing import Pool

if __name__ == '__main__':

    """带测试集"""
    args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    trainX_unlabeled, trainX_labeled, trainY_labeled, testX_labeled, testY_labeled = get_900(k=args.k)
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    def f(args):
        models = []
        """自己的弱模型"""
        for i in range(1,24):
            if i != 13 and  i !=14:
                model_address = f'./submit/58-{i}/submit_pt/modelSubmit_2.pth'
                model = Model_2(method_id=i)
                model.load_state_dict(torch.load(model_address))
                # model_1 =model.to(DEVICE)
                models.append(model)

        """自己的集成模型"""
        for i in [1, 4]:
            folder_name = f'59-{i}'
            while os.path.exists(os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')):

                model_address = f'./submit/59-{i}/submit_pt/modelSubmit_2.pth'
                logging.info(f'集成已有的集成模型的路径: {model_address}')
                model = Model_2(method_id=i)
                model.load_state_dict(torch.load(model_address))
                # model_1 =model.to(DEVICE)
                models.append(model)

                folder_name = folder_name + f'-{i}'


        Y_pselabeled_ave = label(args, trainX_unlabeled, 1000, True, 98.05, True, *models)



        torch.save(Y_pselabeled_ave, os.path.join(id_path, 'Y_pselabeled_ave.pth'))

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    """完全释放显存的写法""" 
    # 参考https://www.cnblogs.com/dechinphy/p/gc.html
    with Pool(1) as p: 
        red = p.map(f, [args])
    Y_pselabeled_ave = torch.load(os.path.join(id_path, 'Y_pselabeled_ave.pth'))
    if args.no_test == True:
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled, testX_labeled), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave, testY_labeled), axis=0)
    else:
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave), axis=0)
        test_dataset = MyDataset(testX_labeled,testY_labeled,split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序


    train_dataset = MyDataset(confusion_X,confusion_Y,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序
        
    
    model = Model_2(no_grad=False, if_classifier=args.classifier,method_id=args.method_id)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    logging.info(model)
    if args.no_test == True:
        train(args, model, 10000, args.begin_epochs, train_loader, model_save,save=True)
    else:
        train(args, model, 10000, args.begin_epochs, train_loader, model_save,test_loader=test_loader,save=True, testX = testX_labeled, testY= testY_labeled)
    logging.info(datetime.now())
    