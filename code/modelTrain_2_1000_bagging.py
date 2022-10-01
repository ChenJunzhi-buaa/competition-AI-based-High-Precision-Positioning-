#--coding: utf-8--
"""
model2 完全按照西瓜书的bagging写的代码
"""

import torch
from torch.utils.data import  DataLoader
from modelDesign_2 import Helpnet, Model_2
import logging
from utils import *
from datetime import datetime
import copy


if __name__ == '__main__':

    """训弱分类器"""
    args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled = bootstrapping(id_path)
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    model = Helpnet(no_grad=False, if_classifier=args.classifier)
    # model = Model_2(no_grad=False, if_classifier=args.classifier)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    logging.info(model)
    train(args, model, 10000, args.begin_epochs, train_loader, model_save, test_loader, save=True, testX=test_trainX_labeled, testY=test_trainY_labeled, )
    logging.info(datetime.now())


    """bagging"""

    args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled = bootstrapping(id_path)
    
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    help_models = []
    help_model = Helpnet(no_grad=False, if_classifier=args.classifier)
    for i in range(5):
        logging.info(f"加载第{i+1}个弱模型")
        model_path = f"submit/54_{i+1}/submit_pt/modelSubmit_2.pth"
        help_model.load_state_dict(torch.load(model_path))
        # help_model = help_model.to(torch.device(f"cuda:{args.cuda}"))
        help_models.append(copy.deepcopy(help_model))
    X_pselabeled, Y_pselabeled = label(args, trainX_unlabeled,1000,*help_models )
    trainX_labeled = torch.concat((trainX_labeled, X_pselabeled), axis=0)
    trainY_labeled = torch.concat((trainY_labeled, Y_pselabeled), axis=0)
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序

    model = Model_2(no_grad=False, if_classifier=args.classifier)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    args.rlrp = True
    train(args, model, 10000, args.begin_epochs, train_loader, model_save, save=True, test_loader=test_loader, testX=test_trainX_labeled, testY=test_trainY_labeled )
    logging.info(datetime.now())