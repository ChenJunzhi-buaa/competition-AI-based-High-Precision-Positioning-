#--coding: utf-8--
"""
一个模型先用有标签数据训练好，给无标签数据打标签（利用分类头获得标签的置信度，标签置信度足够高的无标签数据才用来继续训练）；
混合数据再训练这个模型；
再打标签；
不断循环。

"""
import torch
from torch.utils.data import  DataLoader
from modelDesign_2 import Model_2
import logging
import os
from utils import *
import math

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

    test_avg_min = 100000
    """1000个数据训练好"""

    model = Model_2(no_grad=False, if_classifier=args.classifier)
    model = model.to(torch.device(f"cuda:{args.cuda}"))  
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
