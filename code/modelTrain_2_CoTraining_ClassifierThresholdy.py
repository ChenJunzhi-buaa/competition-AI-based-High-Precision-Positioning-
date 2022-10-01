#--coding: utf-8--
"""
两个模型一起训练的半监督策略。
A给无标签数据打标签,训B;
B再给无标签数据打标签,训A;
不断循环。

实际效果很一般，
"""
import numpy as np
import torch
from torch.utils.data import  DataLoader
from modelDesign_2 import Model_2, Model_2_18
import logging
import os
from shutil import copyfile
import argparse
from utils import *
from datetime import datetime

if __name__ == '__main__':
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--big_bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--rlrp', default=False, action='store_true' )
    parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    parser.add_argument('--seed', default=1, type=int )
    parser.add_argument('--classifier', default=True, action='store_false' )
    parser.add_argument('--begin_epochs', default=10000, type=int)
    parser.add_argument('--co_epochs', default=500, type=int)
    parser.add_argument('--no_seed', default=False, action = 'store_true' )
    parser.add_argument('--change_learning_rate_epochs', default=100, type=int)
    args = parser.parse_args()
    """注意评测设备只有一块gpu"""
    DEVICE=torch.device(f"cuda:{args.cuda}")
    split_ratio = args.sr
    """保存好要提交的文件、训练代码、训练日志"""
    id_path = os.path.join('submit',str(args.submit_id))
    if not os.path.exists(id_path):
        os.mkdir(id_path)
    submit_path = os.path.join(id_path, 'submit_pt')
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)
    logging.basicConfig(filename=os.path.join(id_path,"model2_log.txt"), filemode='w', level=logging.DEBUG)
    logging.info(datetime.now())
    logging.info(args)
    model_save = os.path.join(submit_path,'modelSubmit_2.pth')
    model_save_A = os.path.join(id_path,'modelSubmit_2_A.pth')
    model_save_B = os.path.join(id_path,'modelSubmit_2_B.pth')
    
    copyfile('pytorch_Template/modelDesign_2.py', os.path.join(submit_path, 'modelDesign_2.py'))
    copyfile('pytorch_Template/utils.py', os.path.join(id_path, 'utils.py'))
    if '/' in __file__:
        copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    elif '\\' in __file__:
        copyfile(__file__, os.path.join(id_path, __file__.split('\\')[-1]))
    """设置随机数种子"""
    if args.no_seed == False:
        seed_value = args.seed
        seed_everything(seed_value=seed_value)
        logging.info(f'seed_value:{seed_value}')
    else:
        logging.info(f'不设定可复现')
    """加载数据"""
    file_name1 = os.path.join('data','Case_3_Training.npy')
    logging.info('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    trainX_labeled = trainX[:1000,:,:,:]
    trainX_unlabeled = trainX[1000:,:,:,:]
    file_name2 = os.path.join('data','Case_3_Training_Label.npy')
    logging.info('The current dataset is : %s'%(file_name2))
    POS = np.load(file_name2)
    trainY_labeled = POS.transpose((1,0)) #[none, 2]

    """构造分类标签"""
    if args.classifier == True:
        trainY_class_labeled = get_class_label(trainY_labeled)
        trainY_labeled = np.concatenate((trainY_labeled, trainY_class_labeled), axis=1)
    """转化为tensor"""
    trainX_labeled = torch.tensor(trainX_labeled)
    trainY_labeled = torch.tensor(trainY_labeled)
    trainX_unlabeled = torch.tensor(trainX_unlabeled)


    """打乱数据顺序"""
    index_L = np.arange(len(trainX_labeled))
    np.random.shuffle(index_L)
    trainX_labeled = trainX_labeled[index_L]
    trainY_labeled = trainY_labeled[index_L]
    index_U = np.arange(len(trainX_unlabeled))
    np.random.shuffle(index_U)
    trainX_unlabeled = trainX_unlabeled[index_U]


    
    """分出测试集"""
    test_trainX_labeled = trainX_labeled[0:int(split_ratio*len(trainX_labeled))]
    test_trainY_labeled = trainY_labeled[0:int(split_ratio*len(trainY_labeled))]

    """分出训练集"""
    trainX_labeled = trainX_labeled[int(split_ratio*len(trainX_labeled)):]
    trainY_labeled = trainY_labeled[int(split_ratio*len(trainY_labeled)):]
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.bs,
                                               shuffle=True)  # shuffle 标识要打乱顺序


    """加载模型"""
    modelA = Model_2(no_grad=False, if_classifier=args.classifier)
    modelB = Model_2_18(no_grad=False, if_classifier=args.classifier)
    modelA = modelA.to(DEVICE)
    modelB = modelB.to(DEVICE)
    logging.info(modelA)
    logging.info(modelB)
    
    test_avg_min = 10000;  

    """train A"""
    logging.info("###################### train A ##################################")
    test_avg_min_A = train(args, modelA, 10000, args.begin_epochs, train_loader, test_loader, model_save_A, True)
    """train B"""
    logging.info("##################### train B#######################################")
    test_avg_min_B = train(args, modelB, 10000, args.begin_epochs, train_loader, test_loader, model_save_B, True)
    """"训练NUMBER轮"""
    NUMBER =  10
    """改学习率策略"""
    args.rlrp = True
    for i in range(NUMBER):
        logging.info(f" ")
        logging.info(f"############## i : {i} ###############################################")
        """label by A """
        logging.info("############## label by A ###############################################")
        with torch.no_grad():
            X_pselabeled_good, Y_pselabeled_good = label(args,trainX_unlabeled, modelA)
        logging.info(f"good label numbers:{X_pselabeled_good.shape[0]}")
        train_dataset = MyDataset(torch.concat((trainX_labeled, X_pselabeled_good), axis=0), torch.concat((trainY_labeled, Y_pselabeled_good), axis=0), split_ratio=0)
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.big_bs,shuffle=True)  # shuffle 标识要打乱顺序
        """train B"""
        logging.info("############## train B ###############################################")
        test_avg_min_B = train(args, modelB, test_avg_min_B, args.co_epochs, train_loader, test_loader, model_save_B, False)

        """label by B """
        logging.info("############## label by B ###############################################")
        with torch.no_grad():
            X_pselabeled_good, Y_pselabeled_good = label(args, trainX_unlabeled, modelB,)
        logging.info(f"good label numbers:{X_pselabeled_good.shape[0]}")
        train_dataset = MyDataset(torch.concat((trainX_labeled, X_pselabeled_good), axis=0), torch.concat((trainY_labeled, Y_pselabeled_good), axis=0), split_ratio=0)
        train_loader = DataLoader(dataset=train_dataset,batch_size=args.big_bs,shuffle=True)  # shuffle 标识要打乱顺序
        """train A"""
        logging.info("############## train A ###############################################")
        test_avg_min_A = train(args, modelA, test_avg_min_A, args.co_epochs, train_loader, test_loader, model_save, True)






