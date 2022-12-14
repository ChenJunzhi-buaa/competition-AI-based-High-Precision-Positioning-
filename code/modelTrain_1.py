#--coding: utf-8--
"""
model1的训练文件
"""
from utils import seed_everything

from copy import copy
import numpy as np

import torch
from torch.utils.data import DataLoader
from modelDesign_1 import Model_1
import logging
import argparse
import os
from shutil import copyfile
from datetime import datetime
import copy
from utils import train, MyDataset

if __name__ == '__main__':
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--rlrp', default=False, action='store_true', help = 'ReduceLROnPlateau' )
    parser.add_argument('--calr', default=False, action='store_true', help = 'CosineAnnealingLR' )
    parser.add_argument('--cawr', default=False, action='store_true', help = 'CosineAnnealingWarmRestarts' )
    parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    parser.add_argument('--seed', default=42, type=int )
    parser.add_argument('--no_seed', default=True, action = 'store_false' )
    parser.add_argument('--epochs', default=10000, type=int)

    parser.add_argument('--classifier', default=False, action='store_true' )
    parser.add_argument('--change_learning_rate_epochs', default=100, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true' )
    parser.add_argument('--method_id', default=1, type=int, help="the method id  ")
    parser.add_argument('--no_test', default=False, action = 'store_true' )
    
    parser.add_argument('--half_pre', default=False, action = 'store_true' )
    args = parser.parse_args()
    """注意评测设备只有一块gpu"""
    DEVICE=torch.device(f"cuda:{args.cuda}")
    BATCH_SIZE = args.bs
    LEARNING_RATE = args.lr
    split_ratio = args.sr
    """保存好要提交的文件、训练代码、训练日志"""
    id_path = os.path.join('submit',str(args.submit_id))
    if not os.path.exists(id_path):
        os.mkdir(id_path)
    submit_path = os.path.join(id_path, 'submit_pt')
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)
    logging.basicConfig(filename=os.path.join(id_path,"model1_log.txt"), filemode='w', level=logging.DEBUG)
    logging.info(datetime.now())
    logging.info(args)
    model_save = os.path.join(submit_path,'modelSubmit_1.pth')
    copyfile('pytorch_Template/modelDesign_1.py', os.path.join(submit_path, 'modelDesign_1.py'))
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    """设置随机数种子"""
    if args.no_seed == False:
        seed_value = args.seed
        seed_everything(seed_value=seed_value)
        logging.info(f'seed_value:{seed_value}')
    else:
        logging.info(f'不设定可复现')
    """加载数据"""
    file_name1 = 'data/Case_1_2_Training.npy'
    logging.info('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    file_name2 = 'data/Case_1_2_Training_Label.npy'
    logging.info('The current dataset is : %s'%(file_name2))
    POS = np.load(file_name2)
    trainY = POS.transpose((1,0)) #[none, 2]


    """数据扩增"""
    trainX_copy = copy.deepcopy(trainX)
    trainX_copy[:,:,4:20,:]=0
    trainX_copy[:,:,24:48,:]=0
    trainX_copy[:,:,52:68,:]=0
    trainX = np.concatenate((trainX, trainX_copy), axis=0)
    trainY = np.concatenate((trainY, trainY), axis=0)
    
    """打乱数据顺序"""
    index = np.arange(len(trainX))
    np.random.shuffle(index)
    trainX = trainX[index]
    trainY = trainY[index]
    """加载模型"""
    model = Model_1(no_grad=False, method_id=args.method_id)
    # model.load_state_dict(torch.load('submit/61-2/submit_pt/modelSubmit_1.pth',))
    model = model.to(DEVICE)
    logging.info(model)
    if args.half_pre == True:
        date_type = torch.float16
        model = model.half()
    else:
        date_type = torch.float32
    if args.no_test == False:
        train_num = int(len(trainX) * (1-split_ratio))
        testX = torch.tensor(trainX[train_num:], dtype = date_type)
        testY = torch.tensor(trainY[train_num:], dtype = date_type)
        trainX = torch.tensor(trainX[:train_num], dtype = date_type)
        trainY = torch.tensor(trainY[:train_num], dtype = date_type)

        train_dataset = MyDataset(trainX,trainY,split_ratio=0)
        train_loader = DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin_memory)  # shuffle 标识要打乱顺序
        test_dataset = MyDataset(testX,testY,split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin_memory)  # shuffle 标识要打乱顺序
        train(args, model, 10000, args.epochs, train_loader, model_save, test_loader, True, testX, testY )
    else:
        trainX = torch.tensor(trainX, dtype = date_type)
        trainY = torch.tensor(trainY, dtype = date_type)
        train_dataset = MyDataset(trainX,trainY,split_ratio=0)
        train_loader = DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=args.pin_memory)  # shuffle 标识要打乱顺序
        train(args, model, 10000, args.epochs, train_loader, model_save,save = True,)
logging.info(datetime.now())


