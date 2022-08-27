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
from utils import seed_everything
from datetime import datetime
import copy
class MyDataset(Dataset):
    def __init__(self, trainX,trainY,split_ratio):
        N = trainX.shape[0]
       
        TrainNum = int((N*(1-split_ratio)))
        self.x = trainX[:TrainNum].astype(np.float32)
        self.y = trainY[:TrainNum].astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        
        return (x, y)

class MyTestset(Dataset):
    def __init__(self, trainX,trainY,split_ratio):
        N = trainX.shape[0]
       
        TrainNum = int((N*(1-split_ratio)))
        self.x = trainX[TrainNum:].astype(np.float32)
        self.y = trainY[TrainNum:].astype(np.float32)

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        
        return (x, y)
 


# BATCH_SIZE = 100
# LEARNING_RATE = 0.001
TOTAL_EPOCHS = 10000
split_ratio = 0.1
change_learning_rate_epochs = 100


DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:0")


if __name__ == '__main__':
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    """注意评测设备只有一块gpu"""
    DEVICE=torch.device(f"cuda:{args.cuda}")
    BATCH_SIZE = args.bs
    LEARNING_RATE = args.lr
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
    copyfile('pytorch_Template/modelDesign_2.py', os.path.join(submit_path, 'modelDesign_2.py'))
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    """设置随机数种子"""
    seed_value = 1
    seed_everything(seed_value=seed_value)
    logging.info(f'seed_value:{seed_value}')
    """加载数据"""
    file_name1 = 'data/Case_3_Training.npy'
    logging.info('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    trainX_labeled = trainX[:1000,:,:,:]
    trainX_unlabeled = trainX[1000:,:,:,:]
    file_name2 = 'data/Case_3_Training_Label.npy'
    logging.info('The current dataset is : %s'%(file_name2))
    POS = np.load(file_name2)
    trainY_labeled = POS.transpose((1,0)) #[none, 2]

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
    """训练集数据扩增"""
    trainY_labeled_aug = trainY_labeled
    trainX_labeled_aug = trainX_labeled
    # for i in range(len(trainX_labeled)):
    for j in range(9):
        delete_num = np.random.choice(range(1,15),1)
        mask = np.random.choice(18,delete_num,replace=False)
        mask = np.concatenate((mask*4, mask*4+1, mask*4+2, mask*4+3))
        X = copy.deepcopy(trainX_labeled)
        X[:, :, mask, : ] = 0
        trainX_labeled_aug = np.concatenate((trainX_labeled_aug, X), axis = 0)
        trainY_labeled_aug = np.concatenate((trainY_labeled_aug, trainY_labeled), axis = 0)
    train_dataset = MyDataset(trainX_labeled_aug,trainY_labeled_aug,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyDataset(test_trainX_labeled,test_trainY_labeled,split_ratio=0)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    
    # train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio)
    # train_loader = DataLoader(dataset=train_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True)  # shuffle 标识要打乱顺序
    # test_dataset = MyTestset(trainX_labeled,trainY_labeled,split_ratio)
    # test_loader = DataLoader(dataset=test_dataset,
    #                                            batch_size=BATCH_SIZE,
    #                                            shuffle=True)  # shuffle 标识要打乱顺序




    criterion = nn.MSELoss().to(DEVICE)
    """加载模型"""
    model = Model_2(no_grad=False)
    model = model.to(DEVICE)
    logging.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)
    
    test_avg_min = 10000;
    for epoch in range(TOTAL_EPOCHS):
        model.train()       
        optimizer.param_groups[0]['lr'] = LEARNING_RATE /np.sqrt(np.sqrt(epoch+1))
        
        # Learning rate decay
        if (epoch + 1) % change_learning_rate_epochs == 0:
            optimizer.param_groups[0]['lr'] /= 2 
        logging.info('lr:%.4e' % optimizer.param_groups[0]['lr'])
           
        #Training in this epoch  
        loss_avg = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)
            
            # 清零
            optimizer.zero_grad()
            output = model(x)
            # 计算损失函数
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            loss_avg += loss.item() 
            
        loss_avg /= len(train_loader)
        
        #Testing in this epoch
        model.eval()
        test_avg = 0
        for i, (x, y) in enumerate(test_loader):
            x = x.float().to(DEVICE)
            y = y.float().to(DEVICE)

            output = model(x)
            # 计算损失函数
            loss_test = criterion(output, y)
            test_avg += loss_test.item() 
        
        test_avg /= len(test_loader)
        
        if test_avg < test_avg_min:
            logging.info('Model saved!')
            test_avg_min = test_avg
            model.to("cuda:0")
            torch.save(model.state_dict(), model_save)
            model.to(DEVICE)
        logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))