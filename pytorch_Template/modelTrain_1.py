# TODO 
# 数据增强：全用和只用4个；用4、5、6、---、16个
# 改loss，改为dist和评测对应的
# 分测试集之前，打乱一下可能比较好
# 从某个checkpoint开始训练
# 模型集成bagging，5折
# 把模型变大
# 时域频域变换
# 归0的和不归0的，数据扩增，或许还能用在model2
from utils import seed_everything

from copy import copy
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modelDesign_1 import Model_1
import logging
import argparse
import os
from shutil import copyfile
from datetime import datetime
import copy

from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau,CosineAnnealingLR,CosineAnnealingWarmRestarts
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
    parser.add_argument('--weight_decay', type=float, default=2e-4)
    parser.add_argument('--rlrp', default=False, action='store_true', help = 'ReduceLROnPlateau' )
    parser.add_argument('--calr', default=False, action='store_true', help = 'CosineAnnealingLR' )
    parser.add_argument('--cawr', default=False, action='store_true', help = 'CosineAnnealingWarmRestarts' )
    parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    parser.add_argument('--seed', default=42, type=int )
    parser.add_argument('--no_seed', default=False, action = 'store_true' )
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
    model = Model_1(no_grad=False)
    # model.load_state_dict(torch.load('submit/46-2-1/submit_pt/modelSubmit_1.pth',))
    model = model.to(DEVICE)
    logging.info(model)
    
    train_dataset = MyDataset(trainX,trainY,split_ratio)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyTestset(trainX,trainY,split_ratio)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    criterion = nn.MSELoss().to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.9)
    if args.rlrp == True:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=30,)
    if args.calr == True:
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
    if args.cawr == True:
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2)
    test_avg_min = 10000;
    for epoch in range(TOTAL_EPOCHS):
        model.train()       
        if args.rlrp == False and args.calr == False and args.cawr == False:
          
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
        with torch.no_grad():
            test_avg = 0
            for i, (x, y) in enumerate(test_loader):
                x = x.float().to(DEVICE)
                y = y.float().to(DEVICE)

                output = model(x)
                # 计算损失函数
                loss_test = criterion(output, y)
                test_avg += loss_test.item() 
            
            test_avg /= len(test_loader)

            """更新学习率"""
            if args.rlrp == True:
                scheduler.step(test_avg) 
            if args.calr == True or args.cawr == True:
                scheduler.step() 

            if test_avg < test_avg_min:
                logging.info('Model saved!')
                test_avg_min = test_avg

                # torch.save(model, model_save)
                model.to("cuda:0")
                torch.save(model.state_dict(), model_save)
                model.to(DEVICE)
            logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))

logging.info(datetime.now())


