# TODO 
# 数据增强：全用和只用4个；用4、5、6、---、16个
# 改loss，改为dist和评测对应的
# 分测试集之前，打乱一下可能比较好
# 从某个checkpoint开始训练
# 模型集成bagging，5折
# 把模型变大
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
from utils import seed_everything
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
 


BATCH_SIZE = 100
LEARNING_RATE = 0.001
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
    args = parser.parse_args()
    """保存好要提交的文件、训练代码、训练日志"""
    id_path = os.path.join('submit',str(args.submit_id))
    if not os.path.exists(id_path):
        os.mkdir(id_path)
    submit_path = os.path.join(id_path, 'submit_pt')
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)
    logging.basicConfig(filename=os.path.join(id_path,"model1_log.txt"), filemode='w', level=logging.DEBUG)
    model_save = os.path.join(submit_path,'modelSubmit_1.pth')
    copyfile('pytorch_Template/modelDesign_1.py', os.path.join(submit_path, 'modelDesign_1.py'))
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    """设置随机数种子"""
    seed_value = 1
    seed_everything(seed_value=seed_value)
    logging.info(f'seed_value:{seed_value}')
    """加载数据"""
    file_name1 = 'data/Case_1_2_Training.npy'
    logging.info('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    file_name2 = 'data/Case_1_2_Training_Label.npy'
    logging.info('The current dataset is : %s'%(file_name2))
    POS = np.load(file_name2)
    trainY = POS.transpose((1,0)) #[none, 2]
    """打乱数据顺序"""
    index = np.arange(len(trainX))
    np.random.shuffle(index)
    trainX = trainX[index]
    trainY = trainY[index]

    model = Model_1()
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

            # torch.save(model, model_save)
            model.to("cuda:0")
            torch.save(model.state_dict(), model_save)
            model.to(DEVICE)
        logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))

