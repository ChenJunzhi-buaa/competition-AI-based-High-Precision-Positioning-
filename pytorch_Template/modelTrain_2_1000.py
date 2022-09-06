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
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
class MyDataset(Dataset):
    def __init__(self, trainX,trainY,split_ratio):
        N = trainX.shape[0]
       
        TrainNum = int((N*(1-split_ratio)))
        self.x = trainX[:TrainNum].float()
        self.y = trainY[:TrainNum].float()

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
        self.x = trainX[TrainNum:].float()
        self.y = trainY[TrainNum:].float()

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
# split_ratio = 0.1
change_learning_rate_epochs = 100


DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:0")


if __name__ == '__main__':
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--rlrp', default=False, action='store_true' )
    parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    parser.add_argument('--seed', default=1, type=int )
    parser.add_argument('--classifier', default=False, action='store_true' )
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
    logging.basicConfig(filename=os.path.join(id_path,"model2_log.txt"), filemode='w', level=logging.DEBUG)
    logging.info(datetime.now())
    logging.info(args)
    model_save = os.path.join(submit_path,'modelSubmit_2.pth')
    copyfile('pytorch_Template/modelDesign_2.py', os.path.join(submit_path, 'modelDesign_2.py'))
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    """设置随机数种子"""
    seed_value = args.seed
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

    """构造分类标签"""
   
    def get_class_label(trainY_labeled):
         #标签
        class_label = [[0,1,2,3,4,5,],
                    [6,7,8,9,10,11,],
                    [12,13,14,15,16,17]
                    ]
        trainY_class_labeled = np.zeros((0,1), dtype=np.uint8)
        for x in range(trainY_labeled.shape[0]):
            location = trainY_labeled[x]
            location_0 = location[0]
            location_1 = location[1]
            if location_0 >=0 and location_0 < 20:
                class_label_index0 = 0
            elif location_0 >=20 and location_0 < 40:
                class_label_index0 = 1
            elif location_0 >= 40 and location_0 < 60:
                class_label_index0 = 2
            elif location_0 >= 60 and location_0 < 80:
                class_label_index0 = 3
            elif location_0 >= 80 and location_0 < 100:
                class_label_index0 = 4
            elif location_0 >= 100 and location_0 <= 120:
                class_label_index0 = 5

            if location_1 >= 0 and location_1 < 20:
                class_label_index1 = 0
            elif location_1 >= 20 and location_1 < 40:
                class_label_index1 = 1
            elif location_1 >= 40 and location_1 <= 60:
                class_label_index1 = 2
            trainY_class_labeled = np.concatenate((trainY_class_labeled, np.expand_dims(np.array([class_label[class_label_index1][class_label_index0]]), axis=0)),axis=0)
        return trainY_class_labeled
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
    """训练集数据扩增"""
    # trainX_labeled, trainY_labeled = Model_2().data_aug(x = trainX_labeled, y = trainY_labeled)
    """测试集数据扩增"""
    # test_trainX_labeled, test_trainY_labeled = Model_2().data_aug(x = test_trainX_labeled, y = test_trainY_labeled)
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio=0)
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
    if args.classifier == True:
        criterion_classifier = nn.CrossEntropyLoss().to(DEVICE)
    """加载模型"""
    model = Model_2(no_grad=False, if_classifier=args.classifier)
    model = model.to(DEVICE)
    logging.info(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=args.weight_decay)
    
    test_avg_min = 10000;
    if args.rlrp == True:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=30,)
    for epoch in range(TOTAL_EPOCHS):
        model.train()     
        if args.rlrp == False:
          
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

            if args.classifier == True:
                output_r, output_c = model(x)
                loss_r = criterion(output_r, y[:,:2])
                loss_c = criterion_classifier(output_c, y[:,2].long())
                loss =  + loss_r + loss_c
                logging.info(f"回归训损失{loss_r:.4f}，分类训练损失{loss_c:.4f}")
            else:
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

            if args.classifier == True:
                output_r, output_c = model(x)
                # loss_test = criterion(output_r, y[:,:2]) + criterion_classifier(output_c, y[:,2].long())
                loss_test = criterion(output_r, y[:,:2])
                
            else:
                output = model(x)
                # 计算损失函数
                loss_test = criterion(output, y)
            test_avg += loss_test.item() 
        
        test_avg /= len(test_loader)
        """更新学习率"""
        if args.rlrp == True:
            scheduler.step(test_avg) 
        if test_avg < test_avg_min:
            logging.info('Model saved!')
            test_avg_min = test_avg
            model.to("cuda:0")
            torch.save(model.state_dict(), model_save)
            model.to(DEVICE)
        logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
    logging.info(datetime.now())


