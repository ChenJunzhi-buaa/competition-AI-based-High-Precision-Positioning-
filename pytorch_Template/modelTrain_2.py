import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from modelDesign_2 import Model_2
import logging
logging.basicConfig(filename="pytorch_Template/model2_log.txt", filemode='w', level=logging.DEBUG)
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
TOTAL_EPOCHS = 100
split_ratio = 0.1
change_learning_rate_epochs = 100

model_save = 'pytorch_Template/modelSubmit_2.pth'

"""注意评测设备只有一块gpu"""
DEVICE=torch.device("cpu")
if torch.cuda.is_available():
    DEVICE=torch.device("cuda:1")


if __name__ == '__main__':

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

    model = Model_2()
    model = model.to(DEVICE)
    logging.info(model)
    logging.info('######################################################################')
    logging.info('标签数据训练出一个最佳模型')
    train_dataset = MyDataset(trainX_labeled,trainY_labeled,split_ratio)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyTestset(trainX_labeled,trainY_labeled,split_ratio)
    test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    criterion = nn.L1Loss().to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
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
            torch.save(model.state_dict(), model_save)
        logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
    #torch.save(model model_save)
    #torch.save(model.state_dict(), model_save)

# else:
#     logging.info("load torch model")
#     model_ckpt = torch.load(model, model_save)
    logging.info('######################################################################')
    logging.info('无标签数据打上伪标签')
    model.load_state_dict(torch.load(model_save))
    model = model.to(DEVICE)
    model.eval()
    trainX_unlabeled = torch.from_numpy(trainX_unlabeled.astype(np.float32))
    batch_num = trainX_unlabeled.shape[0]//BATCH_SIZE + bool(trainX_unlabeled.shape[0]%BATCH_SIZE)
    trainY_pselabeled = torch.zeros((0,2))
    trainY_pselabeled = trainY_pselabeled
    with torch.no_grad(): #没有这行的话，显存会越占越大
        for i in range(batch_num): 
            if i < batch_num - 1:
                trainX_unlabeled_batch = trainX_unlabeled[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
            else:
                trainX_unlabeled_batch = trainX_unlabeled[i*BATCH_SIZE:]
            trainX_unlabeled_batch = trainX_unlabeled_batch.to(DEVICE)
            trainY_pselabeled_batch = model(trainX_unlabeled_batch)
            trainY_pselabeled_batch = trainY_pselabeled_batch.to('cpu')
            trainY_pselabeled = torch.concat((trainY_pselabeled, trainY_pselabeled_batch), dim=0)
logging.info('######################################################################')
logging.info('标签数据和伪标签无数据一块训练出新模型')

pse_dataset = MyDataset(np.array(trainX_unlabeled),np.array(trainY_pselabeled),split_ratio=0)
concat_dataset = ConcatDataset((pse_dataset,train_dataset))
concat_loader = DataLoader(dataset=concat_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 
for epoch in range(TOTAL_EPOCHS):
        model.train()       
        optimizer.param_groups[0]['lr'] = LEARNING_RATE /np.sqrt(np.sqrt(epoch+1))
        
        # Learning rate decay
        if (epoch + 1) % change_learning_rate_epochs == 0:
            optimizer.param_groups[0]['lr'] /= 2 
            logging.info('lr:%.4e' % optimizer.param_groups[0]['lr'])
           
        #Training in this epoch  
        loss_avg = 0
        for i, (x, y) in enumerate(concat_loader):
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