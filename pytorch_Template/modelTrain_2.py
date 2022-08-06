import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from modelDesign_2 import Model_2

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
TOTAL_EPOCHS = 20
split_ratio = 0.1
change_learning_rate_epochs = 100

model_save = 'modelSubmit_2.pth'

DEVICE=torch.device("cpu")
if torch.cuda.is_available():
        DEVICE=torch.device("cuda:0")


if __name__ == '__main__':

    file_name1 = 'data/Case_3_Training.npy'
    print('The current dataset is : %s'%(file_name1))
    CIR = np.load(file_name1)
    trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    
    file_name2 = 'data/Case_3_Training_Label.npy'
    print('The current dataset is : %s'%(file_name2))
    POS = np.load(file_name2)
    trainY = POS.transpose((1,0)) #[none, 2]

    model = Model_2()
    model = model.to(DEVICE)
    print(model)
    
    train_dataset = MyDataset(trainX,trainY,split_ratio)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)  # shuffle 标识要打乱顺序
    test_dataset = MyTestset(trainX,trainY,split_ratio)
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
            print('lr:%.4e' % optimizer.param_groups[0]['lr'])
           
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
        print('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
        if test_avg < test_avg_min:
            print('Model saved!')
            test_avg_min = test_avg

            # torch.save(model, model_save)
            torch.save(model.state_dict(), model_save)
    #torch.save(model model_save)
    #torch.save(model.state_dict(), model_save)

# else:
#     print("load torch model")
#     model_ckpt = torch.load(model, model_save)
