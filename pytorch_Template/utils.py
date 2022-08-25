import torch
import random
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
class MyDataset(Dataset):
    def __init__(self, trainX,trainY,split_ratio,TrainNum=0):
        if TrainNum == 0:
            N = trainX.shape[0]
            
            TrainNum = int((N*(1-split_ratio)))
        self.x = trainX[:TrainNum]
        self.y = trainY[:TrainNum]

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        
        return (x, y)

class MyTestset(Dataset):
    def __init__(self, trainX,trainY,split_ratio,TrainNum=0):
        if TrainNum == 0:
            N = trainX.shape[0]
        
            TrainNum = int((N*(1-split_ratio)))
        self.x = trainX[TrainNum:]
        self.y = trainY[TrainNum:]

        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):     
        x = self.x[idx]
        y = self.y[idx]
        
        return (x, y)

if __name__ == '__main__':
    seed = 42
    seed_everything(seed)
