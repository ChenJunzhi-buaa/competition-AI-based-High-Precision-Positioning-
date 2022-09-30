from torch.utils.data import Dataset



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