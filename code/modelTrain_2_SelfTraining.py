# TODO
# psedo label应该不断更新
# 网络变小

from pickle import NONE
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from modelDesign_2 import Model_2
import logging
from shutil import copyfile
import argparse
import os
from utils import seed_everything,MyDataset

class SSL():

    def __init__(self, labelset_X , labelset_Y, unlabelset_X, model_save=None, device="cuda:0") :
        """
        labelset_X: shape (num,*,*,*,,)
        """
        self.labelset_X = torch.tensor(labelset_X, dtype=torch.float32)
        self.labelset_Y = torch.tensor(labelset_Y, dtype=torch.float32)

        self.unlabelset_X = torch.tensor(unlabelset_X, dtype=torch.float32)
        self.model0 = Model_2()
        # """加载比较强的模型"""
        self.model0.load_state_dict(torch.load("submit/22/submit_pt/modelSubmit_2.pth"))
        self.model1 = None
        self.model2 = None
        self.unlabelset_X1 = None
        self.unlabelset_X2 = None
        self.unlabelset_X3 = None
        self.unlabelset_left = None
        self.device = device
        self.model_last = None
        self.concat_X = None
        self.concat_Y = None
        self.test_avg_min = 10000
        self.model_save = model_save

        """分出测试集"""
        test_labelset_X = self.labelset_X[-int(len(self.labelset_X)/10.0):]
        test_labelset_Y = self.labelset_Y[-int(len(self.labelset_X)/10.0):]
        test_set = MyDataset(test_labelset_X,test_labelset_Y,split_ratio=0)
        self.test_loader = DataLoader(dataset=test_set,
                                                batch_size=100,
                                                shuffle=True) 
        self.labelset_X = self.labelset_X[:-int(len(self.labelset_X)/10.0)]
        self.labelset_Y = self.labelset_Y[:-int(len(self.labelset_X)/10.0)]
    def main(self,thres_len_un=2000):
        T = 20
        EPOCHS = 500
        for t in range(T):
            # logging.info(f"unlabelset_X.shape[0]:{self.unlabelset_X.shape[0]}")
            logging.info(f"第{t}次循环")
            logging.info("1000个数据训练 model0   ##################################")
            self.model0 = self.train(self.model0, self.labelset_X, self.labelset_Y,TOTAL_EPOCHS=EPOCHS,test_loader=self.test_loader)
            
            self.unlabelset_Y = self.label(self.unlabelset_X, self.model0)
            self.concat_X = torch.concat((self.unlabelset_X, self.labelset_X), dim=0)
            self.concat_Y = torch.concat((self.unlabelset_Y, self.labelset_Y), dim=0)
            logging.info("混合数据训练 model0   ##################################")
            self.model0 = self.train(self.model0, self.concat_X,self.concat_Y, TOTAL_EPOCHS=EPOCHS,test_loader=self.test_loader)
            # self.get_new_set()
            # t = i + 1
        # self.model_last = self.model0


    def train(self, model, train_X, train_Y, BATCH_SIZE=100, LEARNING_RATE=0.001, TOTAL_EPOCHS=50, change_learning_rate_epochs=100, test_loader=None):
        model = model.to(self.device)
        train_dataset = MyDataset(train_X,train_Y,split_ratio=0)

        train_loader = DataLoader(dataset=train_dataset,
                                                batch_size=BATCH_SIZE,
                                                shuffle=True)  # shuffle 标识要打乱顺序
        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-4)
        if test_loader == None:
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
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    
                    # 清零
                    optimizer.zero_grad()
                    output = model(x)
                    # 计算损失函数
                    loss = criterion(output, y)
                    loss.backward()
                    optimizer.step()
                    
                    loss_avg += loss.item() 
                    
                loss_avg /= len(train_loader)
                logging.info('Epoch : %d/%d, Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg))
        else:
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
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    
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
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)

                    output = model(x)
                    # 计算损失函数
                    loss_test = criterion(output, y)
                    test_avg += loss_test.item() 
                
                test_avg /= len(test_loader)
                
                if test_avg < self.test_avg_min:
                    logging.info('Model saved!')
                    self.test_avg_min = test_avg
                    model.to("cuda:0")
                    torch.save(model.state_dict(), model_save)
                    model.to(self.device)
                logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,self.test_avg_min))
        return model

    def split_unlabelset(self, ):
        """TODO:无标签数据集的长度设定,或许要改进"""
        if 3*len(self.labelset_X) < len(self.unlabelset_X):
            split_len = len(self.labelset_X)
        else:
            split_len = len(self.unlabelset_X) // 3

        """打乱一下"""
        index = np.arange(len(self.unlabelset_X))
        np.random.shuffle(index)
        self.unlabelset_X = self.unlabelset_X[index]

        self.unlabelset_X1 = self.unlabelset_X[:split_len]
        self.unlabelset_X2 = self.unlabelset_X[split_len:split_len*2]
        self.unlabelset_X3 = self.unlabelset_X[split_len*2:split_len*3]
        self.unlabelset_left = self.unlabelset_X[split_len*3:] # 剩下的
    def get_new_set(self,thres=1):
        logging.info("训练model0   ##################################")
        self.model0 = self.train(self.labelset_X, self.labelset_Y,TOTAL_EPOCHS=1)
        self.split_unlabelset()
        Y1 = self.label(self.unlabelset_X1, self.model0)
        Y2 = self.label(self.unlabelset_X2, self.model0)
        X1_concat = torch.concat((self.labelset_X, self.unlabelset_X1), dim=0)
        Y1_concat = torch.concat((self.labelset_Y, Y1), dim = 0)
        X2_concat = torch.concat((self.labelset_X, self.unlabelset_X2), dim=0)
        Y2_concat = torch.concat((self.labelset_Y, Y2), dim=0)
        logging.info("训练model1   ##################################")
        self.model1 = self.train(X1_concat,Y1_concat,TOTAL_EPOCHS=1)
        logging.info("训练model2   ###################################")
        self.model2 = self.train(X2_concat,Y2_concat,TOTAL_EPOCHS=1)
        Y3_model1 = self.label(self.unlabelset_X3, self.model1)
        Y3_model2 = self.label(self.unlabelset_X3, self.model2)
        diff = (Y3_model1 - Y3_model2).square().sum(dim=1).sqrt()
        """方式1:两个模型预测足够接近的样本才打上标签"""
        # diff_flag = diff < thres
        # Labeled_X3 = self.unlabelset_X3[diff_flag]
        # Unlabeled_X3 = self.unlabelset_X3[diff_flag.logical_not()]
        # Labeled_Y3 = ((Y3_model1 + Y3_model2)/2.0)[diff_flag]
        """方式2:两个模型预测最接近的前一半的样本打上标签"""
        Labeled_X3 = self.unlabelset_X3[diff.sort()[1][:len(diff)//2]]
        Unlabeled_X3 = self.unlabelset_X3[diff.sort()[1][len(diff)//2:]]
        Labeled_Y3 = ((Y3_model1 + Y3_model2)/2.0)[diff.sort()[1][:len(diff)//2]]

        self.labelset_X = torch.concat((self.labelset_X,Labeled_X3), dim=0)
        self.labelset_Y = torch.concat((self.labelset_Y,Labeled_Y3), dim=0)
        self.unlabelset_X = torch.concat((self.unlabelset_X1, self.unlabelset_X2, Unlabeled_X3), dim=0)
        if len(self.unlabelset_left) > 0:
            self.unlabelset_X = torch.concat((self.unlabelset_X, self.unlabelset_left), dim=0)

    def label(self, X, model, BATCH_SIZE=100):
        model.eval()
        batch_num = X.shape[0]//BATCH_SIZE + bool(X.shape[0]%BATCH_SIZE)
        Y_pselabeled = torch.zeros((0,2))
        with torch.no_grad(): #没有这行的话，显存会越占越大
            for i in range(batch_num): 
                if i < batch_num - 1:
                    X_batch = X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                else:
                    X_batch = X[i*BATCH_SIZE:]
                X_batch = X_batch.to(self.device)
                Y_pselabeled_batch = model(X_batch)
                Y_pselabeled_batch = Y_pselabeled_batch.to('cpu')
                Y_pselabeled = torch.concat((Y_pselabeled, Y_pselabeled_batch), dim=0)
        return Y_pselabeled







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    args = parser.parse_args()
    """注意评测设备只有一块gpu"""
    DEVICE=torch.device(f"cuda:{args.cuda}")

    id_path = os.path.join('submit',str(args.submit_id))
    if not os.path.exists(id_path):
        os.mkdir(id_path)
    submit_path = os.path.join(id_path, 'submit_pt')
    if not os.path.exists(submit_path):
        os.mkdir(submit_path)
    logging.basicConfig(filename=os.path.join(id_path,"model2_log.txt"), filemode='w', level=logging.DEBUG)
    model_save = os.path.join(submit_path,'modelSubmit_2.pth')
    copyfile('pytorch_Template/modelDesign_2.py', os.path.join(submit_path, 'modelDesign_2.py'))
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))

    seed_value = 1
    seed_everything(seed_value=seed_value)
    logging.info(f'seed_value:{seed_value}')

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
    


    
    ssl = SSL(labelset_X=trainX_labeled, labelset_Y=trainY_labeled, unlabelset_X=trainX_unlabeled, device=DEVICE, model_save=model_save)
    ssl.main()
    # ssl.model_last = ssl.model_last.to("cuda:0")
    # torch.save(ssl.model_last.state_dict(), model_save)

    # np.save('unlabelset_X.npy', ssl.unlabelset_X.to('cpu').numpy())
    # np.save('labelset_X.npy', ssl.labelset_X.to('cpu').numpy())
    # np.save('labelset_Y.npy', ssl.labelset_Y.to('cpu').numpy())

    # logging.info(f"ssl.unlabelset_X.shape:{ssl.unlabelset_X.shape}")
    # logging.info(f"ssl.labelset_X.shape:{ssl.labelset_X.shape}")
    # logging.info(f"ssl.labelset_Y.shape:{ssl.labelset_Y.shape}")
