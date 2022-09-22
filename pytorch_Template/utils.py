import torch
import random
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from email.policy import default
import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
import os
from datetime import datetime
import copy
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import argparse
from shutil import copyfile
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


def score(y_predict:torch.tensor, testY:torch.tensor):
    Num = len(testY)
    Diff   = torch.sqrt(torch.sum((torch.square(y_predict - testY)),1)) # Euclidean distance
    Order = torch.sort(Diff).values
    Scores =  100 - Order[int(Num*0.9)]
    # print('The score of Case 3 is '  + np.str(100 - Scores))
    return Scores

def predict(args, testX, *models):
    y_test_s = []
    for model in models:
        model = model.to(torch.device(f"cuda:{args.cuda}"))
        model.eval()
        with torch.no_grad():
            y_predict = model(testX).cpu().data
        # y_test_avg = y_test_avg + y_predict
        y_test_s.append(y_predict)
        # score(y_predict)
    return y_test_s

def test(args,testX:torch.tensor, testY:torch.tensor, weights:torch.tensor=None, *models):
    if weights == None:
        weights = torch.ones(len(models), dtype=float)
    testY = testY
    y_test_avg = 0
    y_test_s = predict(args, testX, *models)
    for i in range(len(models)):
        y_test_avg = y_test_avg + weights[i]*y_test_s[i]
    y_test_avg = y_test_avg/sum(weights)
    return score(y_test_avg, testY)


def train(args, model, test_avg_min, TOTAL_EPOCHS, train_loader, model_save, test_loader=None, save=True, testX=None, testY=None, score_max=0):
    criterion = nn.MSELoss().to(torch.device(f"cuda:{args.cuda}"))
    if args.classifier == True:
        criterion_classifier = nn.CrossEntropyLoss().to(torch.device(f"cuda:{args.cuda}"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.rlrp == True:
        scheduler = ReduceLROnPlateau(optimizer, factor=0.8, patience=30,)

    for epoch in range(TOTAL_EPOCHS):
        epoch_begin_time = datetime.now()
        model.train()     
        if args.rlrp == False:
        
            optimizer.param_groups[0]['lr'] = args.lr /np.sqrt(np.sqrt(epoch+1))
            # Learning rate decay
            if (epoch + 1) % args.change_learning_rate_epochs == 0:
                optimizer.param_groups[0]['lr'] /= 2 

        logging.info('lr:%.4e' % optimizer.param_groups[0]['lr'])
        
        #Training in this epoch  
        loss_avg = 0
        for i, (x, y) in enumerate(train_loader):
            x = x.float().to(torch.device(f"cuda:{args.cuda}"))
            y = y.float().to(torch.device(f"cuda:{args.cuda}"))
            
            # 清零
            optimizer.zero_grad()

            if args.classifier == True:
                output_r, output_c = model(x)
                loss_r = criterion(output_r, y[:,:2])
                loss_c = criterion_classifier(output_c, y[:,2].long())
                loss =  loss_r + loss_c
                if i % (int(1/4*len(train_loader))) == 0:
                    logging.info(f"iter {i}/{len(train_loader)} : regression train loss {loss_r:.4f}, classifier train loss {loss_c:.4f}")
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
        with torch.no_grad():
            if testX is None and test_loader is None:
                if save:
                    if (epoch + 1) % 200 == 0:
                        logging.info('Model saved!')
                        # model.to("cuda:0")
                        model.to("cpu")
                        torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(model_save)), f'modelSubmit_2_{epoch+1}epochs.pth'))
                        model.to(torch.device(f"cuda:{args.cuda}"))
                logging.info('Epoch : %d/%d, Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg))
            elif testX is None and test_loader is not None:
                test_avg = 0
                for i, (x, y) in enumerate(test_loader):
                    x = x.float().to(torch.device(f"cuda:{args.cuda}"))
                    y = y.float().to(torch.device(f"cuda:{args.cuda}"))

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
                    
                    test_avg_min = test_avg
                    if save:
                        logging.info('Model saved!')
                        # model.to("cuda:0")
                        model.to("cpu")
                        torch.save(model.state_dict(), model_save)
                        model.to(torch.device(f"cuda:{args.cuda}"))
                logging.info('Epoch : %d/%d, Loss: %.4f, Test: %.4f, BestTest: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,test_avg,test_avg_min))
            elif testX is not None:
                testX = testX.to(torch.device(f"cuda:{args.cuda}"))
                score = test(args,testX,testY,None, model)

                """"记录一下测试loss，与score的变化进行比较"""
                test_avg = 0
                for i, (x, y) in enumerate(test_loader):
                    x = x.float().to(torch.device(f"cuda:{args.cuda}"))
                    y = y.float().to(torch.device(f"cuda:{args.cuda}"))

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
                    scheduler.step(-score) # 加负号是因为，之前是loss希望下降，现在是score希望升高
                if score > score_max:
                    score_max = score
                    if save:
                        logging.info('Model saved!')
                        # model.to("cuda:0")
                        model.to("cpu")
                        torch.save(model.state_dict(), model_save)
                        model.to(torch.device(f"cuda:{args.cuda}"))
                logging.info('Epoch : %d/%d, Loss: %.4f, TestScore: %.4f, BestTestScore: %.4f, test_Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,score,score_max, test_avg))
        epoch_stop_time = datetime.now()
        logging.info(f"每个epoch耗时{epoch_stop_time-epoch_begin_time}")
    logging.info(datetime.now())
    return test_avg_min



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


def pre2():
    """命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit_id', type=str, required=True)
    parser.add_argument('--cuda', default=0)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--big_bs', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--rlrp', default=False, action='store_true' )
    parser.add_argument('--sr', default=0.1, type=float, help='split_ratio' )
    parser.add_argument('--seed', default=1, type=int )
    parser.add_argument('--classifier', default=False, action='store_true' )
    parser.add_argument('--epochs', default=2000, type=int)
    parser.add_argument('--begin_epochs', default=10000, type=int)
    parser.add_argument('--no_seed', default=True, action = 'store_false' )
    parser.add_argument('--change_learning_rate_epochs', default=100, type=int)
    parser.add_argument('--k', default=0, type=int, help="in case3, the k_th labelled data is test set ")
    parser.add_argument('--method_id', default=1, type=int, help="the method id  ")

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin_memory', default=False, action='store_true' )
    parser.add_argument('--no_test', default=False, action = 'store_true' )
    args = parser.parse_args()
    TOTAL_EPOCHS = args.epochs
    """注意评测设备只有一块gpu"""
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
    copyfile('pytorch_Template/utils.py', os.path.join(id_path, 'utils.py'))
    
    """设置随机数种子"""
    if args.no_seed == False:
        seed_value = args.seed
        seed_everything(seed_value=seed_value)
        logging.info(f'seed_value:{seed_value}')
    else:
        logging.info(f'不设定可复现')
    """加载数据"""
    # file_name1 = 'data/Case_3_Training.npy'
    # logging.info('The current dataset is : %s'%(file_name1))
    # CIR = np.load(file_name1)
    # trainX = CIR.transpose((2,1,3,0))  #[none, 256, 72, 2]
    # trainX_labeled = trainX[:1000,:,:,:]
    # trainX_unlabeled = trainX[1000:,:,:,:]
    # file_name2 = 'data/Case_3_Training_Label.npy'
    # logging.info('The current dataset is : %s'%(file_name2))
    # POS = np.load(file_name2)
    # trainY_labeled = POS.transpose((1,0)) #[none, 2]

    
    

    # if args.classifier == True:
    #     trainY_class_labeled = get_class_label(trainY_labeled)
    #     trainY_labeled = np.concatenate((trainY_labeled, trainY_class_labeled), axis=1)
    # """转化为tensor"""
    # trainX_labeled = torch.tensor(trainX_labeled)
    # trainY_labeled = torch.tensor(trainY_labeled)
    # trainX_unlabeled = torch.tensor(trainX_unlabeled)


    # """打乱数据顺序"""
    # index_L = np.arange(len(trainX_labeled))
    # np.random.shuffle(index_L)
    # trainX_labeled = trainX_labeled[index_L]
    # trainY_labeled = trainY_labeled[index_L]
    # index_U = np.arange(len(trainX_unlabeled))
    # np.random.shuffle(index_U)
    # trainX_unlabeled = trainX_unlabeled[index_U]
    


    
    # """分出测试集"""
    # test_trainX_labeled = trainX_labeled[0:int(split_ratio*len(trainX_labeled))]
    # test_trainY_labeled = trainY_labeled[0:int(split_ratio*len(trainY_labeled))]

    # """分出训练集"""
    # trainX_labeled = trainX_labeled[int(split_ratio*len(trainX_labeled)):]
    # trainY_labeled = trainY_labeled[int(split_ratio*len(trainY_labeled)):]
    trainX_labeled = torch.tensor(np.load('data/case3/Case_3_Training_train.npy'), dtype=torch.float)
    trainY_labeled = torch.tensor(np.load('data/case3/Case_3_Training_train_label.npy'), dtype=torch.float)
    test_trainX_labeled = torch.tensor(np.load('data/case3/Case_3_Training_test.npy'), dtype=torch.float)
    test_trainY_labeled = torch.tensor(np.load('data/case3/Case_3_Training_test_label.npy'), dtype=torch.float)
    trainX_unlabeled = torch.tensor(np.load('data/case3/Case_3_unlabeled.npy'), dtype = torch.float)
    if args.classifier == False:
        trainY_labeled = trainY_labeled[:,:2]
        test_trainY_labeled = test_trainY_labeled[:,:2]
    """训练集数据扩增"""
    # trainX_labeled, trainY_labeled = Model_2().data_aug(x = trainX_labeled, y = trainY_labeled)
    """测试集数据扩增"""
    # test_trainX_labeled, test_trainY_labeled = Model_2().data_aug(x = test_trainX_labeled, y = test_trainY_labeled)

    return args, id_path, submit_path,  trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled, model_save

def bootstrapping(id_path):

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
    """自助法"""
    data_len =1000
    train_index = np.random.choice(data_len, size=(data_len), replace=True )
    test_index = set(range(data_len)).difference(set(train_index))
    test_trainX_labeled = torch.tensor(trainX_labeled[list(test_index)], dtype=torch.float)
    test_trainY_labeled = torch.tensor(trainY_labeled[list(test_index)], dtype=torch.float)
    trainX_labeled = torch.tensor(trainX_labeled[train_index], dtype=torch.float)
    trainY_labeled = torch.tensor(trainY_labeled[train_index], dtype=torch.float)
    logging.info(f"test_index:{test_index}")
    np.save(os.path.join(id_path, 'test_index.npy'), np.array(test_index))
    np.save(os.path.join(id_path, 'test_trainX_labeled.npy'), test_trainX_labeled)
    np.save(os.path.join(id_path, 'test_trainY_labeled.npy'), test_trainY_labeled)
    return trainX_unlabeled, trainX_labeled, trainY_labeled, test_trainX_labeled, test_trainY_labeled

def get_1000():
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
    trainX_labeled = torch.tensor(trainX_labeled, dtype=torch.float)
    trainY_labeled = torch.tensor(trainY_labeled, dtype=torch.float)
    trainX_unlabeled = torch.tensor(trainX_unlabeled, dtype=torch.float)
    return trainX_unlabeled, trainX_labeled, trainY_labeled
def get_900(k=0):
    """
    k: 10份数据，第i份被作为验证集
    """
    """加载数据"""
    assert(k>=0 and k<=9 and type(k) is int)
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
    trainX_labeled = torch.tensor(trainX_labeled, dtype=torch.float)
    trainY_labeled = torch.tensor(trainY_labeled, dtype=torch.float)
    trainX_unlabeled = torch.tensor(trainX_unlabeled, dtype=torch.float)

    shuffle_1000 = np.load('./data/shuffle_1000.npy')
    test_index = shuffle_1000[k*100:((k+1)*100)]
    test_index.sort()
    train_index = np.array(list(set(shuffle_1000).difference(set(test_index))))
    
    testX_labeled = trainX_labeled[test_index]
    testY_labeled = trainY_labeled[test_index]
    trainX_labeled  = trainX_labeled[train_index]
    trainY_labeled = trainY_labeled[train_index]
    assert( len(trainX_labeled)==900 and  len(trainY_labeled)==900 and len(testX_labeled)==100 and len(testY_labeled)==100)
    return trainX_unlabeled, trainX_labeled, trainY_labeled, testX_labeled, testY_labeled

def label(args, X, BATCH_SIZE=1000, if_weight=False, weight_thres = 98.0, if_adaptive_weight_thres=True, *models):
    def get_weight(model, weight_thres):
        _, _, _, testX, testY = get_900(k=0)
        # Y = label(args, testX, 1000, False, weight_thres, model)
        testX = testX.to(torch.device(f"cuda:{args.cuda}"))
        score = test(args,testX,testY,None, model)
        logging.info(f"weak model score: {score}")
        if score-weight_thres<=0:
            return 0.0
        else:
            return score-weight_thres
    X = torch.tensor(X, dtype=torch.float)
    # X_pselabeled_ave = 0
    Y_pselabeled_ave = 0
    weights = torch.ones(len(models), dtype=torch.float)
    
    """获取最佳的权重阈值"""
    weight_thres_best = 98.0
    test_ave_score_best = 0.0
    if if_adaptive_weight_thres == True and len(models)>1: 
        _, _, _, testX, testY = get_900(k=0)
        testX = testX.to(torch.device(f"cuda:{args.cuda}"))
        y_test_s = predict(args, testX, *models)
        score_s = [score(y_test_s[i], testY) for i in range(len(models))]
        logging.info(f"所有{len(models)}个子模型，各自的验证分数分别为:\n{score_s}")
        for weight_thres_pre in np.arange(98.0,98.8,0.01): 
            for weight_add in np.arange(0,2.0,0.01):
                logging.info(f"########################################################################")
                logging.info(f"权重阈值为:  {weight_thres_pre}  时：")
                logging.info(f"权重附加值为:    {weight_add}    时：")
                weights_pre = (score_s > weight_thres_pre)*(score_s - weight_thres_pre) + weight_add
                weights_pre = [round(weight_each, 4) for weight_each in weights_pre]
                logging.info(f"权重为:  {weights_pre}")
                y_test_avg = 0
                for i in range(len(models)):
                    y_test_avg = y_test_avg + weights_pre[i]*y_test_s[i]
                y_test_avg = y_test_avg/sum(weights_pre)
                test_ave_score = score(y_test_avg, testY)
                logging.info(f"模型平均的测试分数为:    {test_ave_score}")
                if test_ave_score > test_ave_score_best:
                    test_ave_score_best = test_ave_score
                    weight_thres_best = weight_thres_pre
                    weights = weights_pre
                    logging.info(f"***** 获得最佳权重 ******")
                    logging.info(f"最佳模型平均的测试分数为:    {test_ave_score_best}")
        logging.info(f"######################### ******************************* #####################################")
        logging.info(f"######################### 最终 #####################################")
        logging.info(f"最佳权重阈值为:  {weight_thres_best}")
        logging.info(f"最佳权重为:  {weights}")
        logging.info(f"最佳模型平均的测试分数为:    {test_ave_score_best}")

        
    Y_pselabeled_s = []
    for model_index, model in enumerate(models):
        model = model.to(torch.device(f"cuda:{args.cuda}"))
        model.eval()
        batch_num = X.shape[0]//BATCH_SIZE + bool(X.shape[0]%BATCH_SIZE)
        # X_pselabeled = torch.zeros((0, X.shape[1], X.shape[2], X.shape[3]))
        Y_pselabeled = torch.zeros((0,2))
        with torch.no_grad(): #没有这行的话，显存会越占越大
            for i in range(batch_num): 
                if i < batch_num - 1:
                    X_batch = X[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
                else:
                    X_batch = X[i*BATCH_SIZE:]
                X_batch = X_batch.to(torch.device(f"cuda:{args.cuda}")).float()
                Y_pselabeled_batch = model(X_batch)
                Y_pselabeled_batch = Y_pselabeled_batch.to('cpu')      
                Y_pselabeled = torch.concat((Y_pselabeled, Y_pselabeled_batch), dim=0)
                # X_pselabeled = torch.concat((X_pselabeled, X_batch.to('cpu')), dim=0)
        # X_pselabeled_ave = X_pselabeled_ave + X_pselabeled
            Y_pselabeled_s.append(Y_pselabeled)
            if if_weight == True and if_adaptive_weight_thres == False:
                weights[model_index] = get_weight(model, weight_thres)
    
    logging.info(f"weights{weights}")
    if if_adaptive_weight_thres == False:
        """测一下集成模型在验证集上的score"""
        _, _, _, testX, testY = get_900(k=0)
        testX = testX.to(torch.device(f"cuda:{args.cuda}"))
        logging.info(f"集成模型在验证集上的score:   {test(args,testX,testY,weights,*models)}")

    for i in range(len(models)):
        Y_pselabeled_ave = Y_pselabeled_ave + weights[i] * Y_pselabeled_s[i]
    Y_pselabeled_ave = Y_pselabeled_ave / sum(weights)

    """调整出界标签"""
    Y_pselabeled_ave[:,0][Y_pselabeled_ave[:,0]>120.0] = 120.0
    Y_pselabeled_ave[:,0][Y_pselabeled_ave[:,0]<0] = 0.0
    Y_pselabeled_ave[:,1][Y_pselabeled_ave[:,1]>60.0] = 60.0
    Y_pselabeled_ave[:,1][Y_pselabeled_ave[:,1]<0] = 0.0
    return  Y_pselabeled_ave