
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import logging
import os
import torch

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

__all__ = ["bootstrapping", "get_900", "get_1000"]