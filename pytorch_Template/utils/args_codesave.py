import numpy as np
import torch
import logging
import os
from datetime import datetime
import argparse
from shutil import copyfile, copytree

from utils.seed import seed_everything
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
    parser.add_argument('--smaller_test_split', default=None, type=float, help='split test set to be more small' )
    args = parser.parse_args()
    """注意评测设备只有一块gpu"""
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

    # copyfile('pytorch_Template/utils.py', os.path.join(id_path, 'utils.py'))
    copytree('pytorch_Template/utils', os.path.join(id_path, 'utils'), dirs_exist_ok=True)
    
    """设置随机数种子"""
    if args.no_seed == False:
        seed_value = args.seed
        seed_everything(seed_value=seed_value)
        logging.info(f'seed_value:{seed_value}')
    else:
        logging.info(f'不设定可复现')
    """加载数据"""

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
