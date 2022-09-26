#--coding: utf-8--

# TODO
# seed随机数种子的选择可能挺有用
# 虚拟标签的应该在那个120，60的范围内
# 转化为分类问题
# 数据归一化
# BN层
########以下三行是解决p.map(f, [args])卡住问题的########
#########参考https://divertingpan.github.io/post/pytorch_bizarre_error_debug/###############
import os
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 
import torch
from torch.utils.data import DataLoader
from modelDesign_2 import Model_2
import logging
import os
from shutil import copyfile
from utils import *
from datetime import datetime
from multiprocessing import Pool

if __name__ == '__main__':

    """带测试集"""
    args, id_path, submit_path,  _, _, _, _, _, model_save = pre2()
    trainX_unlabeled, trainX_labeled, trainY_labeled, testX_labeled, testY_labeled = get_900(k=args.k)
    copyfile(__file__, os.path.join(id_path, __file__.split('/')[-1]))
    def f(args):
        models = []
        """自己的弱模型"""
        for i in range(1,24):
            if i != 13 and  i !=14:
                model_address = f'./submit/58-{i}/submit_pt/modelSubmit_2.pth'
                model = Model_2(method_id=i)
                model.load_state_dict(torch.load(model_address))
                # model_1 =model.to(DEVICE)
                models.append(model)
        """xm"""
        for i in [101,102,103]:
            model_address = f'./submit/{i}/submit_pt/modelSubmit_2.pth'
            model = Model_2(method_id=i)
            model.load_state_dict(torch.load(model_address))
            # model_1 =model.to(DEVICE)
            models.append(model)
        """自己的集成模型"""
        if args.no_esembled == False:
            for i in range(0,30):
                folder_name = f'59-{i}'
                model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')
                while os.path.exists(model_address): 
                    logging.info(f'集成已有的集成模型的路径: {model_address}')
                    model = Model_2(method_id=i)
                    model.load_state_dict(torch.load(model_address))
                    # model_1 =model.to(DEVICE)
                    models.append(model)

                    folder_name = folder_name + f'-{i}'
                    model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')
        """自己的二代集成模型（只用一半半监督数据）"""
        if args.no_esembled_half == False:
            for i in range(0,30):
                folder_name = f'62-{i}'
                model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')
                while os.path.exists(model_address): 
                    logging.info(f'集成已有的集成模型的路径: {model_address}')
                    model = Model_2(method_id=i)
                    model.load_state_dict(torch.load(model_address))
                    # model_1 =model.to(DEVICE)
                    models.append(model)

                    folder_name = folder_name + f'-{i}'
                    model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')

        """自己的三代集成模型（只用3000个半监督数据）"""
        if args.no_esembled_3000 == False:
            for i in range(0,30):
                folder_name = f'63-{i}'
                model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')
                while os.path.exists(model_address): 
                    logging.info(f'集成已有的集成模型的路径: {model_address}')
                    model = Model_2(method_id=i)
                    model.load_state_dict(torch.load(model_address))
                    # model_1 =model.to(DEVICE)
                    models.append(model)

                    folder_name = folder_name + f'-{i}'
                    model_address = os.path.join('./submit/', folder_name, 'submit_pt/modelSubmit_2.pth')
        Y_pselabeled_ave = label(args, trainX_unlabeled, 1000, True, 98.05, True, *models)



        torch.save(Y_pselabeled_ave, os.path.join(id_path, 'Y_pselabeled_ave.pth'))

        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    """完全释放显存的写法""" 
    # 参考https://www.cnblogs.com/dechinphy/p/gc.html
    # f(args)
    with Pool(1) as p: 
        red = p.map(f, [args])
    Y_pselabeled_ave = torch.load(os.path.join(id_path, 'Y_pselabeled_ave.pth'))
    if args.no_test == True:
        if args.copy_test == True:
            confusion_X = torch.cat((trainX_labeled, trainX_unlabeled, testX_labeled.repeat(16,*([1]*(len(testX_labeled.shape)-1)))), axis=0)
            confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave, testY_labeled.repeat(16,*([1]*(len(testY_labeled.shape)-1)))), axis=0)
        else:
            confusion_X = torch.cat((trainX_labeled, trainX_unlabeled, testX_labeled), axis=0)
            confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave, testY_labeled), axis=0)
        
    elif args.smaller_test_split is not None:
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled, testX_labeled[int(args.smaller_test_split * len(testX_labeled)):]), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave, testY_labeled[int(args.smaller_test_split * len(testX_labeled)):]), axis=0)
        test_dataset = MyDataset(testX_labeled[:int(args.smaller_test_split * len(testX_labeled))],testY_labeled[:int(args.smaller_test_split * len(testX_labeled))],split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序
        logging.info(f"测试集数量：{len(test_dataset)}")
    elif args.half_pseudo == True:
        pseudo_len = len(trainX_unlabeled)
        pseudo_half_index = torch.randperm(pseudo_len)[:int(pseudo_len/2)]
        trainX_unlabeled_half = trainX_unlabeled[pseudo_half_index]
        Y_pselabeled_ave_half = Y_pselabeled_ave[pseudo_half_index]
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled_half), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave_half), axis=0)
        test_dataset = MyDataset(testX_labeled,testY_labeled,split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序                            

        logging.info(f"测试集数量：{len(test_dataset)}")
    elif args._3000_pseudo == True:
        pseudo_len = len(trainX_unlabeled)
        pseudo_half_index = torch.randperm(pseudo_len)[:3000]
        trainX_unlabeled_half = trainX_unlabeled[pseudo_half_index]
        Y_pselabeled_ave_half = Y_pselabeled_ave[pseudo_half_index]
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled_half), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave_half), axis=0)
        test_dataset = MyDataset(testX_labeled,testY_labeled,split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序                            

        logging.info(f"测试集数量：{len(test_dataset)}")
        args.big_bs = int(args.big_bs/4)
        logging.info(f"新bs：{args.big_bs}")
    else:
        confusion_X = torch.cat((trainX_labeled, trainX_unlabeled), axis=0)
        confusion_Y = torch.cat((trainY_labeled, Y_pselabeled_ave), axis=0)
        test_dataset = MyDataset(testX_labeled,testY_labeled,split_ratio=0)
        test_loader = DataLoader(dataset=test_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序                            

        logging.info(f"测试集数量：{len(test_dataset)}")
    train_dataset = MyDataset(confusion_X,confusion_Y,split_ratio=0)
    train_loader = DataLoader(dataset=train_dataset,
                                               batch_size=args.big_bs,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               pin_memory=args.pin_memory
                                               )  # shuffle 标识要打乱顺序
        
    
    model = Model_2(no_grad=False, if_classifier=args.classifier,method_id=args.method_id)
    model = model.to(torch.device(f"cuda:{args.cuda}"))
    logging.info(model)
    if args.no_test == True:
        train(args, model, 10000, args.begin_epochs, train_loader, model_save,save=True)
    else:
        train(args, model, 10000, args.begin_epochs, train_loader, model_save,test_loader=test_loader,save=True, testX = testX_labeled, testY= testY_labeled)
    logging.info(datetime.now())
    