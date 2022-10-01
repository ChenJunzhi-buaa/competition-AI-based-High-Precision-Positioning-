
import numpy as np
import torch
import logging
from utils.get_data import get_900

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

"""好多个子模型一块给无标签数据打标签"""
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
        for weight_thres_pre in np.arange(98.0,99.1,0.01): 
            for weight_add in np.arange(0,4.0,0.01):
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

    logging.info(f"######################### 开始打标签 #####################################")
    Y_pselabeled_s = []
    for model_index, model in enumerate(models):
        logging.info(f"######################### 第{model_index + 1}个模型打标签 #####################################")
        if weights[model_index] == 0:
            Y_pselabeled_s.append(torch.zeros((X.shape[0],2)))
            continue
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
    logging.info(f"######################### 打标签结束 #####################################")
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

"""下面两个函数是想获得模型集成的最佳模型集成权重"""
def get_thres_by_large(args, X, *models):
    X = torch.tensor(X, dtype=torch.float)
    weights = torch.ones(len(models), dtype=torch.float)
    
    """获取最佳的权重阈值"""
    weight_thres_best = 98.0
    test_ave_score_best = 0.0
    
    _, _, _, testX, testY = get_900(k=0)
    testX = testX.to(torch.device(f"cuda:{args.cuda}"))
    y_test_s = predict(args, testX, *models)
    score_s = [score(y_test_s[i], testY) for i in range(len(models))]
    logging.info(f"所有{len(models)}个子模型，各自的验证分数分别为:\n{score_s}")
    for weight_thres_pre in np.arange(98.0,99.1,0.01): 
        for weight_add in np.arange(0,4.0,0.01):
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
    return  weights, test_ave_score_best

def get_thres(args, X, BATCH_SIZE=1000, if_weight=False, weight_thres = 98.0, if_adaptive_weight_thres=True, *models):
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
        # weights_by_lagre = torch.tensor(score_s) -98
        # weights_by_lagre = weights_by_lagre/weights_by_lagre.sum()
        
        weights_by_lagre, test_ave_score_best = get_thres_by_large(args, X, *models)
        """在利用get_thres_by_large()获得模型集成权重之后,外加随机数希望获得更好的模型权重"""
        weights_by_lagre = torch.tensor(weights_by_lagre)
        weights_by_lagre = weights_by_lagre/weights_by_lagre.sum()
        logging.info(f"根据得分获得的初始权重为:  {weights_by_lagre}")
        while(1):
            weights_pre = torch.rand(len(models), )-0.5
            # weights_pre = weights_pre/weights_pre.sum()
            weights_pre = weights_pre + weights_by_lagre
            weights_pre = weights_pre*(weights_pre>0)
            y_test_avg = 0
            for i in range(len(models)):
                y_test_avg = y_test_avg + weights_pre[i]*y_test_s[i]
            y_test_avg = y_test_avg/sum(weights_pre)
            test_ave_score = score(y_test_avg, testY)
            # logging.info(f"模型平均的测试分数为:    {test_ave_score}")
            if test_ave_score > test_ave_score_best:
                test_ave_score_best = test_ave_score
                # weight_thres_best = weight_thres_pre
                weights = weights_pre
                logging.info(f"***** 获得最佳权重 ******")
                logging.info(f"权重为:  {weights_pre}")
                logging.info(f"最佳模型平均的测试分数为:    {test_ave_score_best}")
        logging.info(f"######################### ******************************* #####################################")
        logging.info(f"######################### 最终 #####################################")
        # logging.info(f"最佳权重阈值为:  {weight_thres_best}")
        logging.info(f"最佳权重为:  {weights}")
        logging.info(f"最佳模型平均的测试分数为:    {test_ave_score_best}")

    