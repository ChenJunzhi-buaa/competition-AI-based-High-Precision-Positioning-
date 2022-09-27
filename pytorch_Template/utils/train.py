import torch
import numpy as np
import os
import torch.nn as nn
import logging
from datetime import datetime
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
from utils.label_predict_score import test

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
            x = x.to(torch.device(f"cuda:{args.cuda}"))
            y = y.to(torch.device(f"cuda:{args.cuda}"))
            
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
                if args.rlrp == True:
                    scheduler.step(loss_avg)
            elif testX is None and test_loader is not None:
                test_avg = 0
                for i, (x, y) in enumerate(test_loader):
                    x = x.to(torch.device(f"cuda:{args.cuda}"))
                    y = y.to(torch.device(f"cuda:{args.cuda}"))

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
                    x = x.to(torch.device(f"cuda:{args.cuda}"))
                    y = y.to(torch.device(f"cuda:{args.cuda}"))

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
                
                if test_avg < test_avg_min:
                    test_avg_min = test_avg
                    if save:
                        logging.info('min_testloss Model saved!')
                        # model.to("cuda:0")
                        model.to("cpu")
                        torch.save(model.state_dict(), os.path.join(os.path.dirname(os.path.dirname(model_save)), f'modelSubmit_2_min_testloss.pth'))
                        model.to(torch.device(f"cuda:{args.cuda}"))
                logging.info('Epoch : %d/%d, Loss: %.4f, TestScore: %.4f, BestTestScore: %.4f, test_Loss: %.4f, test_Loss_min: %.4f' % (epoch + 1, TOTAL_EPOCHS, loss_avg,score,score_max, test_avg, test_avg_min))
        epoch_stop_time = datetime.now()
        logging.info(f"每个epoch耗时{epoch_stop_time-epoch_begin_time}")
    logging.info(datetime.now())
    return test_avg_min