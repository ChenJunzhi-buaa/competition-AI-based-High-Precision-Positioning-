# competition-AI-based-High-Precision-Positioning
竞赛记录https://datafountain.cn/competitions/575/ranking?isRedance=0&sch=1954  
代码在 4卡3090 `/data/cjz/location/`

# 涉及到的代码技巧
| 涉及到的代码技巧 | 代码 |
| ---- | ---- |
| 1. 将子模型的参数复制到集成模型。 |`model1_esemble.py`，`modelDesign_1.py` |
| 2. 将模型参数的精度从默认的32位降低到16位 |`model1_esemble.py` |
| 3. 将子模型的参数复制到集成模型。 |`model1_esemble.py` |
| 4. 让一些脚本所占用的显存在运行完之后完全释放出来 | `modelTrain_2_SemiSupervisedEsemble.py` |
| 5. 复制文件（夹） | `code/utils/args_codesave.py` |
| 6. 设置完全可复现 | `code/utils/seed.py` |
| 7. dataloader的num_workers和pin_memory两个参数 | `code/modelTrain_1.py` |
| 8. `with torch.no_grad():` ,省内存|   |
| 9. Dataset构建| `code/utils/dataset.py`  | |
| 10. 位置参数，默认参数，可变参数的混合使用 | `code/utils/label_predict_score.py`中的label()函数|


