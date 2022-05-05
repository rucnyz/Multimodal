# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 20:43
# @Author  : nieyuzhou
# @File    : imputation.py
# @Software: PyCharm
import pickle

import numpy as np
import torch
from sklearn.metrics import mean_squared_error


def RMSE(view_num, x_pred, x, missing_index):
    loss = torch.tensor([0.0])
    all_num = 0
    for num in range(view_num):
        # 注意这里的改动会使得其余模型均无法使用
        loss += (torch.pow((x_pred[num] - x[num]), 2.0) * missing_index[:, [num]].logical_not()).sum()
        all_num += missing_index[:, [num]].logical_not().sum() * x[num].shape[-1]
    return torch.sqrt(loss / all_num)


# 数据
random_state = 100
# 数据
h_data = open('../data/imputation/AE_data.pkl', 'rb')
X_pred, X, missing_index = pickle.load(h_data)
# 计算聚类准确率
# loss = mean_squared_error(y_pred, y, multioutput = 'raw_values', squared = False)

loss = RMSE(8, X_pred, X, missing_index)

print(loss)
