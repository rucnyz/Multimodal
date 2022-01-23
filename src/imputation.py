# -*- coding: utf-8 -*-
# @Time    : 2022/1/23 20:43
# @Author  : nieyuzhou
# @File    : imputation.py
# @Software: PyCharm
import pickle

import numpy as np
from sklearn.metrics import mean_squared_error

# 数据
# h_data = open('data/representations/imputation_data.pkl', 'rb')
# y_pred, y = pickle.load(h_data)
y_pred = np.random.randn(100, 3)
y = np.random.randn(100, 3)
# 计算聚类准确率
# loss = mean_squared_error(y_pred, y, multioutput = 'raw_values', squared = False)
loss = mean_squared_error(y_pred, y, squared = False)
print(loss)
