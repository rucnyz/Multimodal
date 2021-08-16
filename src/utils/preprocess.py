# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 17:55
# @Author  : nieyuzhou
# @File    : preprocess.py
# @Software: PyCharm
from sklearn.preprocessing import MinMaxScaler
def normalize(x, min = 0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x