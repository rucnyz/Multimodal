# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 17:55
# @Author  : nieyuzhou
# @File    : preprocess.py
# @Software: PyCharm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder


def normalize(x, min = 0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x


def get_missing_index(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 3.2 of the paper
    :return:Sn
    """
    one_rate = 1 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size = (alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size = (alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size = (alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size = (alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size = (alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)

    return matrix
