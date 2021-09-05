# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 17:55
# @Author  : nieyuzhou
# @File    : preprocess.py
# @Software: PyCharm

import numpy as np
import torch
from numpy.random import randint
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder


def missing_data_process(args, train_data, valid_data, missing_index):
    if args.model == "TMC":
        train_data.replace_missing_data(args, missing_index)
        valid_data.replace_missing_data(args, missing_index)
        args.train_batch_size = args.batch_size
        args.valid_batch_size = args.batch_size
    elif args.model == "CPM":
        args.batch_size = args.num
        args.train_batch_size = int(args.num * 4 / 5)
        args.valid_batch_size = args.num - int(args.num * 4 / 5)
        print("人为设定batch size弃用，现使用整个数据集作为一个batch")
        # 所以为啥batch调成了一整个训练集大小呢，如果多次batch的话就需要初始化多个lsd_train(每个都得单独训练因为不同batch本身概率分布
        # 也是不同的，更不用说有的非整数训练集最后一个batch大小还和前面不一样，而且这样分成多个矩阵运算也会使得速度更慢，大矩阵运算相比于多个
        # 小矩阵运算绝对会快很多很多很多，因为可以并行)


def Normalize(data):
    """
    :param data:Input data
    :return:normalized data
    """
    m = np.mean(data)
    mx = np.max(data)
    mn = np.min(data)
    return (data - m) / (mx - mn)


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
    # missing_rate较高，每个数据分配一个模态
    if one_rate <= (1 / view_num):
        #  OneHotEncoder transforms each categorical feature with n_categories possible values into
        #  n_categories binary features, with one of them 1, and all others 0.
        enc = OneHotEncoder()
        # Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [low, high).
        # If high is None (the default), then results are from [0, low).
        # (2000,6)0/1矩阵
        view_preserve = enc.fit_transform(randint(0, view_num, size = (alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:  # 模态不缺失
        # (2000,6)全是1矩阵
        matrix = randint(1, 2, size = (alldata_len, view_num))
        return matrix
    # missing_rate较低，每个数据的模态不止一个
    while error >= 0.005:
        enc = OneHotEncoder()
        # 2000个数据每个数据分配一个模态
        view_preserve = enc.fit_transform(randint(0, view_num, size = (alldata_len, 1))).toarray()
        # 2000个数据每个数据分配一个模态后，还需要分配的模态总数
        one_num = view_num * alldata_len * one_rate - alldata_len
        # 每个数据还要分配多少个模态
        ratio = one_num / (view_num * alldata_len)
        # 保证每个数据分配的模态个数不超过ratio，约为ratio
        # 若randint(0, 100, size = (alldata_len, view_num)小于ratio * 100则为1，大于则为0
        matrix_iter = (randint(0, 100, size = (alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        # 被分配了超过一遍（即重复分配）的模态总数
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        # (1 - a / one_num): 分配到的比例（剩下a个需要继续分配）
        one_num_iter = one_num / (1 - a / one_num)
        # 因为有重复分配的，ratio一定小于one_rate，因此再分配一次
        # 再分配one_num_iter个模态，其中(one_num-a)/one_num*one_num_iter个是只分配一次的，接近one_num，one_num中又有a个重复分配的？？？
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size = (alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        # ratio与one_rate接近
        error = abs(one_rate - ratio)

    return matrix


def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    a = np.random.uniform(low, high, (fan_in, fan_out))
    a = a.astype('float32')
    a = torch.from_numpy(a)
    if torch.cuda.is_available():
        a = a.cuda()
    return a
