# -*- coding: utf-8 -*-
# @Time    : 2022/1/28 12:35
# @Author  : HCY
# @File    : UKB_balanced_dataset.py
# @Software: PyCharm

import os
import pickle
import torch

import pandas as pd
from torch.utils.data import Dataset

from utils.preprocess import *
from numpy.random import randint


class UKB_BALANCED_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):  # name: train/valid/test
        super(UKB_BALANCED_Dataset, self).__init__()
        self.missing_index = None
        assert name in ['train', 'valid', 'test']  # assert:断言函数，不满足条件则直接触发异常，不必执行接下来的代码

        # dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')
        if os.getcwd().endswith("src"):
            os.chdir("../")
        elif os.getcwd().endswith("supervised") or os.getcwd().endswith("unsupervised"):
            os.chdir("../../../")
        dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')

        self.full_data = dict()
        self.name = name

        full_data = pickle.load(open(dataroot + "/data_balanced.pkl", "rb"))
        full_labels = pickle.load(open(dataroot + "/label_balanced.pkl", "rb"))

        if name == "train":
            classifier_dims = []
            # 数据情况：34240个数据，8个模态，2个类别，每个模态数据有不同个数特征
            args.classes = int(full_labels.max() + 1)  # 类别数量
            args.num = len(full_labels)  # 数据总数
            args.views = len(full_data)  # 模态数量
            args.weight = torch.tensor(np.bincount(full_labels).max() / np.bincount(full_labels), dtype = torch.float32)
            self.views = args.views
            for v in range(args.views):  # 8个模态
                full_data[v] = full_data[v][:int(args.num * 4 / 5)]  # 取80%作为训练集
                classifier_dims.append(full_data[v].shape[1])
            # classifier_dims为[4, 6, 7, 19, 7, 4, 38, 2]，即每个模态的特征数
            full_labels = full_labels[:int(args.num * 4 / 5)]
            args.classifier_dims = classifier_dims
        elif name == "valid":
            self.views = args.views
            for v in range(args.views):
                full_data[v] = full_data[v][int(args.num * 4 / 5):]  # 取20%作为验证集
            full_labels = full_labels[int(args.num * 4 / 5):]
        elif name == "test":
            pass

        # torch.from_numpy: 从numpy数组创建一个张量，数组和张量共享相同内存．
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(args.views):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))

    def __getitem__(self, idx):  # fetching a data sample 第idx数据的所有模态和label
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        missing_index = self.missing_index[idx]
        # 返回所有数据
        return idx, data, target, missing_index
        # return list(data.values()), target

    def __len__(self):  # return the size of the dataset，即数据量
        return len(self.full_labels)

    def replace_with_zero(self, args, missing_index):
        if self.name == "train":
            for v in range(args.views):
                self.full_data[v][missing_index[:int(args.num * 4 / 5)][:, v] == 0] = 0
        elif self.name == "valid":
            for v in range(args.views):
                self.full_data[v][missing_index[int(args.num * 4 / 5):][:, v] == 0] = 0

    def set_missing_index(self, missing_index):
        self.missing_index = missing_index

    # 均值替换缺失值
    def replace_with_mean(self):
        for v in range(self.views):
            self.full_data[v][self.missing_index[:, v] == 0] = self.full_data[v].mean(dim = 0)