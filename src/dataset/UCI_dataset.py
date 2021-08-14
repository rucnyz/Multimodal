# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:36
# @Author  : nieyuzhou
# @File    : UCI_dataset.py
# @Software: PyCharm
import pandas as pd
import torch
from mvlearn.datasets import load_UCImultifeature
from torch.utils.data import Dataset


class UCI_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):
        super(UCI_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args

        self.full_data, self.full_labels = load_UCImultifeature()
        # 76 Fourier coefficients of the character shapes
        # 216 profile correlations
        # 64 Karhunen-Love coefficients
        # 240 pixel averages in 2 x 3 windows
        # 47 Zernike moments
        # 6 morphological features
        self.full_labels = self.full_labels.astype("int64")
        for v in range(len(self.full_data)):
            self.full_data[v] = self.full_data[v].astype("float32")
        # TODO 可能存在的预处理

    def __getitem__(self, idx):
        data = []
        for i in range(len(self.full_data)):
            data.append(torch.from_numpy(self.full_data[i][idx]))
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)
