# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:36
# @Author  : nieyuzhou
# @File    : UCI_dataset.py
# @Software: PyCharm
import torch
import numpy as np
from mvlearn.datasets import load_UCImultifeature
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


def normalize(x, min = 0):
    if min == 0:
        scaler = MinMaxScaler([0, 1])
    else:
        scaler = MinMaxScaler((-1, 1))
    norm_x = scaler.fit_transform(x)
    return norm_x


class UCI_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):
        super(UCI_Dataset, self).__init__()
        self.full_data = dict()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        full_data, full_labels = load_UCImultifeature(views = [0,1,2])
        num = len(full_labels)
        classifier_dims = []
        views = len(full_data)
        if name == "train":
            for v in range(views):
                full_data[v] = full_data[v][:int(num * 4 / 5)]
                classifier_dims.append([full_data[v].shape[1]])
            full_labels = full_labels[:int(num * 4 / 5)]
            args.views = views
            args.classifier_dims = classifier_dims
        elif name == "valid":
            for v in range(len(full_data)):
                full_data[v] = full_data[v][int(num * 4 / 5):]
            full_labels = full_labels[int(num * 4 / 5):]
        # 76 Fourier coefficients of the character shapes
        # 216 profile correlations
        # 64 Karhunen-Love coefficients
        # 240 pixel averages in 2 x 3 windows
        # 47 Zernike moments
        # 6 morphological features
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(len(full_data)):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))


        # 测试模态缺失的情况
        # self.full_data[0][:] = 0
        # self.full_data[1][:] = 0
        # self.full_data[2][:] = 0
        # self.full_data[3][:] = 0
        # self.full_data[4][:] = 0
        # self.full_data[5][:] = 0
        # 只有第五个100次epoch为19.5
        # 只有第四个100次epoch为53.25
        # 只有第三个100次epoch为92.25
        # 只有第二个100次epoch为63.0
        # 只有第一个100次epoch为74.0
        # 只有第零个100次epoch为57.0

    def __getitem__(self, idx):
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)
