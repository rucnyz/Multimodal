# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:36
# @Author  : nieyuzhou
# @File    : UCI_dataset.py
# @Software: PyCharm
from mvlearn.datasets import load_UCImultifeature
from torch.utils.data import Dataset

from utils.preprocess import *


class UCI_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):  # name: train/valid/test
        super(UCI_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']  # assert:断言函数，不满足条件则直接触发异常，不必执行接下来的代码

        self.full_data = dict()
        self.name = name
        full_data, full_labels = load_UCImultifeature()  # full_data是一个长度为6的dict

        if name == "train":
            classifier_dims = []
            # 数据情况：2000个数据，6个模态，10个类别，每个类别200个数据，每个模态数据有不同个数特征
            args.classes = int(full_labels.max() + 1)  # 类别数量
            args.num = len(full_labels)  # 数据总数
            args.views = len(full_data)  # 模态数量
            for v in range(args.views):  # 6个模态
                full_data[v] = full_data[v][:int(args.num * 4 / 5)]   # 取80%作为训练集
                classifier_dims.append(full_data[v].shape[1])
            # classifier_dims为[[76], [216], [64], [240], [47], [6]]，即每个模态的特征数
            full_labels = full_labels[:int(args.num * 4 / 5)]
            args.classifier_dims = classifier_dims
        elif name == "valid":
            for v in range(args.views):
                full_data[v] = full_data[v][int(args.num * 4 / 5):]  # 取20%作为验证集
            full_labels = full_labels[int(args.num * 4 / 5):]
        elif name == "test":
            pass
        # 76 Fourier coefficients of the character shapes
        # 216 profile correlations
        # 64 Karhunen-Love coefficients
        # 240 pixel averages in 2 x 3 windows
        # 47 Zernike moments
        # 6 morphological features

        # torch.from_numpy: 从numpy数组创建一个张量，数组和张量共享相同内存．
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(args.views):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))

    def __getitem__(self, idx): # fetching a data sample 第idx数据的所有模态和label
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self): # return the size of the dataset，即数据量
        return len(self.full_labels)

    def replace_missing_data(self, args, missing_index):
        if self.name == "train":
            for v in range(args.views):
                self.full_data[v][missing_index[:int(args.num * 4 / 5)][:, v] == 0] = -1
        elif self.name == "valid":
            for v in range(args.views):
                self.full_data[v][missing_index[int(args.num * 4 / 5):][:, v] == 0] = -1
