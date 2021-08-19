# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:36
# @Author  : nieyuzhou
# @File    : UCI_dataset.py
# @Software: PyCharm
import torch
from mvlearn.datasets import load_UCImultifeature
from torch.utils.data import Dataset

from utils.preprocess import *


class UCI_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args, index = None):
        super(UCI_Dataset, self).__init__()
        self.full_data = dict()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        full_data, full_labels = load_UCImultifeature()
        args.classes = int(full_labels.max() + 1)
        num = len(full_labels)
        classifier_dims = []
        views = len(full_data)
        missing_index = dict()
        if name == "train":
            self.missing_index = get_missing_index(views, num, args.missing_rate)
            for v in range(views):
                full_data[v] = full_data[v][:int(num * 4 / 5)]
                classifier_dims.append([full_data[v].shape[1]])
                missing_index[v] = self.missing_index[:int(num * 4 / 5)][:, v]
            full_labels = full_labels[:int(num * 4 / 5)]
            args.views = views
            args.classifier_dims = classifier_dims
        elif name == "valid":
            for v in range(views):
                full_data[v] = full_data[v][int(num * 4 / 5):]
                missing_index[v] = index[int(num * 4 / 5):][:, v]
            full_labels = full_labels[int(num * 4 / 5):]
        elif name == "test":
            pass
        # 76 Fourier coefficients of the character shapes
        # 216 profile correlations
        # 64 Karhunen-Love coefficients
        # 240 pixel averages in 2 x 3 windows
        # 47 Zernike moments
        # 6 morphological features
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(len(full_data)):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))
            # 将缺失值设置为-1
            self.full_data[v][missing_index[v] == 0] = -1

    def __getitem__(self, idx):
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)
