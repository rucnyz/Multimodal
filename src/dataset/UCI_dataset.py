# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 10:36
# @Author  : nieyuzhou
# @File    : UCI_dataset.py
# @Software: PyCharm
from mvlearn.datasets import load_UCImultifeature
from torch.utils.data import Dataset

from utils.preprocess import *


class UCI_Dataset(Dataset):
    def __init__(self, name, args):
        super(UCI_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']

        self.full_data = dict()
        self.name = name
        full_data, full_labels = load_UCImultifeature()

        if name == "train":
            classifier_dims = []
            args.classes = int(full_labels.max() + 1)
            args.num = len(full_labels)
            args.views = len(full_data)
            for v in range(args.views):
                full_data[v] = full_data[v][:int(args.num * 4 / 5)]
                classifier_dims.append(full_data[v].shape[1])
            full_labels = full_labels[:int(args.num * 4 / 5)]
            args.classifier_dims = classifier_dims
        elif name == "valid":
            for v in range(args.views):
                full_data[v] = full_data[v][int(args.num * 4 / 5):]
            full_labels = full_labels[int(args.num * 4 / 5):]
        elif name == "test":
            pass
        # 76 Fourier coefficients of the character shapes
        # 216 profile correlations
        # 64 Karhunen-Love coefficients
        # 240 pixel averages in 2 x 3 windows
        # 47 Zernike moments
        # 6 morphological features
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(args.views):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))

    def __getitem__(self, idx):
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)

    def replace_missing_data(self, args, missing_index):
        if self.name == "train":
            for v in range(args.views):
                self.full_data[v][missing_index[:int(args.num * 4 / 5)][:, v] == 0] = -1
        elif self.name == "valid":
            for v in range(args.views):
                self.full_data[v][missing_index[int(args.num * 4 / 5):][:, v] == 0] = -1
