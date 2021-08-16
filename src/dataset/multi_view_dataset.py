# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 17:16
# @Author  : nieyuzhou
# @File    : multi_view_dataset.py
# @Software: PyCharm
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio
from utils.preprocess import normalize


class Multiview_Dataset(Dataset):
    def __init__(self, name, args):
        super(Multiview_Dataset, self).__init__()
        self.full_data = dict()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataset = args.dataset
        self.full_data = dict()
        root = os.getcwd()
        dataset = sio.loadmat(root + "\\data\\" + self.dataset + ".mat")
        full_labels = dataset["Y"]
        args.classes = full_labels.max() + 1

        full_data = np.squeeze(dataset["X"])
        self.num = len(full_labels)
        classifier_dims = []
        views = len(full_data)

        if self.name == "train":
            for v in range(views):
                self.full_data[v] = full_data[v][:int(self.num * 4 / 5)]
                classifier_dims.append([full_data[v].shape[1]])
            self.full_labels = full_labels[:int(self.num * 4 / 5)]
            args.views = views
            args.classifier_dims = classifier_dims
        elif name == "valid":
            for v in range(views):
                self.full_data[v] = full_data[v][int(self.num * 4 / 5):]
            self.full_labels = full_labels[int(self.num * 4 / 5):]

        for v in range(len(full_data)):
            self.full_data[v] = torch.from_numpy(normalize(self.full_data[v]).astype(np.float32))
        self.full_labels = torch.from_numpy(self.full_labels.astype(np.int64))

    def __getitem__(self, idx):
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)
