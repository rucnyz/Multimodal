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

from utils.shuffle import *
from utils.preprocess import *


class Multiview_Dataset(Dataset):
    def __init__(self, name, args):
        super(Multiview_Dataset, self).__init__()
        self.full_data = dict()
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args
        self.dataset = args.dataset
        self.full_data = dict()
        # 导入数据
        root = os.getcwd()
        dataset = sio.loadmat(root + "\\data\\" + self.dataset + ".mat")
        full_labels = np.squeeze(dataset["Y"])
        full_labels = full_labels - full_labels.min()
        full_data = np.squeeze(dataset["X"])

        args.classes = int(full_labels.max() + 1)
        self.num = len(full_labels)
        classifier_dims = []
        views = len(full_data)
        # 打乱数据划分数据集
        if name == "train":
            full_data, full_labels = shuffle(full_data, full_labels, name, args.seed)
            for v in range(views):
                classifier_dims.append([full_data[v].shape[1]])
            args.views = views
            args.classifier_dims = classifier_dims
        elif name == "valid":
            full_data, full_labels = shuffle(full_data, full_labels, name, args.seed)

        for v in range(len(full_data)):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))

    def __getitem__(self, idx):
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        # 返回所有数据
        return idx, data, target

    def __len__(self):
        return len(self.full_labels)
