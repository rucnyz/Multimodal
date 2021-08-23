# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 17:16
# @Author  : nieyuzhou
# @File    : multi_view_dataset.py
# @Software: PyCharm
import os

import scipy.io as sio
from torch.utils.data import Dataset

from utils.preprocess import *
from utils.shuffle import *


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
        dataset = sio.loadmat(root + "/data/" + self.dataset + ".mat")
        full_labels = np.squeeze(dataset["Y"])
        full_labels = full_labels - full_labels.min()
        full_data = np.squeeze(dataset["X"])

        # 打乱数据划分数据集
        if name == "train":
            classifier_dims = []
            args.classes = int(full_labels.max() + 1)
            args.num = len(full_labels)
            args.views = len(full_data)
            full_data, full_labels = shuffle(full_data, full_labels, name, args.seed)
            for v in range(args.views):
                classifier_dims.append(full_data[v].shape[1])
            args.classifier_dims = classifier_dims
        elif name == "valid":
            full_data, full_labels = shuffle(full_data, full_labels, name, args.seed)
        elif name == "test":
            pass

        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(len(full_data)):
            # 这个数据集暂时不要用
            if args.dataset == "NUSWIDEOBJ":
                self.full_data[v] = torch.from_numpy(full_data[v].astype(np.float32))
            else:
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
