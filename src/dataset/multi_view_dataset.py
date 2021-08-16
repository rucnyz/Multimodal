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
        dataset = sio.loadmat(root+"\\data\\"+self.dataset+".mat")
        full_labels = dataset["Y"]
        full_data = np.squeeze(dataset["X"])

        for v in range(len(full_data)):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))
        args.classes = full_labels.max()

        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))