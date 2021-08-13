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
    def __init__(self, name, args, token_to_ix = None):
        super(UCI_Dataset, self).__init__()
        self.token_to_ix = token_to_ix
        assert name in ['train', 'valid', 'test']
        self.name = name
        self.args = args

        full_data, self.full_labels = load_UCImultifeature()
        # 76 Fourier coefficients of the character shapes
        self.fou_data = pd.DataFrame(full_data[0])
        # 216 profile correlations
        self.fac_data = pd.DataFrame(full_data[1])
        # 64 Karhunen-Love coefficients
        self.kar_data = pd.DataFrame(full_data[2])
        # 240 pixel averages in 2 x 3 windows
        self.pix_data = pd.DataFrame(full_data[3])
        # 47 Zernike moments
        self.zer_data = pd.DataFrame(full_data[4])
        # 6 morphological features
        self.mor_data = pd.DataFrame(full_data[5])

        # TODO 可能存在的预处理

    def __getitem__(self, idx):
        # 返回8项数据为一个字典
        data = dict()
        # 还没改好
        return idx, torch.from_numpy(self.fou_data), torch.from_numpy(self.fac_data), torch.from_numpy(
            self.kar_data), torch.from_numpy(self.pix_data), torch.from_numpy(self.zer_data), torch.from_numpy(
            self.mor_data), torch.from_numpy(self.full_labels)

    def __len__(self):
        return len(self.full_labels)
