# -*- coding: utf-8 -*-
# @Time    : 2022/1/24 21:35
# @Author  : HCY
# @File    : UKB_ad_dataset.py
# @Software: PyCharm
import os
import pickle

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from utils.preprocess import *


def preprocess_data():
    dataroot = os.path.join(os.getcwd() + '/../../data' + '/ukb_data')

    data1 = pd.read_stata(dataroot + '/【控制变量】depression_covariant.dta')
    data1 = data1[data1['n_eid'] > 0]
    data2 = pd.read_stata(dataroot + '/history.dta')
    data3 = pd.read_stata(dataroot + '/20002.dta')
    data4 = pd.read_stata(dataroot + '/ad+mci.dta')

    data2['fa_his'] = np.nan
    data2['fa_his'][data2['n_20107_0_0'] == 10] = 1
    data2['fa_his'][
        [True if data2['n_20107_0_0'][i] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 11, 13, -17, -27] else False for i in
         range(len(data2))]] = 0
    data2['mo_his'] = np.nan
    data2['mo_his'][data2['n_20110_0_0'] == 10] = 1
    data2['mo_his'][
        [True if data2['n_20110_0_0'][i] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 11, 13, -17, -27] else False for i in
         range(len(data2))]] = 0
    data2['bro_his'] = np.nan
    data2['bro_his'][data2['n_20111_0_0'] == 10] = 1
    data2['bro_his'][
        [True if data2['n_20111_0_0'][i] in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 11, 13, -17, -27] else False for i in
         range(len(data2))]] = 0
    data2['family_history_ad'] = data2['fa_his'] + data2['mo_his'] + data2['bro_his']
    data2[data2['family_history_ad'] >= 1] = 1

    data3 = data3[[True if data3['n_20002_0_0'][i] not in [1263, 1266, 1240, 1262, 1261, 1397, 1258] else False for i in
                   range(len(data3))]]

    data4['disease_inc'] = 0
    data4['disease_inc'][(data4['mci_inc'] == 0) & (data4['ad_inc'] == 0)] = 0
    data4['disease_inc'][(data4['mci_inc'] == 1) & (data4['ad_inc'] == 0)] = 1
    data4['disease_inc'][(data4['mci_inc'] == 0) & (data4['ad_inc'] == 1)] = 2
    data4['disease_inc'][(data4['mci_inc'] == 1) & (data4['ad_inc'] == 1)] = np.nan

    data_all = pd.merge(data1, data2, on = 'n_eid', how = 'inner')
    data_all = pd.merge(data_all, data3['n_eid'], on = 'n_eid', how = 'inner')
    data_all = pd.merge(data_all, data4[['disease_inc', 'n_eid']], on = 'n_eid', how = 'inner')

    population = ["age", "sex", "family_history_ad"]
    economy = ["lowincome", "workstatus", "highschool", "isolation2", "deprivation"]
    lifestyle = ["healthy_PA", "healthy_diet", "healthy_smoking", "healthy_alcohol", "healthy_obesity",
                 "sleep_score"]
    ill = ["disease_inc"]

    data_final = data_all[population + economy + lifestyle + ill]
    data_final = data_final.replace('NA', np.nan)
    data_final = data_final.dropna(axis = 0, how = 'any')  # drop all rows that have any NaN values

    full_data = {0: data_final[population], 1: data_final[economy], 2: data_final[lifestyle]}

    full_labels = data_final[ill].values.squeeze()
    return full_data, full_labels


class UKB_AD_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):  # name: train/valid/test
        super(UKB_AD_Dataset, self).__init__()
        assert name in ['train', 'valid', 'test']  # assert:断言函数，不满足条件则直接触发异常，不必执行接下来的代码

        if os.getcwd().endswith("src"):
            os.chdir("../")
        elif os.getcwd().endswith("supervised") or os.getcwd().endswith("unsupervised"):
            os.chdir("../../../")
        dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')

        self.full_data = dict()
        self.name = name

        full_data = pickle.load(open(dataroot + "/data_ad.pkl", "rb"))
        full_labels = pickle.load(open(dataroot + "/label_ad.pkl", "rb"))

        if name == "train":
            classifier_dims = []
            # 数据情况：232962个数据，3个模态，3个类别，每个模态数据有不同个数特征
            args.classes = int(full_labels.max() + 1)  # 类别数量
            args.num = len(full_labels)  # 数据总数
            args.views = len(full_data)  # 模态数量
            self.views = args.views
            np.bincount(full_labels)
            args.weight = torch.tensor(np.bincount(full_labels).max() / np.bincount(full_labels), dtype = torch.float32)
            for v in range(args.views):  # 3个模态
                full_data[v] = full_data[v][:int(args.num * 4 / 5)]  # 取80%作为训练集
                classifier_dims.append(full_data[v].shape[1])
            # classifier_dims为[3,5,6]，即每个模态的特征数
            full_labels = full_labels[:int(args.num * 4 / 5)]
            args.classifier_dims = classifier_dims
        elif name == "valid":
            self.views = args.views
            for v in range(args.views):
                full_data[v] = full_data[v][int(args.num * 4 / 5):]  # 取20%作为验证集
            full_labels = full_labels[int(args.num * 4 / 5):]
        elif name == "test":
            pass

        # torch.from_numpy: 从numpy数组创建一个张量，数组和张量共享相同内存．
        self.full_labels = torch.from_numpy(full_labels.astype(np.int64))
        for v in range(args.views):
            self.full_data[v] = torch.from_numpy(normalize(full_data[v]).astype(np.float32))

    def __getitem__(self, idx):  # fetching a data sample 第idx数据的所有模态和label
        data = dict()
        for i in range(len(self.full_data)):
            data[i] = self.full_data[i][idx]
        target = self.full_labels[idx]
        missing_index = self.missing_index[idx]
        # 返回所有数据
        return idx, data, target, missing_index
        # return list(data.values()), target

    def __len__(self):  # return the size of the dataset，即数据量
        return len(self.full_labels)

    def replace_with_zero(self, args, missing_index):
        if self.name == "train":
            for v in range(args.views):
                self.full_data[v][missing_index[:int(args.num * 4 / 5)][:, v] == 0] = 0
        elif self.name == "valid":
            for v in range(args.views):
                self.full_data[v][missing_index[int(args.num * 4 / 5):][:, v] == 0] = 0

    def set_missing_index(self, missing_index):
        self.missing_index = missing_index

    # 均值替换缺失值
    def replace_with_mean(self):
        for v in range(self.views):
            self.full_data[v][self.missing_index[:, v] == 0] = self.full_data[v].mean(dim = 0)

    def replace_with_nan(self):
        for v in range(self.views):
            self.full_data[v][self.missing_index[:, v] == 0] = np.NaN


if __name__ == '__main__':
    full_data, full_labels = preprocess_data()
    full_labels = full_labels.astype(np.int32)
    dataroot = os.path.join(os.path.dirname(os.path.dirname(os.path.join(os.getcwd()))) + '/data' + '/ukb_data')
    pickle.dump(full_data, open(dataroot + "/data_ad.pkl", "wb"))
    pickle.dump(full_labels, open(dataroot + "/label_ad.pkl", "wb"))
