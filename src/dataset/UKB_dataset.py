# -*- coding: utf-8 -*-
# @Time    : 2021/9/7 15:54
# @Author  : HCY
# @File    : UKB_dataset.py
# @Software: PyCharm
import os
import pickle

import numpy as np
import torch

import pandas as pd
from torch.utils.data import Dataset

from utils.preprocess import *
from numpy.random import randint
from fancyimpute import IterativeSVD


def preprocess_data():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataroot = os.path.join(os.getcwd() + '/../../data' + '/ukb_data')

    data1 = pd.read_stata(dataroot + '/【控制变量】depression_covariant.dta')
    data1 = data1[data1['n_eid'] > 0]
    data2 = pd.read_stata(dataroot + '/mdd_grs.dta')
    data2 = data2[data2['iid'] > 0]
    data2.rename(columns = {'iid': 'n_eid'}, inplace = True)
    data3 = pd.read_stata(dataroot + '/【结局变量】depression_outcome.dta')
    data_all = pd.merge(data1, data2, on = 'n_eid', how = 'inner')
    data_all = pd.merge(data_all, data3[['dep_inc', 'n_eid']], on = 'n_eid', how = 'inner')
    data_all.drop(["n_eid", "pat_x", "pat_y", "mat_x", "mat_y", "sex_x", "gender", "fid_x", "fid_y", "phenotype_x",
                   "phenotype_y"], axis = 1, inplace = True)
    data_all.rename(columns = {'sex_y': 'sex'}, inplace = True)

    population = ["age", "sex", "screening", "family_history"]
    economy = ["lowincome", "workstatus", "highschool", "isolation2", "deprivation", "housing_tenure"]
    lifestyle = ["healthy_PA", "healthy_diet", "healthy_smoking", "healthy_alcohol", "healthy_obesity",
                 "sleep_score",
                 "healthy_score"]
    blood = ["n_30010_0_0", "n_30000_0_0", "n_30100_0_0", "n_30750_0_0", "n_30040_0_0", "n_30050_0_0",
             "n_30060_0_0",
             "n_30070_0_0", "n_30080_0_0", "n_30150_0_0", "n_30210_0_0", "n_30140_0_0", "n_30200_0_0",
             "n_30130_0_0",
             "n_30190_0_0", "n_30160_0_0", "n_30220_0_0", "n_30120_0_0", "n_30180_0_0"]
    metabolism = ["n_30740_0_0", "n_30690_0_0", "n_30870_0_0", "n_30780_0_0", "n_30760_0_0", "n_30640_0_0",
                  "n_30630_0_0"]
    urine = ["n_30510_0_0", "n_30500_0_0", "n_30520_0_0", "n_30530_0_0"]
    gene = data_all.columns[285:323].tolist()
    others = ["n_20002_0_0", "n_20003_0_0"]
    ill = ["dep_inc"]

    data_final_all = data_all[population + economy + lifestyle + blood + metabolism + urine + others + gene + ill]
    data_final_all = data_final_all.replace('NA', np.nan)

    # 有缺失值
    healthy_all = data_final_all[data_final_all["dep_inc"] == 0]
    depressed_all = data_final_all[data_final_all["dep_inc"] == 1]
    data_final_all_balanced = pd.concat([depressed_all, healthy_all.sample(n = len(depressed_all))])

    full_data_all = {0: data_final_all_balanced[population], 1: data_final_all_balanced[economy],
                     2: data_final_all_balanced[lifestyle], 3: data_final_all_balanced[blood],
                     4: data_final_all_balanced[metabolism], 5: data_final_all_balanced[urine],
                     6: data_final_all_balanced[gene], 7: data_final_all_balanced[others]}
    full_labels_all = data_final_all_balanced[ill].values.squeeze()

    # data_final_all_balanced = data_final_all_balanced.replace('NA', np.nan)
    alldata_len = len(data_final_all_balanced)
    view_num = len(full_data_all)
    matrix = randint(1, 2, size = (alldata_len, view_num))
    for v in range(view_num):
        for i in range(alldata_len):
            if full_data_all[v].iloc[[i]].T.isnull().any().bool():
                matrix[i][v] = 0

    # 无缺失值
    # data_final = data_final_all.replace('NA', np.nan)
    data_final = data_final_all.dropna(axis = 0, how = 'any')  # drop all rows that have any NaN values

    full_data = {0: data_final[population], 1: data_final[economy], 2: data_final[lifestyle], 3: data_final[blood],
                 4: data_final[metabolism], 5: data_final[urine], 6: data_final[gene], 7: data_final[others]}

    full_labels = data_final[ill].values.squeeze()

    healthy = data_final[data_final["dep_inc"] == 0]
    depressed = data_final[data_final["dep_inc"] == 1]
    data_final_balanced = pd.concat([depressed, healthy.sample(n = len(depressed))])

    full_data_balanced = {0: data_final_balanced[population], 1: data_final_balanced[economy],
                          2: data_final_balanced[lifestyle], 3: data_final_balanced[blood],
                          4: data_final_balanced[metabolism], 5: data_final_balanced[urine],
                          6: data_final_balanced[gene], 7: data_final_balanced[others]}

    full_labels_balanced = data_final_balanced[ill].values.squeeze()
    return full_data_all, full_labels_all, full_data, full_labels, full_data_balanced, full_labels_balanced, matrix


class UKB_Dataset(Dataset):
    # 把dataroot改成mimic的目录
    def __init__(self, name, args):  # name: train/valid/test
        super(UKB_Dataset, self).__init__()
        self.missing_index = None
        assert name in ['train', 'valid', 'test']  # assert:断言函数，不满足条件则直接触发异常，不必执行接下来的代码

        # dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')
        if os.getcwd().endswith("src"):
            os.chdir("../")
        elif os.getcwd().endswith("supervised") or os.getcwd().endswith("unsupervised"):
            os.chdir("../../../")
        dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')

        self.full_data = dict()
        self.name = name

        self.device = args.device

        full_data = pickle.load(open(dataroot + "/data.pkl", "rb"))
        full_labels = pickle.load(open(dataroot + "/label.pkl", "rb"))

        if name == "train":
            classifier_dims = []
            # 数据情况：34240个数据，8个模态，2个类别，每个模态数据有不同个数特征
            args.classes = int(full_labels.max() + 1)  # 类别数量
            args.num = len(full_labels)  # 数据总数
            args.views = len(full_data)  # 模态数量
            args.weight = torch.tensor(np.bincount(full_labels).max() / np.bincount(full_labels), dtype = torch.float32)
            self.views = args.views
            for v in range(args.views):  # 8个模态
                full_data[v] = full_data[v][:int(args.num * 4 / 5)]  # 取80%作为训练集
                classifier_dims.append(full_data[v].shape[1])
            # classifier_dims为[4, 6, 7, 19, 7, 4, 38, 2]，即每个模态的特征数
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

    # def feat_concat(self, x):
    #     concat_X = torch.tensor([], device=self.device)
    #     for i in range(self.views):
    #         concat_X = torch.cat((concat_X, x[i].to(self.device)), dim=1)
    #     return concat_X

    def replace_with_nan(self):
        for v in range(self.views):
            self.full_data[v][self.missing_index[:, v] == 0] = np.NaN
        # full_data_concat = UKB_Dataset.feat_concat(self, self.full_data)
        # self.full_data = torch.Tensor(IterativeSVD().fit_transform(full_data_concat))



if __name__ == '__main__':
    full_data_all, full_labels_all, full_data, full_labels, full_data_balanced, full_labels_balanced, matrix = preprocess_data()
    full_labels = full_labels.astype(np.int32)
    full_labels_all = full_labels_all.astype(np.int32)
    dataroot = os.path.join(os.path.dirname(os.path.dirname(os.path.join(os.getcwd()))) + '/data' + '/ukb_data')
    # pickle.dump(full_data, open(dataroot + "/data.pkl", "wb"))
    # pickle.dump(full_labels, open(dataroot + "/label.pkl", "wb"))
    pickle.dump(full_data_all, open(dataroot + "/data_all.pkl", "wb"))
    pickle.dump(full_labels_all, open(dataroot + "/label_all.pkl", "wb"))
    pickle.dump(full_data_balanced, open(dataroot + "/data_balanced.pkl", "wb"))
    pickle.dump(full_labels_balanced, open(dataroot + "/label_balanced.pkl", "wb"))
    pickle.dump(matrix, open(dataroot + "/missing_index_all.pkl", "wb"))
