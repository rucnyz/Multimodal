# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 22:53
# @Author  : nieyuzhou
# @File    : shuffle.py
# @Software: PyCharm
from sklearn.model_selection import StratifiedShuffleSplit


def shuffle(data, label, flag, seed):
    y_train, y_test = 0, 0
    sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = seed)
    X_train, X_test = dict(), dict()
    for v in range(len(data)):
        # 这里循环的次数由n_splits决定
        for train_index, test_index in sss.split(data[v], label):
            X_train[v], X_test[v] = data[v][train_index], data[v][test_index]
            y_train, y_test = label[train_index], label[test_index]
    if flag == "train":
        return X_train, y_train
    elif flag == "valid":
        return X_test, y_test
