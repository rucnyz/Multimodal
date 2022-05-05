# -*- coding: utf-8 -*-
# @Time    : 2022/1/20 21:38
# @Author  : nieyuzhou
# @File    : cluster.py
# @Software: PyCharm
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans, MeanShift
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.manifold import TSNE

if os.getcwd().endswith("src"):
    os.chdir("../")


def cluster_accuracy(y_predict, y_true):
    """
    调用匈牙利算法，实现最佳类别的分配
    """
    D = max(y_predict.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype = np.int64)
    for i in range(y_predict.size):
        cost[y_predict[i], y_true[i]] += 1
    ind = linear_sum_assignment(np.max(cost) - cost)
    ind = np.array(ind).T
    return sum([cost[i, j] for i, j in ind]) * 1.0 / y_predict.size


random_state = 100
# 数据
h_data = open('data/representations/AE_data.pkl', 'rb')
X, y = pickle.load(h_data)
X.requires_grad_(False)
# X, y = make_blobs(n_samples = [300, 300, 400], centers = None, n_features = 2, random_state = random_state)
# KMeans聚类
cluster = KMeans(n_clusters = 10, random_state = random_state)
# cluster = DBSCAN(eps = 3, min_samples = 200)
y_pred = cluster.fit_predict(X)
# 计算AMI
print(adjusted_mutual_info_score(y_pred, y))
# 计算聚类准确率
#print(cluster_accuracy(y_pred, y))
# 画出聚类的图像
# tsne = TSNE(learning_rate = 'auto')
# x_new = tsne.fit_transform(X)
# plt.figure(figsize = (12, 6))
# plt.scatter(x_new[:, 0], x_new[:, 1], c = y_pred)
# plt.savefig('./learn/CPM_GAN(U).svg', format='svg')

