# -*- coding: utf-8 -*-
# @Time    : 2022/1/20 21:38
# @Author  : nieyuzhou
# @File    : cluster.py
# @Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_mutual_info_score


def cluster_accuracy(y_pred, y_true):
    """
    调用匈牙利算法，实现最佳类别的分配
    """
    D = max(y_pred.max(), y_true.max()) + 1
    cost = np.zeros((D, D), dtype = np.int64)
    for i in range(y_pred.size):
        cost[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(np.max(cost) - cost)
    ind = np.array(ind).T
    return sum([cost[i, j] for i, j in ind]) * 1.0 / y_pred.size


random_state = 100
# 数据（TODO 替换为自己的数据）
X, y = make_blobs(n_samples = [300, 300, 400], centers = None, n_features = 2, random_state = random_state)
# KMeans聚类
kmeans = KMeans(n_clusters = 3, random_state = random_state)
y_pred = kmeans.fit_predict(X)
# 计算AMI
print(adjusted_mutual_info_score(y_pred, y))
# 计算聚类准确率
print(cluster_accuracy(y_pred, y))
# 画出聚类的图像
plt.figure(figsize = (12, 12))
plt.scatter(X[:, 0], X[:, 1], c = y_pred)
plt.show()
