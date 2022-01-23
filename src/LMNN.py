# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 17:27
# @Author  : nieyuzhou
# @File    : LMNN.py
# @Software: PyCharm
from metric_learn import LMNN
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data['data']
Y = iris_data['target']
lmnn = LMNN(k = 5, learn_rate = 1e-6, verbose = True, random_state = 123)
lmnn.fit(X, Y)
lmnn.get_mahalanobis_matrix()
