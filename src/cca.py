# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 0:28
# @Author  : nieyuzhou
# @File    : cca.py
# @Software: PyCharm
import argparse
import os

import torch
from mvlearn.embed import MCCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataset.UCI_dataset import UCI_Dataset
from dataset.UKB_dataset import UKB_Dataset
from dataset.multi_view_dataset import Multiview_Dataset
from utils.loss_func import classification_loss
from utils.pred_func import accuracy_count
from utils.preprocess import *

if os.getcwd().endswith("src"):
    os.chdir("../")
parser = argparse.ArgumentParser()

parser.add_argument('--missing_rate', type = float, default = 0,
                    help = 'view missing rate [default: 0]')
parser.add_argument("--mode", default = 'client')
parser.add_argument("--port", default = 52162)
args = parser.parse_args()
train_dset = UCI_Dataset('train', args)
eval_dset = UCI_Dataset('valid', args)
X = list(train_dset.full_data.values())
X_valid = list(eval_dset.full_data.values())
y = train_dset.full_labels.__array__()
y_valid = eval_dset.full_labels.__array__()
# 设置好丢失模态
missing_index = get_missing_index(args.views, args.num, args.missing_rate)
train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])
# 均值填充

# 使用CCA方法得到隐藏层
components = 1
mcca = MCCA(n_components = components, regs = 0.1)
scores = mcca.fit_transform(X)
s = np.empty((scores.shape[1], scores.shape[0] * scores.shape[2]))
for i in range(scores.shape[0]):
    s[:, i * components:(i + 1) * components] = scores[i]
# 使用非参数方法测试，训练隐藏层(这个不用了，没有训练的部分完全无法起到对比的作用)
# y_onehot = torch.zeros(y.shape[0], args.classes, device = 'cpu').scatter_(1, y.reshape(y.shape[0], 1), 1)
# clf_loss, predicted = classification_loss(y_onehot, y, torch.Tensor(s))
# accuracy = accuracy_score(y, predicted)
# all_num = y.size(0)
# accuracy = accuracy / all_num
# print("非参方法:{:.4f}".format(accuracy))
# 使用SVM测试
clf = SVC().fit(s, y)
y_predict = clf.predict(s)
svm_acc = accuracy_score(y, y_predict)
print("训练集:\n svm:{:.4f}".format(svm_acc))
# -------------------------------------
# 验证集试试
scores = mcca.fit_transform(X_valid)
s_valid = np.empty((scores.shape[1], scores.shape[0] * scores.shape[2]))
for i in range(scores.shape[0]):
    s_valid[:, i * components:(i + 1) * components] = scores[i]
y_predict = clf.predict(s_valid)
svm_acc = accuracy_score(y_valid, y_predict)
print("验证集:\n svm:{:.4f}".format(svm_acc))
