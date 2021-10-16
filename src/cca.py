# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 0:28
# @Author  : nieyuzhou
# @File    : cca.py
# @Software: PyCharm
import argparse
import os

from mvlearn.embed import MCCA

from dataset.UKB_dataset import UKB_Dataset
from utils.loss_func import classification_loss
from utils.pred_func import accuracy_count
from utils.preprocess import *
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
if os.getcwd().endswith("src"):
    os.chdir("../")
parser = argparse.ArgumentParser()

parser.add_argument('--missing_rate', type = float, default = 0,
                    help = 'view missing rate [default: 0]')
parser.add_argument("--mode", default = 'client')
parser.add_argument("--port", default = 52162)
args = parser.parse_args()
train_dset = UKB_Dataset('train', args)
eval_dset = UKB_Dataset('valid', args)
X = list(train_dset.full_data.values())
y = train_dset.full_labels
# 设置好丢失模态
missing_index = get_missing_index(args.views, args.num, args.missing_rate)
train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])
# 均值填充
# 使用CCA方法得到隐藏层
mcca = MCCA(n_components = 1, regs = 0.1)
scores = mcca.fit_transform(X)
scores = np.squeeze(scores).T
scores = torch.Tensor(scores)
# 使用非参数方法测试，训练隐藏层
y_onehot = torch.zeros(y.shape[0], args.classes, device = 'cpu').scatter_(1, y.reshape(y.shape[0], 1), 1)
clf_loss, predicted = classification_loss(y_onehot, y, scores)
accuracy = accuracy_count(predicted, y)
all_num = y.size(0)
accuracy = accuracy/all_num
print("非参方法:{:.4f}".format(accuracy))
# 使用SVM、LR测试
clf1 = LogisticRegression().fit(scores, y)
lr_r2 = clf1.score(scores, y)
clf2 = SVC().fit(scores, y)
svm_r2 = clf2.score(scores, y)
print("lr:{:.4f},       svm:{:.4f}".format(lr_r2,svm_r2))