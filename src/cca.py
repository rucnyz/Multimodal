# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 0:28
# @Author  : nieyuzhou
# @File    : cca.py
# @Software: PyCharm
import argparse
import os

from mvlearn.embed import MCCA
from torch import nn

from dataset.UCI_dataset import UCI_Dataset
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
y = train_dset.full_labels
y_valid = eval_dset.full_labels
# 设置好丢失模态
missing_index = get_missing_index(args.views, args.num, args.missing_rate)
train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])
# 均值填充

# 使用CCA方法得到隐藏层
components = 15
mcca = MCCA(n_components = components, regs = 0.1)
scores_train = mcca.fit_transform(X)
s_train = np.empty((scores_train.shape[1], scores_train.shape[0] * scores_train.shape[2]))
for i in range(scores_train.shape[0]):
    s_train[:, i * components:(i + 1) * components] = scores_train[i]
# 验证集隐藏层
scores_valid = mcca.fit_transform(X_valid)
s_valid = np.empty((scores_valid.shape[1], scores_valid.shape[0] * scores_valid.shape[2]))
for i in range(scores_train.shape[0]):
    s_valid[:, i * components:(i + 1) * components] = scores_valid[i]
# 设置为tensor
s_train = torch.from_numpy(s_train).float()
s_valid = torch.from_numpy(s_valid).float()
# 跑一个简单的模型
net = nn.Sequential(
    nn.Linear(scores_train.shape[0] * scores_train.shape[2], 64),
    nn.Dropout(0.4),
    nn.Linear(64, y.max() - y.min() + 1),
)
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr = 0.1)
# 开始运行
best_valid_acc = 0
for i in range(100):
    predicted = net(s_train)
    loss = loss_fn(predicted, y)
    optim.zero_grad()
    loss.backward()
    optim.step()
    # 计算准确率
    y_predicted = torch.argmax(predicted, dim = 1)
    train_acc = ((y_predicted == y).sum() / y.shape[0] * 100).item()
    # 验证部分
    with torch.no_grad():
        predicted_valid = net(s_valid)
        y_predicted_valid = torch.argmax(predicted_valid, dim = 1)
        valid_acc = ((y_predicted_valid == y_valid).sum() / y_valid.shape[0] * 100).item()
    if best_valid_acc < valid_acc:
        best_valid_acc = valid_acc
    print("第{}个epoch：\n训练准确率:{:.4f}     验证准确率:{:.4f}".format(i, train_acc, valid_acc))
print("------------------------------\n最高验证集准确率:{:.4f}".format(best_valid_acc))
# 使用SVM测试
# clf = SVC().fit(s, y)
# y_predict = clf.predict(s)
# svm_acc = accuracy_score(y, y_predict)
# print("训练集:\n svm:{:.4f}".format(svm_acc))
# -------------------------------------
# 验证集试试

# y_predict = clf.predict(s_valid)
# svm_acc = accuracy_score(y_valid, y_predict)
# print("验证集:\n svm:{:.4f}".format(svm_acc))
