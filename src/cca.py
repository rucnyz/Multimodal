# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 0:28
# @Author  : nieyuzhou
# @File    : cca.py
# @Software: PyCharm
import argparse
import os
import math
import time

from mvlearn.embed import MCCA
from torch import nn

from dataset.UCI_dataset import UCI_Dataset
from utils.preprocess import *
from torch.utils.data import DataLoader


if os.getcwd().endswith("src"):
    os.chdir("../")
parser = argparse.ArgumentParser()

parser.add_argument('--missing_rate', type = float, default = 0,
                    help = 'view missing rate [default: 0]')
parser.add_argument("--mode", default = 'client')
parser.add_argument("--port", default = 52162)
parser.add_argument('--batch_size', type = int, default = 64)  # 200
parser.add_argument('--num_workers', type = int, default = 0)
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
args.train_batch_size = int(args.num * 4 / 5)
args.valid_batch_size = args.num - int(args.num * 4 / 5)
args.device = torch.device("cpu")
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

###################
train_loader = DataLoader(train_dset, args.train_batch_size, num_workers = args.num_workers, shuffle = True,
                              pin_memory = False)
eval_loader = DataLoader(eval_dset, args.valid_batch_size, num_workers = args.num_workers, pin_memory = False)

layer_size = [[150, args.classifier_dims[i]] for i in range(args.views)]


def _make_view(v):
    dims_net = layer_size[v]
    net1 = nn.Sequential()
    w = torch.nn.Linear(s_train.shape[1], dims_net[0])  # ######### 90
    nn.init.xavier_normal_(w.weight)  # xavier_normal 初始化
    nn.init.constant_(w.bias, 0.0)  # 初始化w偏差为常数0
    net1.add_module('lin' + str(0), w)
    for num in range(1, len(dims_net)):  # range(1,2) 只运行一次
        w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
        nn.init.xavier_normal_(w.weight)
        nn.init.constant_(w.bias, 0.0)
        net1.add_module('lin' + str(num), w)
        net1.add_module('drop' + str(num), torch.nn.Dropout(p=0.1))
        # nn.dropout: 防止或减轻过拟合, randomly zeroes some of the elements of the input tensor with probability p
    return net1
    # net1:
    # Sequential(
    #   (lin0): Linear(in_features=90, out_features=150, bias=True)
    #   (lin1): Linear(in_features=150, out_features=76, bias=True)
    #   (drop1): Dropout(p=0.1, inplace=False)
    # )


net = nn.ModuleList(_make_view(v) for v in range(args.views))
net.to(args.device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr = 0.1)
# 开始运行

def train_CPM(args, epoch, net, optim, train_images, train_loader, time_start):
    train_accuracy = 0
    all_num = 0
    clf_loss = 0
    rec_loss = 0
    id = None
    label_onehot = None
    # train_batch等于训练数据集大小，循环只会跑一次
    for step, (idx, X, y, missing_index) in enumerate(train_loader):
        # id用于回传(注意每次idx都是一样的，因为设定好了seed)
        # y: 所有数据的类别标签
        for i in range(args.views):
            X[i] = X[i].to(args.device)
        y = y.to(args.device)
        id = idx
        missing_index.to(args.device)
        # 用训练集标签生成one_hot标签（即每个数据标签变成(1,10)向量，属于那个类别该类别为1，其他为0）
        label_onehot = torch.zeros(args.train_batch_size, args.classes, device = args.device).scatter_(1, y.reshape(
            y.shape[0], 1), 1)

        # 首先进行重建，也就是decoder的过程，具体见reconstruction_loss函数。optim[0]更新的是模型参数
        # reconstruction_loss体现的是经过autoencoder后
        # train the network to minimize reconstruction loss
        for i in range(5):
            # x_pred得到的是训练数据，注意我们这里要做的是尽可能让隐藏层lsd_train通过网络变得更像原训练数据集X
            x_pred = net(s_train[idx])  # 输出是训练数据各模态的特征数
            # 所有不缺失数据的误差平方和--> 使得经过net生成的数据与原来数据接近
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index[idx])
            optim.zero_grad()
            rec_loss.backward(retain_graph = True)
            optim.step()
        # 最后算一次，进行输出
        x_pred = net(s_train[idx])
        clf_loss, predicted = classification_loss(label_onehot, y, net.lsd_train[idx])
        rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index[idx])
        print(
            "\r[Epoch %2d][Step %4d/%4d] Reconstruction Loss: %.4f, Classification Loss = %.4f, Lr: %.2e, %4d m remaining"
            % (epoch + 1, step + 1, train_images, rec_loss, clf_loss,
               *[group['lr'] for group in optim[1].param_groups],
               ((time.time() - time_start) / (step + 1)) * ((len(train_loader.dataset) / args.batch_size) - step) / 60),
            end = '   ')
        train_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)
    train_accuracy = 100 * train_accuracy / all_num

    return clf_loss + rec_loss, train_accuracy, label_onehot, id


def evaluate_CPM(args, net, optim, valid_loader, label_onehot, id):
    valid_accuracy = 0
    all_num = 0
    for step, (idx, X, y, missing_index) in enumerate(valid_loader):
        for i in range(args.views):
            X[i] = X[i].to(args.device)
        y = y.to(args.device)
        missing_index = missing_index.to(args.device)
        with torch.no_grad():
            x_pred = net(s_valid)  # #################?
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
            predicted = ave(net.lsd_train[id], net.lsd_valid, label_onehot)
        # 在eval又不去反向传播损失，完全没必要用classification_loss这个函数, 下面的valid_accuracy直接计算是否分类正确
        print("Reconstruction Loss = {:.4f}".format(rec_loss))
        predicted = predicted.reshape(len(predicted), 1)
        y = y.reshape(len(y), 1)
        valid_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)
    valid_accuracy = 100 * valid_accuracy / all_num
    return valid_accuracy


best_eval_accuracy = 0
decay_count = 0
train_images = math.ceil(len(train_loader.dataset) / args.train_batch_size)
# train for each epoch
for epoch in range(0, 200):
    time_start = time.time()
    loss_sum, train_accuracy, label_onehot, id = train_CPM(args, epoch, net, optim, train_images,
                                                           train_loader, time_start)
    time_end = time.time()
    elapse_time = time_end - time_start
    print('Finished in {:.4f}s'.format(elapse_time))
    epoch_finish = epoch + 1
    print("Train Accuracy :" + str(train_accuracy))
    # Eval
    print('Evaluation...    decay times: {}'.format(decay_count))
    valid_accuracy = evaluate_CPM(args, net, optim, eval_loader, label_onehot, id)
    print('Valid Accuracy :' + str(valid_accuracy) + '\n')
    if valid_accuracy >= best_eval_accuracy:
        best_eval_accuracy = valid_accuracy

print("---------------------------------------------")
print("Best evaluate accuracy:{}".format(best_eval_accuracy))

# # 跑一个简单的模型
# net = nn.Sequential(
#     nn.Linear(scores_train.shape[0] * scores_train.shape[2], 64),
#     nn.Dropout(0.4),
#     nn.Linear(64, y.max() - y.min() + 1),
# )
# loss_fn = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(net.parameters(), lr = 0.1)
# # 开始运行
# best_valid_acc = 0
# for i in range(100):
#     predicted = net(s_train)
#     loss = loss_fn(predicted, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     # 计算准确率
#     y_predicted = torch.argmax(predicted, dim = 1)
#     train_acc = ((y_predicted == y).sum() / y.shape[0] * 100).item()
#     # 验证部分
#     with torch.no_grad():
#         predicted_valid = net(s_valid)
#         y_predicted_valid = torch.argmax(predicted_valid, dim = 1)
#         valid_acc = ((y_predicted_valid == y_valid).sum() / y_valid.shape[0] * 100).item()
#     if best_valid_acc < valid_acc:
#         best_valid_acc = valid_acc
#     print("第{}个epoch：\n训练准确率:{:.4f}     验证准确率:{:.4f}".format(i, train_acc, valid_acc))
# print("------------------------------\n最高验证集准确率:{:.4f}".format(best_valid_acc))

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
