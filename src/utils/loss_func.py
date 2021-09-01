# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 22:45
# @Author  : nieyuzhou
# @File    : loss_func.py
# @Software: PyCharm
# loss function
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import relu


def KL(alpha, c):
    beta = torch.ones((1, c))
    if torch.cuda.is_available():
        beta = beta.cuda()
    S_alpha = torch.sum(alpha, dim = 1, keepdim = True)
    S_beta = torch.sum(beta, dim = 1, keepdim = True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim = 1, keepdim = True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim = 1, keepdim = True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim = 1, keepdim = True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim = 1, keepdim = True)
    E = alpha - 1
    label = F.one_hot(p, num_classes = c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim = 1, keepdim = True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)


def mse_loss(p, alpha, c, global_step, annealing_step = 1):
    S = torch.sum(alpha, dim = 1, keepdim = True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes = c)
    A = torch.sum((label - m) ** 2, dim = 1, keepdim = True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim = 1, keepdim = True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class AdjustedCrossEntropyLoss(nn.Module):
    def __init__(self, args):
        super(AdjustedCrossEntropyLoss, self).__init__()
        self.lambda_epochs = args.lambda_epochs
        self.views = args.views
        self.classes = args.classes

    def forward(self, predicted, y, global_step):
        loss = 0
        for v_num in range(self.views + 1):
            loss += ce_loss(y, predicted[v_num], self.classes, global_step, self.lambda_epochs)
        loss = torch.mean(loss)
        return loss

# 这个计算的可恶心了。。。但现在没时间写了
# 反正意思就是把隐藏层通过一些计算，算出来预测结果，然后计算预测和真实之间的误差
# 难点在于他不是常规的去前向传播，而是用了聚类的思路，使得相同标签的元素在特征空间上越来越近，不同标签的元素越来越远
# 没时间写了先这样吧
def classification_loss(label_onehot, y, lsd_temp):
    # lsd_temp 隐藏层数据(1600,150)
    # 一个聚类的思路
    train_matrix = torch.mm(lsd_temp, lsd_temp.T)  # (1600,1600)
    train_E = torch.eye(train_matrix.shape[0], train_matrix.shape[1])  # 单位矩阵
    train_matrix = train_matrix - train_matrix * train_E  # 去掉对角线元素(1600,1600)

    label_num = label_onehot.sum(0, keepdim = True)
    predicted_full_values = torch.mm(train_matrix, label_onehot) / label_num  # (1600,10)

    predicted = torch.max(predicted_full_values, dim = 1)[1]
    predicted = predicted.type(torch.IntTensor)
    predicted_max_value = torch.max(predicted_full_values, dim = 1, keepdim = False)[0]
    predicted = predicted.reshape([predicted.shape[0], 1])
    theta = torch.ne(y.reshape([y.shape[0], 1]), predicted).type(torch.FloatTensor)
    predicted_y_value = predicted_full_values * label_onehot
    predicted_y = predicted_y_value.sum(axis = 1)
    predicted_max_value = predicted_max_value.reshape([predicted_max_value.shape[0], 1])
    predicted_y = predicted_y.reshape([predicted_y.shape[0], 1])
    return (relu(theta + predicted_max_value - predicted_y)).sum(), predicted.squeeze(1)
