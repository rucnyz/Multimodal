# -*- coding: utf-8 -*-
# @Time    : 2021/8/20 0:26
# @Author  : nieyuzhou
# @File    : model_CPM.py
# @Software: PyCharm
from torch.nn.functional import relu

from utils.loss_func import *
from utils.preprocess import *


class CPM(nn.Module):
    def __init__(self, args):
        super(CPM, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = [[150, args.classifier_dims[i]] for i in range(self.view_num)]
        self.lsd_dim = args.lsd_dim
        self.lamb = 1
        self.num = args.num
        # initialize forward methods
        self.net = nn.ModuleList(self._make_view(v) for v in range(self.view_num))
        self.lsd_train = self.lsd_init('train')
        self.lsd_valid = self.lsd_init('valid')
        self.lsd = torch.cat((self.lsd_train, self.lsd_valid), dim = 0)

    def forward(self, h):
        h_views = dict()
        for v in range(self.view_num):
            h_views[v] = self.net[v](h)
        return h_views

    def lsd_init(self, a):
        h = 0
        if a == 'train':
            h = xavier_init(int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)
        elif a == 'valid':
            h = xavier_init(self.num - int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)
        return h

    def _make_view(self, v):
        dims_net = self.layer_size[v]
        net1 = nn.Sequential()
        w = torch.nn.Linear(self.lsd_dim, dims_net[0])
        nn.init.xavier_normal_(w.weight)
        nn.init.constant_(w.bias, 0.0)
        net1.add_module('lin' + str(0), w)
        for num in range(1, len(dims_net)):
            w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
            nn.init.xavier_normal_(w.weight)
            nn.init.constant_(w.bias, 0.0)
            net1.add_module('lin' + str(num), w)
            net1.add_module('drop' + str(num), torch.nn.Dropout(p = 0.1))
        return net1

    def reconstruction_loss(self, x_pred, x, sn):
        loss = 0
        for num in range(self.view_num):
            loss = loss + (torch.pow((x_pred[num] - x[num]), 2.0) * sn[num]).sum()
        return loss

    def classification_loss(self, label_onehot, y, lsd_temp):
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
        F_h_hn_mean = predicted_y_value.sum(axis = 1)
        predicted_max_value = predicted_max_value.reshape([predicted_max_value.shape[0], 1])
        F_h_hn_mean = F_h_hn_mean.reshape([F_h_hn_mean.shape[0], 1])
        return (relu(theta + predicted_max_value - F_h_hn_mean)).sum(), predicted.squeeze(1)

    def calculate(self, h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[v_num] = self.net[v_num](h)
        return h_views
