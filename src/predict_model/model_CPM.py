# -*- coding: utf-8 -*-
# @Time    : 2021/8/20 0:26
# @Author  : nieyuzhou
# @File    : model_CPM.py
# @Software: PyCharm

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

    def calculate(self, h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[v_num] = self.net[v_num](h)
        return h_views
