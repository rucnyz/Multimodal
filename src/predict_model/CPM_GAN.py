# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 23:02
# @Author  : nieyuzhou
# @File    : CPM_GAN.py
# @Software: PyCharm
from generate_model.GAN import *
from utils.loss_func import *
from utils.preprocess import *


class CPM_GAN(nn.Module):
    def __init__(self, args):
        super(CPM_GAN, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = [[150, args.classifier_dims[i]] for i in range(self.view_num)]
        # self.layer_size: [[150, 76], [150, 216], [150, 64], [150, 240], [150, 47], [150, 6]]
        self.lsd_dim = args.lsd_dim  # args.lsd_dim = 128  # lsd: latent space data
        self.lamb = 1
        self.num = args.num
        # 模型初始化
        self.discriminator = Discriminator()
        self.decoder = Generator()
        self.encoder = nn.ModuleList(self._make_view(v) for v in range(self.view_num))

    def forward(self, h):
        X_pred = dict()
        for v in range(self.view_num):
            X_pred[v] = self.net[v](h)
        return X_pred

    def lsd_init(self, a):
        h = 0
        if a == 'train':
            h = xavier_init(int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)  # 参数随机初始化(均匀分布)
            # requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度
            # self.lsd_dim控制了输入维度为128
        elif a == 'valid':
            h = xavier_init(self.num - int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)
        return h

    def _make_view(self, v):
        dims_net = self.layer_size[v]
        net1 = nn.Sequential()
        w = torch.nn.Linear(self.lsd_dim, dims_net[0])
        nn.init.xavier_normal_(w.weight)  # xavier_normal 初始化
        nn.init.constant_(w.bias, 0.0)  # 初始化w偏差为常数0
        net1.add_module('lin' + str(0), w)
        for num in range(1, len(dims_net)):  # range(1,2) 只运行一次
            w = torch.nn.Linear(dims_net[num - 1], dims_net[num])
            nn.init.xavier_normal_(w.weight)
            nn.init.constant_(w.bias, 0.0)
            net1.add_module('lin' + str(num), w)
            net1.add_module('drop' + str(num), torch.nn.Dropout(p = 0.1))
            # nn.dropout: 防止或减轻过拟合, randomly zeroes some of the elements of the input tensor with probability p
        return net1
