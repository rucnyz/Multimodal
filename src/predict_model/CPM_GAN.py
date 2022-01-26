# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 23:02
# @Author  : nieyuzhou
# @File    : CPM_GAN.py
# @Software: PyCharm
from torch.nn import MultiheadAttention

from generate_model.GAN import *
from utils.loss_func import *
from utils.preprocess import *


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.view_num = args.views
        self.device = args.device
        self.views = args.views
        self.Classifiers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(args.classifier_dims[i], 150),
                nn.ReLU(),
                nn.Linear(150, args.lsd_dim),
                nn.Dropout(p = 0.2)  # 解决梯度爆炸问题
            ) for i in range(self.view_num)])
        self.Q = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.K = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.V = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.attn = MultiheadAttention(args.lsd_dim, 2, batch_first = True)

    def forward(self, X, missing_index):
        # attention = 0
        # for i in range(self.view_num):
        #     attention += self.Classifiers[i](X[i]) * missing_index[:, [i]]
        for i in range(self.views):
            X[i] = X[i].to(self.device)
        missing_index = missing_index.to(self.device)
        Q_vector = torch.tensor([]).to(self.device)
        K_vector = torch.tensor([]).to(self.device)
        V_vector = torch.tensor([]).to(self.device)
        for i in range(self.view_num):
            each = self.Classifiers[i](X[i]) * missing_index[:, [i]]
            Q_vector = torch.cat((Q_vector, self.Q(each).unsqueeze(1)), 1)
            K_vector = torch.cat((K_vector, self.K(each).unsqueeze(1)), 1)
            V_vector = torch.cat((V_vector, self.V(each).unsqueeze(1)), 1)
        output, _ = self.attn(Q_vector, K_vector, V_vector, need_weights = False)
        return output.sum(dim = 1)


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
        self.discriminator = Discriminator(args)
        self.decoder = Generator(args)
        self.encoder = Encoder(args)

    def lsd_init(self, a):
        h = 0
        if a == 'train':
            h = xavier_init(int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)  # 参数随机初始化(均匀分布)
            # requires_grad=True 的作用是让 backward 可以追踪这个参数并且计算它的梯度
            # self.lsd_dim控制了输入维度为128
        elif a == 'valid':
            h = xavier_init(self.num - int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)
        return h
