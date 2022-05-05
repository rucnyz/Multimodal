# -*- coding: utf-8 -*-
# @Time    : 2021/9/18 23:02
# @Author  : nieyuzhou
# @File    : CPM_GAN.py
# @Software: PyCharm
from torch.nn import MultiheadAttention

from generate_model.GAN import *
from utils.loss_func import *
from utils.preprocess import *


# VAE model
class VAE(nn.Module):
    def __init__(self, x_dim = 784, h_dim = 400, z_dim = 20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc_mu = nn.Linear(h_dim, z_dim)  # 均值 向量
        self.fc_sigma = nn.Linear(h_dim, z_dim)  # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, x_dim)

    # 编码过程
    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_sigma(h)

    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = torch.relu(self.fc4(z))
        return torch.sigmoid(self.fc5(h))

    # 整个前向传播过程：编码->解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var


class Encoder(nn.Module):
    def __init__(self, args, zdim = 20):
        super(Encoder, self).__init__()
        self.view_num = args.views
        self.device = args.device
        self.views = args.views
        self.Classifiers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(args.classifier_dims[i], args.lsd_dim),
                nn.ReLU(),
                nn.Linear(args.lsd_dim, args.lsd_dim),
                # nn.Dropout(p = 0.2)  # 解决梯度爆炸问题
            ) for i in range(self.view_num)])

        self.fc_sigma = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(args.classifier_dims[i], args.lsd_dim),
                nn.ReLU(),
                nn.Linear(args.lsd_dim, args.lsd_dim),
            ) for i in range(self.view_num)])  # 保准方差 向量

        self.Q = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.K = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.V = nn.Linear(args.lsd_dim, args.lsd_dim, bias = False)
        self.attn = MultiheadAttention(args.lsd_dim, 2, batch_first = True)

    def forward(self, X, missing_index):
        attention = 0
        kl_div = 0
        for i in range(self.view_num):
            mu = self.Classifiers[i](X[i])
            # log_var = self.fc_sigma[i](X[i])
            log_var = torch.zeros_like(self.fc_sigma[i](X[i]))
            z = self.reparameterize(mu, log_var)
            attention += z * missing_index[:, [i]]
            kl_div += - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return attention, kl_div
        for i in range(self.views):
            X[i] = X[i].to(self.device)
        missing_index = missing_index.to(self.device)
        missing_index = missing_index.to(torch.float32)
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

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(std)
        return mu + eps * std


class CPM_GAN(nn.Module):
    def __init__(self, args):
        super(CPM_GAN, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = [[150, args.classifier_dims[i]] for i in range(self.view_num)]
        # self.layer_size: [[150, 76], [150, 216], [150, 64], [150, 240], [150, 47], [150, 6]]
        self.lsd_dim = args.lsd_dim  # args.lsd_dim = 128  # lsd: latent space data
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
