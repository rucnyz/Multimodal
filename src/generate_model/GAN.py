# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 10:25
# @Author  : nieyuzhou
# @File    : GAN.py
# @Software: PyCharm
import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Classifier(nn.Module):
    def __init__(self, classifier_dims, classes):
        super(Classifier, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(classifier_dims, classes, bias = False))
        self.fc.append(nn.Softplus())  # 经过Softplus算出来的是属于每个类别的概率，因此损失函数是交叉熵

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h


class Discriminator(nn.Module):
    """
    :param classes: Number of classification categories
    :param views: Number of views
    :param classifier_dims: Dimension of the classifier
    :param annealing_epoch: KL divergence annealing epoch during training
    """
    # noinspection PyTypeChecker
    def __init__(self, args):
        super(Discriminator, self).__init__()

        classifier_dims = args.classifier_dims
        self.views = args.views
        self.classes = 2  # real or fake
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        # ModuleList(
        #   (0): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=76, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (1): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=216, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (2): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=64, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (3): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=240, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (4): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=47, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (5): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=6, out_features=2, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        # )

    def forward(self, name, x, missing_index):
        # missing_index: (1600,6) i.e.(train_num, views)
        if name == 'exist':
            x_isreal = dict()
            for v_num in range(self.views):
                x_isreal[v_num] = self.Classifiers[v_num](x[v_num])  # 每个模态预测出的概率
                # 比较两类概率大小 大的为1 小的为0
        elif name == 'miss':
            return


class Generator(nn.Module):  # 相比CPM，删除掉lsd_init，通过encoder产生隐藏层
    def __init__(self, args):
        """
       :param learning_rate:learning rate of network and h
       :param view_num:view number
       :param layer_size:node of each net
       :param lsd_dim:latent space dimensionality
       :param trainLen:training dataset samples
       :param testLen:testing dataset samples
       """
        super(Generator, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = [[150, args.classifier_dims[i]] for i in range(self.view_num)]
        # self.layer_size: [[150, 76], [150, 216], [150, 64], [150, 240], [150, 47], [150, 6]]
        self.lsd_dim = args.lsd_dim  # args.lsd_dim = 128  # lsd: latent space data
        self.lamb = 1
        self.num = args.num
        self.net = nn.ModuleList(self._make_view(v) for v in range(self.view_num))
        # 和TMC类似的体系，每个模态有一个对应的输出，但不完全一样，这是两层全连接层，最后有一个dropout。
        # 而且该网络的输入是lsd_dim，输出是训练数据各模态的特征数
        # 另外这里用ModuleList是类似于list的东西，但不要使用list，因为那样net.parameters()将无法将此识别为网络参数，在优化器传入参数的
        # 时候会有麻烦。ModuleList起到的是注册参数的作用
        # ModuleList(
        #   (0): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=76, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        #   (1): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=216, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        #   (2): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=64, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        #   (3): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=240, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        #   (4): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=47, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        #   (5): Sequential(
        #     (lin0): Linear(in_features=128, out_features=150, bias=True)
        #     (lin1): Linear(in_features=150, out_features=6, bias=True)
        #     (drop1): Dropout(p=0.1, inplace=False)
        #   )
        # )

    # 该前向传播仅用在输入隐藏层，输出训练数据集
    def forward(self, h):
        X_pred = dict()
        for v in range(self.view_num):
            X_pred[v] = self.net[v](h)
        return X_pred

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
        # net1:
        # Sequential(
        #   (lin0): Linear(in_features=128, out_features=150, bias=True)
        #   (lin1): Linear(in_features=150, out_features=76, bias=True)
        #   (drop1): Dropout(p=0.1, inplace=False)
        # )


def train(discriminator, generator, criterion, d_optim, g_optim, epochs, dataloader, print_every = 10):
    iter_count = 0
    for epoch in range(epochs):
        for real_inputs in dataloader:
            real_inputs = real_inputs.to(device)  # 真图片
            fake_inputs = generator(torch.randn(real_inputs.size(0), 100).to(device))  # 生成假图片
            real_labels = torch.ones(real_inputs.size(0)).to(device)  # 真标签
            fake_labels = torch.zeros(real_inputs.size(0)).to(device)  # 假标签

            # 训练判别器
            d_output_real = discriminator(real_inputs).view(-1)  # 鉴别真图片
            d_loss_real = criterion(d_output_real, real_labels)  # 真图片损失
            d_output_fake = discriminator(fake_inputs.detach()).view(-1)  # 鉴别假图片
            d_loss_fake = criterion(d_output_fake, fake_labels)  # 假图片损失
            d_loss = d_loss_fake + d_loss_real  # 计算总损失
            d_optim.zero_grad()  # 判别器梯度清零
            d_loss.backward()  # 反向传播
            d_optim.step()  # 更新鉴别器参数

            # 训练判别器
            fake_inputs = generator(torch.randn(real_inputs.size(0), 100).to(device))  # 生成假图片
            g_output_fake = discriminator(fake_inputs).view(-1)  # 鉴别假图片
            g_loss = criterion(g_output_fake, real_labels)  # 假图片损失
            g_optim.zero_grad()  # 生成器梯度清零
            g_loss.backward()  # 反向传播
            g_optim.step()  # 更新鉴别器参数
            if iter_count % print_every == 0:
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
            iter_count += 1
        torch.save(generator.state_dict(), 'g_' + str(epoch))


if __name__ == '__main__':
    # 测试一下GAN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Discriminator().apply(weights_init).to(device)  # 定义鉴别器
    g = Generator().apply(weights_init).to(device)  # 定义生成器
    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(d.parameters(), lr = 0.0003)
    g_optimizer = torch.optim.Adam(g.parameters(), lr = 0.0003)
