# -*- coding: utf-8 -*-
# @Time    : 2021/8/20 0:26
# @Author  : nieyuzhou
# @File    : model_CPM.py
# @Software: PyCharm

from utils.loss_func import *
from utils.preprocess import *


class CPM(nn.Module):
    def __init__(self, args):
        """
       :param learning_rate:learning rate of network and h
       :param view_num:view number
       :param layer_size:node of each net
       :param lsd_dim:latent space dimensionality
       :param trainLen:training dataset samples
       :param testLen:testing dataset samples
       """
        super(CPM, self).__init__()
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

        self.lsd_train = self.lsd_init('train')
        self.lsd_valid = self.lsd_init('valid')
        # 初始化隐藏层，使用的方法是均匀分布的Xavier初始化，具体可参见我写的PDF以及网络资料
        # (这也是原本就有的，其实pytorch有已经实现的函数我也不知道他为什么要自己写一遍)
        # 所以为啥batch调成了一整个训练集大小呢，因为如果多次batch的话就需要初始化多个lsd_train(每个都得单独训练因为不同batch本身概率分布
        # 也是不同的，更不用说有的非整数训练集最后一个batch大小还和前面不一样，而且这样分成多个矩阵运算也会使得速度更慢，大矩阵运算相比于多个
        # 小矩阵运算绝对会快很多很多很多，因为可以并行)

        self.lsd = torch.cat((self.lsd_train, self.lsd_valid), dim = 0)  # torch.cat是将两个张量（tensor）拼接在一起

    # 该前向传播仅用在输入隐藏层，输出训练数据集
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
            """
            def xavier_init(fan_in, fan_out, constant = 1):
                low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
                high = constant * np.sqrt(6.0 / (fan_in + fan_out))
                a = np.random.uniform(low, high, (fan_in, fan_out))
                a = a.astype('float32')
                a = torch.from_numpy(a)
                return a
            """
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
        # net1:
        # Sequential(
        #   (lin0): Linear(in_features=128, out_features=150, bias=True)
        #   (lin1): Linear(in_features=150, out_features=76, bias=True)
        #   (drop1): Dropout(p=0.1, inplace=False)
        # )

    # 这函数没用着，是原本的前向传播函数
    def calculate(self, h):
        h_views = dict()
        for v_num in range(self.view_num):
            h_views[v_num] = self.net[v_num](h)
        return h_views
