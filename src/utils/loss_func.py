# -*- coding: utf-8 -*-
# @Time    : 2021/8/16 22:45
# @Author  : nieyuzhou
# @File    : loss_func.py
# @Software: PyCharm
# loss function
import torch
from torch.nn.functional import one_hot
from torch import nn


def KL(alpha, c):
    beta = torch.ones((1, c))
    # if torch.cuda.is_available():
    #     beta = beta.cuda()
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
    label = one_hot(p, num_classes = c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim = 1, keepdim = True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return A + B


def mse_loss(p, alpha, c, global_step, annealing_step = 1):
    S = torch.sum(alpha, dim = 1, keepdim = True)
    E = alpha - 1
    m = alpha / S
    label = one_hot(p, num_classes = c)
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
        self.weight = args.weight

    def forward(self, predicted, y, global_step):
        loss = 0
        for v_num in range(self.views + 1):
            loss += ce_loss(y, predicted[v_num], self.classes, global_step, self.lambda_epochs)
        class_weight = y.type(torch.float32).apply_(lambda a: self.weight[int(a)])

        loss = torch.mean(loss.squeeze() * class_weight)  # batch_size个sample的loss均值
        return loss


# 把隐藏层通过一些计算，算出来预测结果，然后计算预测和真实之间的误差
# 难点在于他不是常规的去前向传播，而是用了聚类的思路，使得相同标签的元素在特征空间上越来越近，不同标签的元素越来越远
def classification_loss(label_onehot, y, lsd_temp, weight, device):
    # lsd_temp 隐藏层数据(N,lsd_dim)  # xavier_init(int(self.num * 4 / 5), self.lsd_dim).requires_grad_(True)
    # 一个聚类的思路
    train_matrix = torch.mm(lsd_temp, lsd_temp.T)  # (1600,128)*(128,1600) = (1600,1600)  含负数
    train_E = torch.eye(train_matrix.shape[0], train_matrix.shape[1], device = device)  # (1600,1600)单位矩阵

    train_matrix = train_matrix - train_matrix * train_E  # 去掉对角线元素
    # 相似度：这个(N,N)的矩阵的(i,j)位置的元素，代表着第i个样本和第j个样本的点积(第i个数据是指一个lsd_dim长度的向量)，我们记这个点积结果为 相似度
    label_num = label_onehot.sum(0, keepdim = True)
    # (1,10) 每个类别的数据量
    # [[164., 170., 156., 158., 163., 164., 153., 157., 162., 153.]]
    # 把这个矩阵和label_onehot相乘，得到的矩阵(N,classes)，第(i,j)位置的元素，代表着第i个样本和所有属于第j类的样本的 相似度之和
    predicted_full_values = torch.mm(train_matrix, label_onehot) / label_num  # (1600,10) 有正有负 越大相似度越高 类比点积
    # 因此，找到最大的那个类，也就是找到这个样本和其中样本相似度最大的那个类，我们就可以预测该样本属于这个类

    sum = torch.sum(predicted_full_values, dim=1)
    prob = predicted_full_values / torch.stack([sum] * (y.max()+1), dim = 1)  # pos的概率：通过比例计算

    predicted = torch.max(predicted_full_values, dim = 1)[1]  # 每个sample属于的类
    predicted_max_value = torch.max(predicted_full_values, dim = 1, keepdim = False)[0]  # 每个sample预测的属于的类的相似度（最大）
    predicted = predicted.reshape([predicted.shape[0], 1])  # (1600,1)
    predicted_y_value = predicted_full_values * label_onehot  # (1600,10)*(1600,10)=(1600,10)每个元素对应相乘，不是矩阵乘法
    predicted_y = predicted_y_value.sum(axis = 1)  # 每个sample真实的属于的类的相似度
    predicted_max_value = predicted_max_value.reshape([predicted_max_value.shape[0], 1])
    predicted_y = predicted_y.reshape([predicted_y.shape[0], 1])
    # 而在最后，theta(N,1)代表着每个样本是否被正确预测，正确预测为0，不正确为1
    # 那么这个loss计算的意思是，在之前那个(1600,10)里找每个样本最大的类(也就是被预测的类)并得到其值(注意是得到值而不是位置)，
    # 减去该样本真实的类的值，加上theta
    # 那么会有两种情况：分类正确，两者相减就是0，且此时theta也为0，不计入loss当中；
    # 分类错误，最大减真实肯定大于0，通过梯度下降将会缩小这一差距。同时结果加上了theta的和，也就是被错误分类的总数，梯度下降也将降低这一数值
    # 也就是梯度下降在这里有两个目标同时作为目标，可以加快速度
    # relu在这里没什么意义，因为所有值都肯定大于等于0
    # theta + predicted_max_value和predicted_y有梯度，theta是逻辑判断无法计算梯度，因此有无theta对梯度计算没有影响
    # 增加样本权重
    class_weight = y.type(torch.float32).apply_(lambda a: weight[int(a)])
    loss = (predicted_max_value - predicted_y) * class_weight
    return loss.mean(), predicted.squeeze(1), prob


# 就是计算预测的训练数据和真实训练数据之间的差异，求的是误差平方和，同时用到的missing_index起到了只计算未缺失数据误差的作用
# (因为在矩阵运算时缺失索引为0，乘积后这一项就0了，sum后就没算它)
# 其实这个也可以和classfication_loss一起放到损失函数那个文件里，但忘了
def reconstruction_loss(view_num, x_pred, x, missing_index):
    loss = 0
    all_num = 0
    for num in range(view_num):
        # 注意这里的改动会使得其余模型均无法使用
        loss += (torch.pow((x_pred[num] - x[num]), 2.0) * missing_index[:, [num]]).sum()
        all_num += x[num].shape[-1] * x[num].shape[0]
    return loss / all_num


class MyBCELoss(nn.Module):
    def __init__(self, args):
        super(MyBCELoss, self).__init__()
        self.device = args.device

    def forward(self, name, output, missing_index):
        loss_func = nn.BCELoss().to(self.device)  # 对一个batch里面的数据做二元交叉熵并且求平均
        # output：6个模态，每个模态为（1600，2）向量
        # missing_index: (1600,6) i.e.(train_num, views)
        # y = torch.ones(missing_index.shape[0])  # (1600,1)
        real = dict()
        fake = dict()
        for v in range(len(output)):
            real[v] = output[v][:, 0]  # 取real的概率值
            fake[v] = output[v][:, 1]  # 取fake的概率值

        # 接下来删除不需要的概率值，如果是正样本则删除missing_index中为0的值，如果是生成样本则删除missing_index中为1的值
        real_ = dict()
        fake_ = dict()
        for v in range(len(output)):
            real_[v] = real[v][missing_index[:, v] == 1]
            fake_[v] = fake[v][missing_index[:, v] == 0]
        loss = 0
        if name == 'exist':
            for v in range(len(output)):
                loss += loss_func(real_[v], torch.ones(len(real_[v]), device = self.device))
        elif name == 'miss':
            for v in range(len(output)):
                loss += loss_func(fake_[v], torch.ones(len(fake_[v]), device = self.device))
        elif name == 'decoder':  # 尽可能让生成数据与real接近，是fake的概率接近0
            for v in range(len(output)):
                loss += loss_func(fake_[v], torch.zeros(len(fake_[v]), device = self.device))
        loss = loss.nan_to_num()
        return loss
