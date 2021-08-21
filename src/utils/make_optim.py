# -*- coding: utf-8 -*-
# @Time    : 2021/8/21 11:20
# @Author  : nieyuzhou
# @File    : make_optim.py
# @Software: PyCharm
import torch


def Adam(net, lr):
    return torch.optim.Adam(net.parameters(), lr = lr, weight_decay = 1e-5)


def MixAdam(net, lr):
    optim = []
    optim.append(torch.optim.Adam([{"params": net.net[v_num].parameters()} for v_num in range(net.view_num)], lr))
    optim.append(torch.optim.Adam([net.lsd_train], lr))
    return optim
