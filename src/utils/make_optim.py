# -*- coding: utf-8 -*-
# @Time    : 2021/8/21 11:20
# @Author  : nieyuzhou
# @File    : make_optim.py
# @Software: PyCharm
import torch


def Adam(net, lr):
    return torch.optim.Adam(net.parameters(), lr = lr)


def MixAdam(net, lr):
    optim = []
    optim.append(torch.optim.Adam([{"params": net.net[v_num].parameters()} for v_num in range(net.view_num)], lr))
    optim.append(torch.optim.Adam([net.lsd_train], lr))
    optim.append(torch.optim.Adam([net.lsd_valid], lr))
    return optim


def GAN_Adam(net, lr):  # 都是更改参数
    optim = dict()
    optim.update({"encoder": torch.optim.Adam(net.encoder.parameters(), lr)})
    optim.update({"decoder": torch.optim.Adam(net.decoder.parameters(), lr)})
    optim.update({"discriminator": torch.optim.Adam(net.discriminator.parameters(), lr)})
    return optim
