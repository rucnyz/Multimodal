# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 11:02
# @Author  : nieyuzhou
# @File    : my_model.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class My_Model(nn.Module):
    def __init__(self, args, **argsd):
        super(My_Model, self).__init__()
        self.pic = nn.Conv2d()
    def forward(self, a, b, c, d, e, f):
        return
