# -*- coding: utf-8 -*-
# @Time    : 2021/8/13 11:02
# @Author  : nieyuzhou
# @File    : model_TMC.py
# @Software: PyCharm

from utils.loss_func import *


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


class TMC(nn.Module):

    def __init__(self, args):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()

        classifier_dims = args.classifier_dims
        self.views = args.views
        self.classes = args.classes
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])
        # ModuleList(
        #   (0): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=76, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (1): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=216, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (2): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=64, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (3): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=240, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (4): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=47, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        #   (5): Classifier(
        #     (fc): ModuleList(
        #       (0): Linear(in_features=6, out_features=10, bias=False)
        #       (1): Softplus(beta=1, threshold=20)
        #     )
        #   )
        # )

    # DS组合规则
    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        # 首先定义两个组合
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim = 1, keepdim = True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.classes / S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim = (1, 2), out = None)
            bb_diag = torch.diagonal(bb, dim1 = -2, dim2 = -1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^all
            b_all = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # calculate u^all
            u_all = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_all
            # calculate new e_k
            e_a = torch.mul(b_all, S_a.expand(b_all.shape))
            alpha_all = e_a + 1
            return alpha_all

        # 拓展到多个组合
        for v in range(len(alpha) - 1):
            if v == 0:
                alpha_all = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_all = DS_Combin_two(alpha_all, alpha[v + 1])
        return alpha_all

    # 得到证据evidence
    def infer(self, input_x):
        """
        :param input_x: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input_x[v_num])  # 每个模态预测出的概率
        return evidence

    def forward(self, X):
        # step one: 得到evidence: 每个模态的预测结果
        evidence = self.infer(X)
        alpha = dict()
        for v_num in range(self.views):
            # step two
            alpha[v_num] = evidence[v_num] + 1
        # step three
        alpha_all = self.DS_Combin(alpha)
        evidence_all = alpha_all - 1
        # 在evidence字典最后加上最终的预测结果
        evidence[self.views] = evidence_all
        return evidence
