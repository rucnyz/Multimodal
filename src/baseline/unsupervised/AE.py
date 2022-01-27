# -*- coding: utf-8 -*-
# @Time    : 2022/1/25 21:21
# @Author  : HCY
# @File    : AE.py
# @Software: PyCharm

from generate_model.GAN import *  # Generator
from predict_model.CPM_GAN import *  # Encoder
from utils.loss_func import *
from utils.preprocess import *
from torch.utils.data import DataLoader
import argparse
import os
from torch import nn
from utils.make_optim import Adam
from utils.pred_func import *
from utils.preprocess import get_missing_index, missing_data_process
from torchmetrics import Accuracy
from dataset.UCI_dataset import UCI_Dataset
from dataset.UKB_dataset import UKB_Dataset
from dataset.multi_view_dataset import Multiview_Dataset
from dataset.UKB_ad_dataset import UKB_AD_Dataset


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = [[150, args.classifier_dims[i]] for i in range(self.view_num)]
        # self.layer_size: [[150, 76], [150, 216], [150, 64], [150, 240], [150, 47], [150, 6]]
        self.lsd_dim = args.lsd_dim  # args.lsd_dim = 128  # lsd: latent space data
        self.lamb = 1
        self.num = args.num
        # 模型初始化
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

    def forward(self, x, missing_index):
        lsd_train = self.encoder(x, missing_index)  # （1600，128）
        x_pred = self.decoder(lsd_train)
        return x_pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI', 'UKB',
                                   'UKB_AD'],
                        default = 'UCI')
    parser.add_argument('--missing_rate', type = float, default = 0,
                        help = 'view missing rate [default: 0]')
    parser.add_argument('--seed', type = int, default = 123)
    parser.add_argument('--lr', type = float, default = 0.0008)
    argument = parser.parse_args()
    return argument


if __name__ == '__main__':
    if os.getcwd().endswith("src"):
        os.chdir("../")
    args = parse_args()
    args.dataloader = "UKB_Dataset"
    args.lsd_dim = 128
    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = "cpu"
    # 设置seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Dataset
    # eval: 返回传入字符串的表达式的结果
    train_dset = eval(args.dataloader)('train', args)  # args.dataloader类的构造函数
    eval_dset = eval(args.dataloader)('valid', args)

    # 设置好丢失模态
    missing_index = get_missing_index(args.views, args.num, args.missing_rate)
    train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
    eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])

    # DataLoader: DataSet的打包
    train_loader = DataLoader(train_dset, batch_size = int(args.num * 4 / 5), num_workers = args.num_workers,
                              shuffle = True,
                              pin_memory = False)
    eval_loader = DataLoader(eval_dset, batch_size = args.num - int(args.num * 4 / 5), num_workers = args.num_workers,
                             pin_memory = False)

    epochs = 100
    # Net
    net = AE(args)
    net.to(args.device)
    # 优化器
    optim = Adam(net, args.lr)
    best_eval_accuracy = 0  # 最佳验证准确率
    for epoch in range(epochs):
        net.train(True)

        best_train_accuracy = 0
        rec_loss_sum = 0
        train_accuracy = 0
        all_num = 0
        # train
        accuracy = Accuracy()
        for step, (idx, X, y, missing_index) in enumerate(train_loader):
            # 使用cuda或者cpu设备
            for i in range(args.views):
                X[i] = X[i].to(args.device)
            y = y.to(args.device)
            missing_index = missing_index.to(args.device)
            # 产生one-hot编码的标签
            y_onehot = torch.zeros(y.shape[0], args.classes, device = args.device).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            # 重建原数据,考虑缺失模态情况
            x_pred = net(X, missing_index)
            lsd_train = net.encoder(X, missing_index)
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
            optim.zero_grad()
            rec_loss.backward()
            optim.step()
            rec_loss_sum += rec_loss.item()
            # 计算准确率
            _, predicted = classification_loss(y_onehot, y, net.encoder(X, missing_index), args.weight, args.device)
            train_accuracy = accuracy(predicted, y)
        train_accuracy = accuracy.compute().data
        print("[Epoch %2d] reconstruction loss: %.4f accuracy: %.4f" % (epoch + 1, rec_loss_sum, train_accuracy))
        # valid
        all_num = 0
        val_rec_loss_sum = 0
        valid_accuracy = 0
        net.train(False)
        accuracy = Accuracy()
        with torch.no_grad():
            for step, (idx, X, y, missing_index) in enumerate(eval_loader):
                # 使用cuda或者cpu设备
                for i in range(args.views):
                    X[i] = X[i].to(args.device)
                y = y.to(args.device)
                missing_index = missing_index.to(args.device)
                # 产生one-hot编码的标签
                x_pred = net(X, missing_index)
                val_rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
                val_rec_loss_sum += val_rec_loss.item()
                predicted = ave(lsd_train, net.encoder(X, missing_index), y_onehot)
                valid_accuracy = accuracy(predicted, y)
            valid_accuracy = accuracy.compute().data
            print("valid reconstruction loss: %.4f valid accuracy: %.4f" % (val_rec_loss_sum, valid_accuracy))
            if valid_accuracy >= best_eval_accuracy:
                best_eval_accuracy = valid_accuracy
    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))
