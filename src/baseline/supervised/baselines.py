# -*- coding: utf-8 -*-
# @Time    : 2021/12/15 17:27
# @Author  : nieyuzhou
# @File    : baselines.py
# @Software: PyCharm
import torch
# from cca_zoo.deepmodels import DCCA, architectures
from matplotlib import pyplot as plt
from metric_learn import LMNN
from sklearn.manifold import TSNE

import argparse
import os

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from predict_model.net import MultiLayerPerceptron
from utils.make_optim import Adam
from utils.pred_func import *
from utils.preprocess import get_missing_index, missing_data_process
from dataset.UKB_dataset import UKB_Dataset
from dataset.UCI_dataset import UCI_Dataset
from dataset.UKB_ad_dataset import UKB_AD_Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI', 'UKB', 'UKB_AD'],
                        default = 'UCI')
    parser.add_argument('--missing_rate', type = float, default = 0.2,
                        help = 'view missing rate [default: 0]')
    parser.add_argument('--seed', type = int, default = 123)
    argument = parser.parse_args()
    return argument


# 画图 高维数据可视化
def plot_embedding(x, y_pred, y_true):
    x = TSNE(learning_rate = 'auto').fit_transform(x)
    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.scatter(x[:, 0], x[:, 1], c = y_pred)
    plt.title("predict")

    plt.subplot(122)
    plt.scatter(x[:, 0], x[:, 1], c = y_true)
    plt.title("true")
    plt.show()


# 直接连接
def feat_concat(x, y_true, x_valid):
    args.input_size = sum(args.classifier_dims)
    concat_X = torch.tensor([])
    for i in range(args.views):
        concat_X = torch.cat((concat_X, x[i]), dim = 1)

    concat_X_valid = torch.tensor([])
    for i in range(args.views):
        concat_X_valid = torch.cat((concat_X_valid, x_valid[i]), dim = 1)
    return concat_X, concat_X_valid


# lmnn处理
def lmnn_transform(x, y_true, x_valid):
    args.input_size = sum(args.classifier_dims)
    concat_X = torch.tensor([])
    for i in range(args.views):
        concat_X = torch.cat((concat_X, x[i]), dim = 1)
    lmnn = LMNN(k = 5, learn_rate = 1e-4, verbose = True, random_state = 123, convergence_tol=0.1)
    lmnn.fit(concat_X, y_true)
    concat_X_valid = torch.tensor([])
    for i in range(args.views):
        concat_X_valid = torch.cat((concat_X_valid, x_valid[i]), dim = 1)
    return torch.tensor(lmnn.transform(concat_X), dtype = torch.float32), torch.tensor(lmnn.transform(concat_X_valid),
                                                                                       dtype = torch.float32)


if __name__ == '__main__':
    if os.getcwd().endswith("src"):
        os.chdir("../")
    args = parse_args()
    args.dataloader = "UKB_Dataset"
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
    # 均值填充
    train_dset.replace_with_mean()
    eval_dset.replace_with_mean()

    # DataLoader: DataSet的打包
    train_loader = DataLoader(train_dset, batch_size = int(args.num * 4 / 5), num_workers = args.num_workers,
                              shuffle = True,
                              pin_memory = False)
    eval_loader = DataLoader(eval_dset, batch_size = args.num - int(args.num * 4 / 5), num_workers = args.num_workers,
                             pin_memory = False)

    for idx1, X, y, missing_index1 in train_loader:
        for idx2, X_valid, y_valid, missing_index2 in eval_loader:
            processed_X, processed_X_valid = lmnn_transform(X, y, X_valid)
            best_eval_accuracy = 0
            best_y_valid_predict = torch.tensor([])
            # 定义网络及其他
            # Net
            net = MultiLayerPerceptron(input_size = args.input_size, classes = args.classes)
            # 优化器
            optim = Adam(net, 0.0005)
            # 损失函数
            loss_fn = nn.CrossEntropyLoss(weight = args.weight)
            # 进入循环
            epochs = 200
            for epoch in range(epochs):
                net.train(True)
                loss_sum = 0
                all_num = 0
                # train
                output = net(processed_X)
                loss = loss_fn(output, y)
                optim.zero_grad()
                loss.backward()
                optim.step()
                # 计算准确率
                train_accuracy = accuracy(output, y).data
                print("[Epoch %2d] loss: %.4f accuracy: %.4f" % (epoch + 1, loss_sum, train_accuracy))
                # valid
                net.train(False)
                with torch.no_grad():
                    output = net(processed_X_valid)
                    valid_accuracy = accuracy(output, y_valid)
                    print("valid accuracy: %.4f" % valid_accuracy)
                    if valid_accuracy >= best_eval_accuracy:
                        best_eval_accuracy = valid_accuracy
                        best_y_valid_predict = torch.argmax(output, dim = 1)
            print("---------------------------------------------")
            print("Best evaluate accuracy:{}".format(best_eval_accuracy))
            # plot_embedding(processed_X_valid, best_y_valid_predict, y_valid)
