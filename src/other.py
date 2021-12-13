# -*- coding: utf-8 -*-
# @Time    : 2021/11/19 13:26
# @Author  : nieyuzhou
# @File    : other.py
# @Software: PyCharm
import argparse
import os

from torch import nn
from torch.utils.data import DataLoader

from predict_model.net import FeatConcat
from utils.make_optim import Adam
from utils.pred_func import *
from utils.preprocess import get_missing_index, missing_data_process
from dataset.UKB_dataset import UKB_Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI', 'UKB'],
                        default = 'UKB')
    parser.add_argument('--missing_rate', type = float, default = 0,
                        help = 'view missing rate [default: 0]')
    parser.add_argument('--seed', type = int, default = 123)
    args = parser.parse_args()
    return args


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
    train_loader = DataLoader(train_dset, 400, num_workers = args.num_workers, shuffle = True,
                              pin_memory = False)
    eval_loader = DataLoader(eval_dset, 400, num_workers = args.num_workers, pin_memory = False)
    # batch_size:一次运行多少个sample
    # shuffle是打乱顺序: True两次顺序不同，False两次顺序相同，默认为false
    # num_workers: 采用多进程进行加载，默认为0，即逐进程；>0在windows下会出现错误
    # drop_last: 总sample除以batch_size除不尽的时候True舍弃
    # pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
    # ctrl+p可以查看参数
    epochs = 100
    # Net
    net = FeatConcat(args)  # choices = ["CPM", "CPM_GAN", "TMC"]
    # 优化器
    optim = Adam(net, 0.01)
    # 损失函数
    loss_fn = nn.BCELoss()
    best_eval_accuracy = 0
    for epoch in range(epochs):
        net.train(True)
        loss_sum = 0
        train_accuracy = 0
        all_num = 0
        # train
        for step, (idx, X, y, missing_index) in enumerate(train_loader):
            # 重建损失
            output = net(X)
            predicted = torch.argmax(output, dim = 1)
            y_onehot = torch.zeros(y.shape[0], args.classes).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            loss = loss_fn(output, y_onehot)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # 计算准确率
            train_accuracy += accuracy_count(predicted, y)
            all_num += y.size(0)
        train_accuracy = train_accuracy / all_num
        print("[Epoch %2d] loss: %.4f accuracy: %.4f" % (epoch + 1, loss_sum, (train_accuracy)))
        # valid
        valid_accuracy = 0
        net.train(False)
        with torch.no_grad():
            for step, (idx, X, y, missing_index) in enumerate(eval_loader):
                output = net(X)
                predicted = torch.argmax(output, dim = 1)
                valid_accuracy += accuracy_count(predicted, y)
                all_num += y.size(0)
            valid_accuracy = valid_accuracy / all_num
            print("valid accuracy: %.4f" % valid_accuracy)
            if valid_accuracy >= best_eval_accuracy:
                best_eval_accuracy = valid_accuracy
    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))
