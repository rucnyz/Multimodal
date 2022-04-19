# -*- coding: utf-8 -*-
# @Time    : 2022/1/31 16:38
# @Author  : nieyuzhou
# @File    : svd.py
# @Software: PyCharm
import torch
from fancyimpute import IterativeSVD
import argparse
import os
import pickle
import time
import sklearn

from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AUROC
# from ignite.metrics import ROC_AUC
from predict_model.net import MultiLayerPerceptron
from utils.loss_func import classification_loss
from utils.make_optim import Adam
from utils.pred_func import *
from utils.preprocess import get_missing_index, missing_data_process
from dataset.UKB_dataset import UKB_Dataset
from dataset.UCI_dataset import UCI_Dataset
from dataset.UKB_ad_dataset import UKB_AD_Dataset
from dataset.UKB_balanced_dataset import UKB_BALANCED_Dataset
from dataset.UKB_all_dataset import UKB_ALL_Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI', 'UKB',
                                   'UKB_AD'],
                        default = 'UKB')
    parser.add_argument('--missing_rate', type = float, default = 0.3,
                        help = 'view missing rate [default: 0]')
    parser.add_argument('--seed', type = int, default = 123)
    parser.add_argument('--lr', type = float, default = 0.01)
    argument = parser.parse_args()
    return argument


# 直接连接
def feat_concat(x):
    concat_X = torch.tensor([], device = args.device)
    for i in range(args.views):
        concat_X = torch.cat((concat_X, x[i].to(args.device)), dim = 1)
    return concat_X


if __name__ == '__main__':
    if os.getcwd().endswith("src"):
        os.chdir("../")
    args = parse_args()
    args.dataloader = "UCI_Dataset"
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # args.device = torch.device("cpu")
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
    # Generate missing views
    dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')
    if args.dataloader == 'UKB_ALL_Dataset':
        missing_index = pickle.load(open(dataroot + "/missing_index_all2.pkl", "rb"))
        print("missing_rate = " + str(sum(sum(missing_index)) / (missing_index.shape[0] * missing_index.shape[1])))
    else:
        missing_index = get_missing_index(args.views, args.num, args.missing_rate)
    train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
    eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])

    train_dset.replace_with_nan()
    eval_dset.replace_with_nan()

    # DataLoader: DataSet的打包
    train_loader = DataLoader(train_dset, batch_size = int(args.num * 4 / 5), num_workers = args.num_workers,
                              shuffle = True,
                              pin_memory = False)
    eval_loader = DataLoader(eval_dset, batch_size = args.num - int(args.num * 4 / 5), num_workers = args.num_workers,
                             pin_memory = False)

    epochs = 15
    # Net
    net = MultiLayerPerceptron(input_size = sum(args.classifier_dims)).to(args.device)
    # 优化器
    optim = Adam(net, args.lr)
    # 损失函数
    loss_fn = nn.CrossEntropyLoss(weight = args.weight).to(args.device)
    best_eval_auc = 0
    processed_X_train = torch.tensor([])
    processed_X_valid = torch.tensor([])
    for epoch in range(epochs):
        net.train(True)
        loss_sum = 0
        train_auc = 0
        all_num = 0
        # train
        auc = AUROC(num_classes=y.max()+1).to(args.device)
        start_time = time.time()
        lsds = torch.tensor([])
        for step, (idx, X, y, missing_index) in enumerate(train_loader):
            y = y.to(args.device)
            y_onehot = torch.zeros(y.shape[0], args.classes, device = args.device).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            ys = y
            processed_X = torch.tensor([])
            if epoch == 0:
                processed_X = feat_concat(X)
                # svd降维
                processed_X = torch.Tensor(IterativeSVD().fit_transform(processed_X))
                processed_X_train = processed_X
            else:
                processed_X = processed_X_train

            lsds = torch.cat([lsds, processed_X])


            lsd_train = net(processed_X)
            clf_loss, predicted, prob = classification_loss(y_onehot, y, lsd_train, args.weight, args.device)
            optim.zero_grad()
            clf_loss.backward()
            optim.step()
            loss_sum += clf_loss.item()
            # 计算准确率
            train_auc = auc(prob, y)
        train_auc = auc.compute().data
        print("[Epoch %2d] loss: %.4f auc: %.4f" % (
            epoch + 1, loss_sum, train_auc))
        # valid

        all_num = 0
        valid_auc = 0
        net.train(False)
        auc = AUROC(num_classes=y.max()+1).to(args.device)
        with torch.no_grad():
            for step, (idx, X, y, missing_index) in enumerate(eval_loader):
                y = y.to(args.device)
                processed_X = torch.tensor([])
                if epoch == 0:
                    processed_X = feat_concat(X)
                    # svd降维
                    processed_X = torch.Tensor(IterativeSVD().fit_transform(processed_X))
                    processed_X_valid = processed_X
                else:
                    processed_X = processed_X_valid
                lsd_valid = net(processed_X)
                predicted, prob = ave(lsd_train, lsd_valid, y_onehot)
                valid_auc = auc(prob, y)
            valid_auc = auc.compute().data
            print("valid auc: %.4f  time: %.2f" % (valid_auc, time.time() - start_time))
            if valid_auc >= best_eval_auc:
                file = open('data/representations/featconcat_data.pkl', 'wb')
                pickle.dump((lsds, ys), file)
                best_eval_auc = valid_auc
    print("---------------------------------------------")
    print("Best evaluate auc:{}".format(best_eval_auc))

