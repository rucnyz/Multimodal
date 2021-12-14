# -*- coding: utf-8 -*-
# @Time    : 2021/10/15 0:28
# @Author  : nieyuzhou
# @File    : cca.py
# @Software: PyCharm
import argparse
import os

from cca_zoo.deepmodels import architectures, DCCAE
from mvlearn.embed import KMCCA
from torch import nn
from torch.utils.data import DataLoader

from dataset.UKB_dataset import UKB_Dataset
from utils.loss_func import classification_loss
from utils.pred_func import accuracy_count, ave
from utils.preprocess import *

if os.getcwd().endswith("src"):
    os.chdir("../")
parser = argparse.ArgumentParser()

parser.add_argument('--missing_rate', type = float, default = 0,
                    help = 'view missing rate [default: 0]')
parser.add_argument("--mode", default = 'client')
parser.add_argument("--port", default = 52162)
args = parser.parse_args()

torch.manual_seed(123)  # 设置CPU生成随机数的种子，方便下次复现实验结果
np.random.seed(123)

train_dset = UKB_Dataset('train', args)
eval_dset = UKB_Dataset('valid', args)
# 设置好丢失模态
missing_index = get_missing_index(args.views, args.num, args.missing_rate)
train_dset.set_missing_index(missing_index[:int(args.num * 4 / 5)])
eval_dset.set_missing_index(missing_index[int(args.num * 4 / 5):])
# 均值填充
train_dset.replace_with_mean()
eval_dset.replace_with_mean()

# ----------------------------------------
# 设置loader
train_loader = DataLoader(train_dset, batch_size = 400, shuffle = True,
                          pin_memory = False)
eval_loader = DataLoader(eval_dset, batch_size = 400, pin_memory = False)
# 设置模型和优化器
encoders = []
decoders = []
latent_dims = 100
for ld in args.classifier_dims:
    encoders.append(architectures.Encoder(latent_dims = latent_dims, feature_size = ld))
    decoders.append(architectures.Decoder(latent_dims = latent_dims, feature_size = ld))
dcca = DCCAE(latent_dims = latent_dims, encoders = encoders, decoders = decoders, r = 0.2)
optim_cca = torch.optim.Adam(dcca.parameters(), lr = 0.01)
bce_loss = nn.BCELoss()
# 开始训练
epochs = 100
# 测试一下
best_eval_accuracy = 0
for epoch in range(epochs):
    dcca.train(True)
    loss_sum = 0
    train_accuracy = 0
    all_num = 0
    # train
    for step, (idx, X, y, missing_index) in enumerate(train_loader):
        # 重建损失
        loss = dcca.loss(list(X.values()))
        # 分类损失
        lsd_dim = 0
        lsd = dcca(list(X.values()))
        for i in range(args.views):
            lsd_dim += lsd[i]
        y_onehot = torch.zeros(y.shape[0], args.classes).scatter_(1, y.reshape(
            y.shape[0], 1), 1)
        loss2, _ = classification_loss(y_onehot, y, lsd_dim)
        optim_cca.zero_grad()
        loss.backward()
        loss2.backward()
        optim_cca.step()
        loss_sum += loss2.item()
        # 计算准确率
        predicted = ave(lsd_dim, lsd_dim, y_onehot)
        train_accuracy += accuracy_count(predicted, y)
        all_num += y.size(0)
    train_accuracy = train_accuracy / all_num
    print("[Epoch %2d] loss: %.4f accuracy: %.4f" % (epoch + 1, loss_sum, (train_accuracy)))
    # valid
    valid_accuracy = 0
    all_num = 0
    dcca.train(False)
    with torch.no_grad():
        for step, (idx, X, y, missing_index) in enumerate(eval_loader):
            lsd_dim = 0
            lsd = dcca(list(X.values()))
            y_onehot = torch.zeros(y.shape[0], args.classes).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            for i in range(args.views):
                lsd_dim += lsd[i]
            predicted = ave(lsd_dim, lsd_dim, y_onehot)
            valid_accuracy += accuracy_count(predicted, y)
            all_num += y.size(0)
        valid_accuracy = valid_accuracy / all_num
        print("valid accuracy: %.4f" % valid_accuracy)
        if valid_accuracy >= best_eval_accuracy:
            best_eval_accuracy = valid_accuracy
print("---------------------------------------------")
print("Best evaluate accuracy:{}".format(best_eval_accuracy))

# ----------------------------------------

# 预处理数据集
# X = list(train_dset.full_data.values())
# X_valid = list(eval_dset.full_data.values())
# y = train_dset.full_labels
# y_valid = eval_dset.full_labels
# # 使用CCA方法得到隐藏层
# components = 10
# mcca = KMCCA(n_components = components, regs = 0.1)
# scores_train = mcca.fit_transform(X)
# s_train = np.empty((scores_train.shape[1], scores_train.shape[0] * scores_train.shape[2]))
# for i in range(scores_train.shape[0]):
#     s_train[:, i * components:(i + 1) * components] = scores_train[i]
# # 验证集隐藏层
# scores_valid = mcca.fit_transform(X_valid)
# s_valid = np.empty((scores_valid.shape[1], scores_valid.shape[0] * scores_valid.shape[2]))
# for i in range(scores_train.shape[0]):
#     s_valid[:, i * components:(i + 1) * components] = scores_valid[i]
# # 设置为tensor
# s_train = torch.from_numpy(s_train).float()
# s_valid = torch.from_numpy(s_valid).float()
#
# # 跑一个简单的模型
# net = nn.Sequential(
#     nn.Linear(scores_train.shape[0] * scores_train.shape[2], 128),
#     nn.ReLU(),
#     nn.Linear(128, y.max() - y.min() + 1),
#     nn.Softmax(dim = 1)
# )
# loss_fn = nn.CrossEntropyLoss()
# optim = torch.optim.Adam(net.parameters(), lr = 0.1)
# # 开始运行
# best_valid_acc = 0
# for i in range(200):
#     predicted = net(s_train)
#     loss = loss_fn(predicted, y)
#     optim.zero_grad()
#     loss.backward()
#     optim.step()
#     # 计算准确率
#     y_predicted = torch.argmax(predicted, dim = 1)
#     train_acc = ((y_predicted == y).sum() / y.shape[0] * 100).item()
#     # 验证部分
#     with torch.no_grad():
#         predicted_valid = net(s_valid)
#         y_predicted_valid = torch.argmax(predicted_valid, dim = 1)
#         valid_acc = ((y_predicted_valid == y_valid).sum() / y_valid.shape[0] * 100).item()
#     if best_valid_acc < valid_acc:
#         best_valid_acc = valid_acc
#     print("第{}个epoch：\n训练准确率:{:.4f}     验证准确率:{:.4f}".format(i, train_acc, valid_acc))
# print("------------------------------\n最高验证集准确率:{:.4f}".format(best_valid_acc))
# ----------------------------------------
# 使用SVM测试
# clf = SVC().fit(s, y)
# y_predict = clf.predict(s)
# svm_acc = accuracy_score(y, y_predict)
# print("训练集:\n svm:{:.4f}".format(svm_acc))
# -------------------------------------
# 验证集试试
#
# y_predict = clf.predict(s_valid)
# svm_acc = accuracy_score(y_valid, y_predict)
# print("验证集:\n svm:{:.4f}".format(svm_acc))
