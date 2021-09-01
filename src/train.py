import math
import os
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import utils.loss_func
from utils.pred_func import *

# eval_accuracies = train(net, loss_fn, train_loader, eval_loader, args)
def train(net, loss_fn, optim, train_loader, eval_loader, args):
    if args.log:
        writer = SummaryWriter("./logs_train")  # 向log_dir文件夹写入的事件文件
        logfile = open(
            args.output + "/" + args.name +
            '/log_run.txt',
            'w+'
        )  # args.output默认为ckpt
        logfile.write(str(args))
    best_eval_accuracy = 0  # 最佳验证准确率
    early_stop = 0
    decay_count = 0
    fluctuate_count = 0
    eval_accuracies = []  # 记录每一次验证集的准确率
    train_images = int(len(train_loader.dataset) / args.train_batch_size)  # 1600/64=25

    # train for each epoch
    for epoch in range(0, args.max_epoch):
        time_start = time.time()
        loss_sum, train_accuracy = train_TMC(args, epoch, loss_fn, net, optim, train_images,
                                             train_loader, time_start)
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {:.4f}s'.format(elapse_time))
        epoch_finish = epoch + 1
        print("Train Accuracy :" + str(train_accuracy))

        # Eval
        print('Evaluation...    decay times: {}'.format(decay_count))
        valid_accuracy = evaluate(net, eval_loader, args)
        print('Valid Accuracy :' + str(valid_accuracy) + '\n')
        if args.log:
            # logging train for tensorBoard and file
            writer.add_scalar("loss/train_each_epoch", loss_sum / train_images, epoch)
            logfile.write(
                'Epoch: ' + str(epoch_finish) +
                ', Loss: ' + str(loss_sum / train_images) +
                ', Lr: ' + str([group['lr'] for group in optim.param_groups]) + '\n' +
                'Elapsed time: ' + str(elapse_time) +
                '\n'
            )
            # logging valid for tensorBoard and file
            writer.add_scalars("accuracy", {"train": train_accuracy, "valid": valid_accuracy}, epoch)
            logfile.write(
                'Evaluation Accuracy: ' + str(valid_accuracy) +
                '\n\n'
            )

        eval_accuracies.append(valid_accuracy)
        # 更新best_eval_accuracy
        if valid_accuracy >= best_eval_accuracy:
            fluctuate_count = 0
            if args.save_net:
                # Best
                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.state_dict(),
                    'args': args,
                }
                torch.save(
                    state,
                    args.output + "/" + args.name +
                    '/best' + str(args.seed) + str(args.dataset) + '.pkl'
                )
            best_eval_accuracy = valid_accuracy
            early_stop = 0
        elif fluctuate_count < 50:  # 验证集准确率没有提高
            fluctuate_count += 1
        elif decay_count < args.lr_decay_times:  # 经过了50次，验证集准确率还没有提高
            # args.lr_decay_times 默认为2
            fluctuate_count = 0
            # Decay
            print('LR Decay...')
            decay_count += 1
            if args.save_net:  # 保存模型
                # 加载当前最好的模型
                net.load_state_dict(torch.load(args.output + "/" + args.name +
                                               '/best' + str(args.seed) + str(args.dataset) + '.pkl')['state_dict'])
            for group in optim.param_groups:
                group['lr'] *= args.lr_decay
        else:  # 在max_epoch之前结束，即提前结束
            # Early stop
            early_stop += 1
            if early_stop == args.early_stop:  # args.early_stop默认为3
                print("---------------------------------------------")
                print('Early stop reached')
                print("Best evaluate accuracy:{}".format(best_eval_accuracy))
                if args.save_net:
                    os.rename(args.output + "/" + args.name +
                              '/best' + str(args.seed) + str(args.dataset) + '.pkl',
                              args.output + "/" + args.name +
                              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + str(args.dataset) + '.pkl')
                if args.log:
                    logfile.write(
                        '-----------------------------------------------------\n'
                        'Early stop reached\n'
                        'Best evaluate accuracy:' + str(best_eval_accuracy)
                    )
                    logfile.close()
                    writer.close()
                return eval_accuracies

    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))
    if args.log:
        logfile.write(
            '-----------------------------------------------------\n'
            'Best evaluate accuracy:' + str(best_eval_accuracy)
        )
        writer.close()
        logfile.close()


def train2(net, optim, train_loader, eval_loader, missing_index, args):
    if args.log:
        writer = SummaryWriter("./logs_train")
        logfile = open(
            args.output + "/" + args.name +
            '/log_run.txt',
            'w+'
        )
        logfile.write(str(args))
    best_eval_accuracy = 0
    decay_count = 0
    train_images = int(len(train_loader.dataset) / args.train_batch_size)
    # train for each epoch
    for epoch in range(0, 200):
        time_start = time.time()
        loss_sum, train_accuracy, label_onehot, id = train_CPM(args, epoch, net, optim, train_images,
                                                               train_loader, missing_index, time_start)
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {:.4f}s'.format(elapse_time))
        epoch_finish = epoch + 1
        print("Train Accuracy :" + str(train_accuracy))
        # Eval
        print('Evaluation...    decay times: {}'.format(decay_count))
        valid_accuracy = evaluate_CPM(args, net, optim, eval_loader, missing_index, label_onehot, id)
        print('Valid Accuracy :' + str(valid_accuracy) + '\n')
        if args.log:
            # logging train for tensorBoard and file
            writer.add_scalar("loss/train_each_epoch", loss_sum / train_images, epoch)
            logfile.write(
                'Epoch: ' + str(epoch_finish) +
                ', Loss: ' + str(loss_sum / train_images) +
                ', Lr: ' + str([group['lr'] for group in optim[1].param_groups]) + '\n' +
                'Elapsed time: ' + str(elapse_time) +
                '\n'
            )
            # logging valid for tensorBoard and file
            writer.add_scalars("accuracy", {"train": train_accuracy, "valid": valid_accuracy}, epoch)
            logfile.write(
                'Evaluation Accuracy: ' + str(valid_accuracy) +
                '\n\n'
            )
        if valid_accuracy >= best_eval_accuracy:
            best_eval_accuracy = valid_accuracy

    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))
    if args.log:
        logfile.write(
            '-----------------------------------------------------\n'
            'Best evaluate accuracy:' + str(best_eval_accuracy)
        )
        writer.close()
        logfile.close()


def train_TMC(args, epoch, loss_fn, net, optim, train_images, train_loader, time_start):
    loss_sum = 0
    train_accuracy = 0
    all_num = 0  # 记录训练的sample数
    for step, (idx, X, y) in enumerate(train_loader):
        optim.zero_grad()
        for i in range(args.views):
            X[i] = X[i].to(args.device)
        y = y.to(args.device) # class真实值
        evidence = net(X)  # 在evidence字典最后加上最终的预测结果结果
        loss = loss_fn(evidence, y, epoch)
        # loss.backward(retain_graph = True)
        loss.backward()
        _, predicted = torch.max(evidence[args.views].data, 1)  # 返回综合所有模态得到的每个数据(这里是一个batch的所有数据，默认为64个)概率最大的类别
        # torch.max(a,0)返回每一列中最大值的那个元素，且返回索引
        # torch.max(a,1)返回每一行中最大值的那个元素，且返回其索引
        train_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)  # 即batchsize
        loss_sum += loss.item()

        print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m ""remaining" % (
            epoch + 1, step + 1, math.ceil(train_images), loss_sum / (step + 1), # 已经训练过的step的平均loss
            *[group['lr'] for group in optim.param_groups],
            ((time.time() - time_start) / (step + 1)) * (
                    (len(train_loader.dataset) / args.batch_size) - step) / 60,),
              end = '   ')

        # Gradient norm clipping
        # 如果梯度超过阈值，那么就截断，将梯度变为阈值-->用于解决神经网络训练过拟合的方法
        if args.grad_norm_clip > 0:
            nn.utils.clip_grad_norm_(
                net.parameters(),
                args.grad_norm_clip
            )
        optim.step()
    train_accuracy = 100 * train_accuracy / all_num
    return loss_sum, train_accuracy


def train_CPM(args, epoch, net, optim, train_images, train_loader, missing_index, time_start):
    train_accuracy = 0
    all_num = 0
    classification_loss = 0
    reconstruction_loss = 0
    id = None
    label_onehot = None
    for step, (idx, X, y) in enumerate(train_loader):
        id = idx
        label_onehot = torch.zeros(args.train_batch_size, args.classes).scatter_(1, y.reshape(y.shape[0], 1), 1)
        train_missing_index = dict()
        for i in range(args.views):
            train_missing_index[i] = torch.from_numpy(
                missing_index[:int(args.num * 4 / 5)][idx][:, i].reshape(args.train_batch_size, 1))
        for i in range(5):
            x_pred = net(net.lsd_train[idx])
            reconstruction_loss = net.reconstruction_loss(x_pred, X, train_missing_index)
            optim[0].zero_grad()
            reconstruction_loss.backward(retain_graph = True)
            optim[0].step()
        for i in range(5):
            x_pred = net(net.lsd_train[idx])
            loss1 = net.reconstruction_loss(x_pred, X, train_missing_index)
            loss2, _ = net.lamb * utils.loss_func.classification_loss(label_onehot, y, net.lsd_train[idx])
            optim[1].zero_grad()
            loss1.backward()
            loss2.backward()
            optim[1].step()
        x_pred = net(net.lsd_train[idx])
        classification_loss, predicted = utils.loss_func.classification_loss(label_onehot, y, net.lsd_train[idx])
        reconstruction_loss = net.reconstruction_loss(x_pred, X, train_missing_index)
        print(
            "\r[Epoch %2d][Step %4d/%4d] Reconstruction Loss: %.4f, Classification Loss = %.4f, Lr: %.2e, %4d m remaining"
            % (epoch + 1, step + 1, math.ceil(train_images), reconstruction_loss, classification_loss,
               *[group['lr'] for group in optim[1].param_groups],
               ((time.time() - time_start) / (step + 1)) * ((len(train_loader.dataset) / args.batch_size) - step) / 60),
            end = '   ')
        train_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)
    train_accuracy = 100 * train_accuracy / all_num

    return classification_loss + reconstruction_loss, train_accuracy, label_onehot, id


def evaluate(net, eval_loader, args):
    net.train(False)  # ==net.eval(),使得dropout和BatchNorm层参数被完全冻结
    accuracy = 0
    all_num = 0
    with torch.no_grad():
        for step, (ids, x, y) in enumerate(eval_loader):
            for i in range(args.views):
                x[i] = x[i].to(args.device)
            y = y.to(args.device)
            evidence = net(x)
            _, predicted = torch.max(evidence[args.views].data, 1)
            accuracy += eval(args.pred_func)(predicted, y)
            all_num += y.size(0)
        accuracy = accuracy / all_num
    net.train(True)  # ==net.train()，恢复训练模式
    return 100 * np.array(accuracy)


def evaluate_CPM(args, net, optim, valid_loader, missing_index, label_onehot, id):
    valid_accuracy = 0
    all_num = 0
    for step, (idx, X, y) in enumerate(valid_loader):
        valid_missing_index = dict()
        for i in range(args.views):
            valid_missing_index[i] = torch.from_numpy(
                missing_index[int(args.num * 4 / 5):][:, i].reshape(args.valid_batch_size, 1))

        # update the h
        for i in range(5):
            x_pred = net(net.lsd_valid)
            reconstruction_loss = net.reconstruction_loss(x_pred, X, valid_missing_index)
            optim[2].zero_grad()
            reconstruction_loss.backward()
            optim[2].step()
        x_pred = net(net.lsd_valid)
        reconstruction_loss = net.reconstruction_loss(x_pred, X, valid_missing_index)
        predicted = ave(net.lsd_train[id], net.lsd_valid, label_onehot)
        print("Reconstruction Loss = {:.4f}".format(reconstruction_loss))
        valid_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)
    valid_accuracy = 100 * valid_accuracy / all_num
    return valid_accuracy

# 卷积+池化+非线性激活

# F.conv2d
# reshape: (input, (batch_size不知道多少时写-1, channel, x, x))
# weight: 权重，即卷积核
# stride: 步数

# nn.conv2d
# groups一般都为1
# bias偏置，常为True
# 常设置前五个参数：in_channels out_channels kernel_size在训练中不断进行调整: int(x,x) or tuple(x,y) stride padding
# out_channel=2时有两个卷积核

# nn.maxpool2d 最大池化也称下采样 maxunpool2d上采样
# 作用：保留输入特征的同时减少数据量 池化channel数不会改变
# 常设置一个参数: kernel_size
# dilation空洞卷积，一般不设置
# ceil_mode: True: ceil（向上取整，即保留） 而不是 floor（向下取整，即舍去）
# 池化核没有权重，选择最大的
# stride默认为kernel_size
# input矩阵: torch.tensor([[],[]], dtype=torch.float32)

# 非线性激活
# Relu
# Sigmoid
# input (N,*) output (N,*)
# replace=True 直接修改

# nn.BatchNorm2d 加快神经网络的训练速度 不常用
# nn.RNN 文字识别 特定网络结构
# dropout 按照p的概率随机把数据变为0
# embedding 自然语言处理

# linear 线性层
# in_feature: 输入变量个数 input layer  out_feature: output layer
# kx+b(bias=True)
# (1, 1, 1, -1) torch.flatten [x]
# input-->hidden-->output要经过两个线性层
# torchvision


# loss_function
# nn.L1Loss: 绝对值之和/平均
# nn.MSELoss: 平方和/均值
# nn.CrossEntropyLoss(用于分类问题): input(N,C) target(N)

# optimizer 优化器
# lr(learning rate): 学习率
