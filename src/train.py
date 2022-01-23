import math
import os
import pickle
import time

import torch
from torch.utils.tensorboard import SummaryWriter

from utils.loss_func import *
from utils.pred_func import *


def train(net, optim, train_loader, eval_loader, args):
    if args.log:
        writer = SummaryWriter("./logs_train")  # 向log_dir文件夹写入的事件文件
        logfile = open(
            args.output + "/" + args.name +
            '/log_run.txt',
            'w+'
        )  # args.output默认为ckpt
        logfile.write(str(args))
    best_eval_accuracy = 0  # 最佳验证准确率
    best_train_accuracy = 0
    decay_count = 0
    train_images = math.ceil(len(train_loader.dataset) / args.train_batch_size)
    bce_loss = MyBCELoss(args)
    # train for each epoch
    for epoch in range(0, args.max_epoch):
        # 初始化参数
        all_num = 0
        train_accuracy = 0
        time_start = time.time()
        rec_loss_sum = 0
        dec_loss_sum = 0
        clf_loss_sum = 0
        # 开始运行
        for step, (idx, X, y, missing_index) in enumerate(train_loader):
            # 使用cuda或者cpu设备
            for i in range(args.views):
                X[i] = X[i].to(args.device)
            y = y.to(args.device)
            missing_index = missing_index.to(args.device)
            # 用GAN生成数据补充缺失模态
            if epoch > args.GAN_start:
                with torch.no_grad():
                    x_pred = net.decoder(net.encoder(X, missing_index))
                    fill_data(X, x_pred, missing_index)
                    missing_index = torch.ones(missing_index.shape, device = args.device)
            # 产生one-hot编码的标签
            y_onehot = torch.zeros(y.shape[0], args.classes, device = args.device).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            # --------------------------------------
            # 重建原数据,考虑缺失模态情况
            lsd_train = net.encoder(X, missing_index)  # （1600，128）
            x_pred = net.decoder(lsd_train)
            # 冻结decoder参数(decoder即generator)，计算discriminator损失
            # 取每一个模态X中未缺失部分作为正样本1，和x_pred中X缺失部分对应的数据作为生成样本0，训练discriminator
            # ######## 如何体现是否缺失 ????????? encoder如何生成缺失数据 ????????
            output_real = net.discriminator(X)
            output_fake = net.discriminator(x_pred)
            disc_loss_real = bce_loss('exist', output_real, missing_index)
            disc_loss_fake = bce_loss('miss', output_fake, missing_index)
            disc_loss = disc_loss_real + disc_loss_fake
            optim["discriminator"].zero_grad()
            disc_loss.backward(retain_graph = True)
            optim["discriminator"].step()
            # --------------------------------------
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
            # 冻结discriminator参数，更新decoder参数
            # x_pred中X【缺失】部分对应的数据作为正样本
            output_fake = net.discriminator(x_pred)
            dec_loss = bce_loss('decoder', output_fake, missing_index)
            optim["decoder"].zero_grad()
            (rec_loss + dec_loss).backward(retain_graph = True)
            optim["decoder"].step()
            # --------------------------------------
            # 重建原数据
            x_pred = net.decoder(lsd_train)
            output_fake = net.discriminator(x_pred)
            dec_loss = bce_loss('decoder', output_fake, missing_index)
            # 计算未缺失模态的重建损失和分类损失
            # x_pred-->decoder & encoder  lsd_train-->encoder
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
            clf_loss, predicted = classification_loss(y_onehot, y, lsd_train)
            optim["encoder"].zero_grad()
            (rec_loss + clf_loss + dec_loss).backward()
            optim["encoder"].step()
            # ---------------------------------------
            rec_loss_sum += rec_loss.item()
            dec_loss_sum += dec_loss.item()
            clf_loss_sum += clf_loss.item()
            train_accuracy += eval(args.pred_func)(predicted, y)
            all_num += y.size(0)
            # 训练过step+1次，计算平均loss
            print(
                "\r[Epoch %2d][Step %4d/%4d] rec_loss: %.4f, clf_loss:%.4f, dec_loss:%.4f" % (
                    epoch + 1, step + 1, train_images, rec_loss_sum / (step + 1), clf_loss_sum / (step + 1),
                    dec_loss_sum / (step + 1),), end = " ")
        train_accuracy = 100 * train_accuracy / all_num
        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy
        # 记录时间
        time_end = time.time()
        elapse_time = time_end - time_start
        print('\nFinished in {:.4f}s'.format(elapse_time))
        print("Train Accuracy :" + str(train_accuracy))
        # Eval
        print('Evaluation...    decay times: {}'.format(decay_count))
        valid_accuracy, lsds = evaluate(net, eval_loader, args)
        print('Valid Accuracy :' + str(valid_accuracy) + '\n')
        if (valid_accuracy > best_eval_accuracy) or (
                valid_accuracy == best_eval_accuracy and train_accuracy > best_train_accuracy):
            # Best
            state = {
                'state_dict': net.state_dict(),
                'optim_encoder': optim["encoder"].state_dict(),
                'optim_decoder': optim["decoder"].state_dict(),
                'optim_discriminator': optim["discriminator"].state_dict(),
                'args': args,
            }
            torch.save(
                state,
                args.output + "/" + args.name +
                '/best' + str(args.seed) + str(args.dataset) + str(args.missing_rate) + "-" + str(args.loop_times) +
                '.pkl'
            )
            file = open('data/representations/CPM_GAN' +
                        str(args.missing_rate) + '_' +
                        str(args.loop_times) + '_' +
                        str(args.GAN_start) + '_' + 'data.pkl', 'wb')
            pickle.dump((lsds, eval_loader.dataset.full_labels), file)
            best_eval_accuracy = valid_accuracy
    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))


def train1(net, loss_fn, optim, train_loader, eval_loader, args):
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
    train_images = math.ceil(len(train_loader.dataset) / args.train_batch_size)

    # train for each epoch
    for epoch in range(0, args.max_epoch):
        time_start = time.time()
        loss_sum, train_accuracy = train_TMC(args, epoch, loss_fn, net, optim, train_images,
                                             train_loader, time_start)
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {:.4f}s'.format(elapse_time))
        print("Train Accuracy :" + str(train_accuracy))

        # Eval
        print('Evaluation...    decay times: {}'.format(decay_count))
        valid_accuracy = evaluate_TMC(net, eval_loader, args)
        print('Valid Accuracy :' + str(valid_accuracy) + '\n')
        if args.log:
            # logging train for tensorBoard and file
            writer.add_scalar("loss/train_each_epoch", loss_sum / train_images, epoch)
            logfile.write(
                'Epoch: ' + str(epoch + 1) +
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


def train2(net, optim, train_loader, eval_loader, args):
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
    train_images = math.ceil(len(train_loader.dataset) / args.train_batch_size)
    # train for each epoch
    for epoch in range(0, 200):
        time_start = time.time()
        loss_sum, train_accuracy, label_onehot, id = train_CPM(args, epoch, net, optim, train_images,
                                                               train_loader, time_start)
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {:.4f}s'.format(elapse_time))
        epoch_finish = epoch + 1
        print("Train Accuracy :" + str(train_accuracy))
        # Eval
        print('Evaluation...    decay times: {}'.format(decay_count))
        valid_accuracy = evaluate_CPM(args, net, optim, eval_loader, label_onehot, id)
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
        y = y.to(args.device)  # class真实值
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
            epoch + 1, step + 1, train_images, loss_sum / (step + 1),  # 训练过step+1次，计算平均loss
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


"""
此为运行CPM模型单次epoch的函数，注意batch_size已经被改成了整个数据集大小，也就是说目前不支持迭代多个batch运行模型，原因在model_CPM里71行
模型核心内容：
    计算重建损失(reconstruction_loss)和分类损失(classification_loss),更新模型参数以及隐含层数据
    此模型的大部分流程不在model_CPM的forward当中(源代码用TensorFlow写成，不太好改成完全按照pytorch流程的写法，如果一定要这么改可能要在外
    面套一层模型，但这样能用我也就没再花时间在上面了)
    具体流程见下方注释
Args: 
    missing_index:完整的缺失索引矩阵(N,V)，N为数据总量，V为模态数，在下面每次取前80%数据再按照打乱索引取得对应训练集
    (每次取的内容是一样的，每次都算一遍没有什么特殊意义)
Return:    
    loss,accuracy:作为损失和准确度被记录以及输出
    label_onehot,id:返回供evaluate使用
    label_onehot: 在evaluate的评估函数ave中需要使用训练集的label_onehot(就是使用了onehot编码的训练集标签);
    id: 训练集shuffle后的索引信息，由于训练集是被shuffle打乱过的，而在evaluate时需要使用训练集数据，故必须传进打乱后的索引信息才能保证
    evaluate时使用的是和训练时相同的训练集数据
    
注意：
    后面基本用到训练数据的地方都使用了idx，仔细理解代码后可能会觉得不需要，其实这也是历史遗留问题，开始因为代码跑起来效果奇差找不出原因，就尝试
    做了很多修改，最后真的找到了原因，然而之前的一些修改就保留了下来而没有恢复，比如这个idx开始我以为就是因为不加它所以使模型效果变差，
    但其实应该是没有影响，我也还没有试过删掉对比
"""


def train_CPM(args, epoch, net, optim, train_images, train_loader, time_start):
    train_accuracy = 0
    all_num = 0
    clf_loss = 0
    rec_loss = 0
    id = None
    label_onehot = None
    # train_batch等于训练数据集大小，循环只会跑一次
    for step, (idx, X, y, missing_index) in enumerate(train_loader):
        # id用于回传(注意每次idx都是一样的，因为设定好了seed)
        # y: 所有数据的类别标签
        for i in range(args.views):
            X[i] = X[i].to(args.device)
        y = y.to(args.device)
        id = idx
        missing_index.to(args.device)
        # 用训练集标签生成one_hot标签（即每个数据标签变成(1,10)向量，属于那个类别该类别为1，其他为0）
        label_onehot = torch.zeros(args.train_batch_size, args.classes, device = args.device).scatter_(1, y.reshape(
            y.shape[0], 1), 1)
        # (1600,10)
        # y = y.scatter(dim,index,src)
        # 则结果为：
        # y[ index[i][j][k] ] [j][k] = src[i][j][k] # if dim == 0
        # y[i] [ index[i][j][k] ] [k] = src[i][j][k] # if dim == 1
        # y[i][j] [ index[i][j][k] ]  = src[i][j][k] # if dim == 2
        # 生成训练集缺失模态索引
        # missing_index[:int(args.num * 4 / 5)]是训练集的缺失模态索引
        # missing_index[int(args.num * 4 / 5)：]是验证集的缺失模态索引
        # 实际上，missing_index[:int(args.num * 4 / 5)][idx][:, i]和missing_index[idx][:, i]等价，因为idx最大为1599

        # 首先进行重建，也就是decoder的过程，具体见reconstruction_loss函数。optim[0]更新的是模型参数
        # reconstruction_loss体现的是经过autoencoder后
        # train the network to minimize reconstruction loss
        for i in range(5):
            # x_pred得到的是训练数据，注意我们这里要做的是尽可能让隐藏层lsd_train通过网络变得更像原训练数据集X
            x_pred = net(net.lsd_train[idx])  # 输出是训练数据各模态的特征数
            # 所有不缺失数据的误差平方和--> 使得经过net生成的数据与原来数据接近
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index[idx])
            # CPM的优化器是MixAdam，有3个optim
            # optim[0]更新模型参数
            # optim[1]更新lsd_train数据
            # optim[2]更新lsd_test数据
            optim[0].zero_grad()
            rec_loss.backward(retain_graph = True)
            optim[0].step()
        # 随后同时进行重建以及分类，optim[1]更新的是隐藏层lsd_train数据
        # train the latent space data to minimize reconstruction loss and classification loss
        for i in range(5):
            x_pred = net(net.lsd_train[idx])
            loss1 = reconstruction_loss(args.views, x_pred, X, missing_index[idx])
            loss2, _ = net.lamb * classification_loss(label_onehot, y, net.lsd_train[idx])
            optim[1].zero_grad()
            loss1.backward()
            loss2.backward()
            optim[1].step()
        # 最后算一次，进行输出
        x_pred = net(net.lsd_train[idx])
        clf_loss, predicted = classification_loss(label_onehot, y, net.lsd_train[idx])
        rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index[idx])
        print(
            "\r[Epoch %2d][Step %4d/%4d] Reconstruction Loss: %.4f, Classification Loss = %.4f, Lr: %.2e, %4d m remaining"
            % (epoch + 1, step + 1, train_images, rec_loss, clf_loss,
               *[group['lr'] for group in optim[1].param_groups],
               ((time.time() - time_start) / (step + 1)) * ((len(train_loader.dataset) / args.batch_size) - step) / 60),
            end = '   ')
        train_accuracy += eval(args.pred_func)(predicted, y)
        all_num += y.size(0)
    train_accuracy = 100 * train_accuracy / all_num

    return clf_loss + rec_loss, train_accuracy, label_onehot, id


# 将X中的缺失数据用x_pred即生成数据来代替
def fill_data(X, x_pred, missing_index):
    for i in range(len(X)):
        X[i][missing_index[:, i] == 0] = x_pred[i][missing_index[:, i] == 0]


def evaluate(net, eval_loader, args):
    lsds = torch.tensor([])
    accuracy = 0
    all_num = 0
    net.train(False)
    with torch.no_grad():
        for step, (ids, X, y, missing_index) in enumerate(eval_loader):
            for i in range(args.views):
                X[i] = X[i].to(args.device)
            y = y.to(args.device)
            missing_index = missing_index.to(args.device)
            # 生成one_hot编码的label来进行后续分类
            y_onehot = torch.zeros(y.shape[0], args.classes, device = args.device).scatter_(1, y.reshape(
                y.shape[0], 1), 1)
            lsd_valid = net.encoder(X, missing_index)  # (1600，128)
            x_pred = net.decoder(lsd_valid)
            # 将预测值补充到原数据中,loop_times决定了跑几个来回
            for i in range(args.loop_times):
                fill_data(X, x_pred, missing_index)
                lsd_valid = net.encoder(X, torch.ones(missing_index.shape, device = args.device))
                x_pred = net.decoder(lsd_valid)
            lsds = torch.concat([lsds, lsd_valid])
            predicted = ave(lsd_valid, lsd_valid, y_onehot)
            accuracy += eval(args.pred_func)(predicted, y)
            all_num += y.size(0)
        accuracy = accuracy / all_num
    net.train(True)  # ==net.train()，恢复训练模式
    return 100 * np.array(accuracy), lsds


def evaluate_TMC(net, eval_loader, args):
    net.train(False)
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


# 和前面train_CPM对应
def evaluate_CPM(args, net, optim, valid_loader, label_onehot, id):
    valid_accuracy = 0
    all_num = 0
    for step, (idx, X, y, missing_index) in enumerate(valid_loader):
        for i in range(args.views):
            X[i] = X[i].to(args.device)
        y = y.to(args.device)
        missing_index = missing_index.to(args.device)

        # 注意此处我们不再更新网络参数，只关心验证集的隐藏层数据(很好理解因为网络相当于在训练时被更新好了，
        # 现在验证时，我们需要让隐藏层和数据集对应才能验证网络更新的咋样)
        for i in range(5):
            x_pred = net(net.lsd_valid)
            rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
            optim[2].zero_grad()
            rec_loss.backward()
            optim[2].step()
        x_pred = net(net.lsd_valid)  # (400,76)(400,216)(400,64)(400,240)(400,47)(400,6)
        rec_loss = reconstruction_loss(args.views, x_pred, X, missing_index)
        predicted = ave(net.lsd_train[id], net.lsd_valid, label_onehot)
        # 在eval又不去反向传播损失，完全没必要用classification_loss这个函数, 下面的valid_accuracy直接计算是否分类正确
        print("Reconstruction Loss = {:.4f}".format(rec_loss))
        predicted = predicted.reshape(len(predicted), 1)
        y = y.reshape(len(y), 1)
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
