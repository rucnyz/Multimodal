import math
import os
import time
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.pred_func import *


def train(net, loss_fn, train_loader, eval_loader, args):
    if args.log:
        writer = SummaryWriter("./logs_train")
        logfile = open(
            args.output + "/" + args.name +
            '/log_run.txt',
            'w+'
        )
        logfile.write(str(args))

    best_eval_accuracy = 0
    early_stop = 0
    decay_count = 0
    fluctuate_count = 0
    # Load the optimizer paramters
    optim = torch.optim.Adam(net.parameters(), lr = args.lr_base, weight_decay = 1e-5)
    eval_accuracies = []
    train_images = len(train_loader.dataset) / args.batch_size
    # train for each epoch
    for epoch in range(0, args.max_epoch):
        loss_sum = 0
        train_accuracy = 0
        all_num = 0
        time_start = time.time()
        for step, (idx, X, ans) in enumerate(train_loader):
            optim.zero_grad()
            for i in range(args.views):
                X[i] = X[i].to(args.device)
            ans = ans.to(args.device)
            evidence = net(X)
            loss = loss_fn(evidence, ans, epoch)
            loss.backward()
            train_accuracy += eval(args.pred_func)(evidence[args.views], ans)
            all_num += ans.size(0)
            loss_sum += loss.item()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m ""remaining" % (
                epoch + 1, step + 1, math.ceil(train_images), loss_sum / (step + 1),
                *[group['lr'] for group in optim.param_groups],
                ((time.time() - time_start) / (step + 1)) * (
                        (len(train_loader.dataset) / args.batch_size) - step) / 60,),
                  end = '   ')

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )

            optim.step()

        train_accuracy = 100 * train_accuracy / all_num

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
        elif fluctuate_count < 20:
            fluctuate_count += 1
        elif decay_count < args.lr_decay_times:
            fluctuate_count = 0
            # Decay
            print('LR Decay...')
            decay_count += 1
            if args.save_net:
                net.load_state_dict(torch.load(args.output + "/" + args.name +
                                               '/best' + str(args.seed) + str(args.dataset) + '.pkl')['state_dict'])
            # adjust_lr(optim, args.lr_decay)
            for group in optim.param_groups:
                group['lr'] *= args.lr_decay
        else:
            # Early stop
            early_stop += 1
            if early_stop == args.early_stop:
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


def evaluate(net, eval_loader, args):
    net.train(False)  # ==net.eval(),使得dropout和BatchNorm层参数被完全冻结
    accuracy = 0
    all_num = 0
    with torch.no_grad():
        for step, (ids, x, ans) in enumerate(eval_loader):
            for i in range(args.views):
                x[i] = x[i].to(args.device)
            ans = ans.to(args.device)
            evidence = net(x)
            accuracy += eval(args.pred_func)(evidence[args.views], ans)
            all_num += ans.size(0)
        accuracy = accuracy / all_num
    net.train(True)  # ==net.train()，恢复训练模式
    return 100 * np.array(accuracy)
