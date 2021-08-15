import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.utils.pred_func import *


def train(net, train_loader, eval_loader, args):
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
    # Load the loss function
    # loss_fn = args.loss_fn
    # if torch.cuda.is_available():
    #     loss_fn = loss_fn.cuda()
    eval_accuracies = []
    train_images = len(train_loader.dataset) / args.batch_size
    # train for each epoch
    for epoch in range(0, args.max_epoch):
        loss_sum = 0
        time_start = time.time()
        for step, (idx, X, ans) in enumerate(train_loader):
            optim.zero_grad()
            if torch.cuda.is_available():
                for i in range(len(X)):
                    X[i] = X[i].cuda()
                ans = ans.cuda()
            evidence, evidence_all, loss = net(X, ans, step)
            # loss = loss_fn(pred, ans)
            loss.backward()

            loss_sum += loss.item()

            print("\r[Epoch %2d][Step %4d/%4d] Loss: %.4f, Lr: %.2e, %4d m ""remaining" % (
                epoch + 1, step + 1, int(train_images), loss_sum / (step + 1),
                *[group['lr'] for group in optim.param_groups],
                ((time.time() - time_start) / (step + 1)) * (
                        (len(train_loader.dataset) / args.batch_size) - step) / 60,),
                  end = '          ')

            # Gradient norm clipping
            if args.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    net.parameters(),
                    args.grad_norm_clip
                )

            optim.step()

        # logging for tensorBoard
        writer.add_scalar("train_loss_each_epoch", loss_sum / train_images, epoch)
        time_end = time.time()
        elapse_time = time_end - time_start
        print('Finished in {}s'.format(elapse_time))
        epoch_finish = epoch + 1

        # Logging
        logfile.write(
            'Epoch: ' + str(epoch_finish) +
            ', Loss: ' + str(loss_sum / len(train_loader.dataset)) +
            ', Lr: ' + str([group['lr'] for group in optim.param_groups]) + '\n' +
            'Elapsed time: ' + str(int(elapse_time)) +
            ', Speed(s/batch): ' + str(elapse_time / step) +
            '\n\n'
        )
        # Eval
        if epoch_finish >= args.eval_start:
            print('Evaluation... {}'.format(fluctuate_count))
            accuracy = evaluate(net, eval_loader, args)
            print('Accuracy :' + str(accuracy))
            # logging for tensorBoard
            writer.add_scalar("test_accuracy", accuracy, epoch)
            eval_accuracies.append(accuracy)
            if accuracy >= best_eval_accuracy:
                fluctuate_count = 0
                # Best
                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.state_dict(),
                    'args': args,
                }
                torch.save(
                    state,
                    args.output + "/" + args.name +
                    '/best' + str(args.seed) + '.pkl'
                )
                best_eval_accuracy = accuracy
                early_stop = 0
            elif fluctuate_count < 20:
                fluctuate_count += 1
            elif decay_count < args.lr_decay_times:
                fluctuate_count = 0
                # Decay
                print('LR Decay...')
                decay_count += 1
                net.load_state_dict(torch.load(args.output + "/" + args.name +
                                               '/best' + str(args.seed) + '.pkl')['state_dict'])
                # adjust_lr(optim, args.lr_decay)
                for group in optim.param_groups:
                    group['lr'] *= args.lr_decay
            else:
                # Early stop
                early_stop += 1
                if early_stop == args.early_stop:
                    logfile.write('Early stop reached' + '\n')
                    print('Early stop reached')
                    logfile.write('best_overall_acc :' + str(best_eval_accuracy) + '\n\n')
                    print('best_eval_acc :' + str(best_eval_accuracy) + '\n\n')
                    os.rename(args.output + "/" + args.name +
                              '/best' + str(args.seed) + '.pkl',
                              args.output + "/" + args.name +
                              '/best' + str(best_eval_accuracy) + "_" + str(args.seed) + '.pkl')
                    logfile.close()
                    return eval_accuracies
    writer.close()
    print("---------------------------------------------")
    print("Best evaluate accuracy:{}".format(best_eval_accuracy))


def evaluate(net, eval_loader, args):
    accuracy = 0
    all_num = 0
    net.train(False)  # 和net.eval()效果一样,使得dropout和BatchNorm层参数被完全冻结
    with torch.no_grad():
        for step, (ids, x, ans) in enumerate(eval_loader):
            if torch.cuda.is_available():
                x = x.cuda()
            evidences, evidence_all, loss = net(x, ans, step)
            _, predicted = torch.max(evidence_all.data, 1)
            accuracy += (predicted == ans).sum().item()
            all_num += ans.size(0)
        accuracy = accuracy / all_num
    net.train(True)  # 和net.train()效果一样，恢复训练模式
    return 100 * np.array(accuracy)
