import argparse
import os
import random
from torch.utils.data import DataLoader

from train import train
from utils.compute_args import compute_args
from utils.preprocess import *
from utils.make_optim import *
from utils.loss_func import AdjustedCrossEntropyLoss
from predict_model.model_TMC import TMC
from predict_model.model_CPM import CPM
from dataset.UCI_dataset import UCI_Dataset
from dataset.multi_view_dataset import Multiview_Dataset


# try to commit tell me why～


def parse_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--model', type = str, default = "TMC", choices = ["CPM", "Model_LAV", "TMC"])
    # Training
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--batch_size', type = int, default = 64)
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--lr_base', type = float, default = 0.0005)
    parser.add_argument('--lr_decay', type = float, default = 0.5)
    parser.add_argument('--lr_decay_times', type = int, default = 2)
    parser.add_argument('--grad_norm_clip', type = float, default = -1)
    parser.add_argument('--early_stop', type = int, default = 3)
    parser.add_argument('--seed', type = int, default = random.randint(0, 9999999))
    # Dataset
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI'],
                        default = 'UCI')
    parser.add_argument('--missing_rate', type = float, default = 0,
                        help = 'view missing rate [default: 0]')
    # record
    parser.add_argument('--log', type = bool, default = False)
    parser.add_argument('--save_net', type = bool, default = False)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # change working directory to the project root
    if os.getcwd().endswith("src"):
        os.chdir("../")

    # Base on args given, compute new args
    args = compute_args(parse_args())

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Dataset
    train_dset = eval(args.dataloader)('train', args)
    eval_dset = eval(args.dataloader)('valid', args)

    # Generate missing views
    missing_index = get_missing_index(args.views, args.num, args.missing_rate)

    # Preprocess data with missing views and load data
    missing_data_process(args, train_dset, eval_dset, missing_index)
    # TODO 这里对比"换缺失数据为-1"和"生成缺失数据"没有意义，因为shuffle导致了两种方法缺失情况不同，但我现在暂时不想改这玩意
    train_loader = DataLoader(train_dset, args.train_batch_size, num_workers = args.num_workers, shuffle = True,
                              pin_memory = True)
    eval_loader = DataLoader(eval_dset, args.valid_batch_size, num_workers = args.num_workers, pin_memory = True)

    # Net
    net = eval(args.model)(args)
    net.to(args.device)

    # Loss function
    loss_fn = eval(args.loss_fn)(args)
    loss_fn.to(args.device)

    # Optimizer
    optim = eval(args.optim)(net, args.lr_base)
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e3) + "k")

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    eval_accuracies = train(net, loss_fn, optim, train_loader, eval_loader,missing_index, args)
