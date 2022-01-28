import torch

import argparse
import random
from torch.utils.data import DataLoader
import pandas

from train import *
from utils.compute_args import compute_args
from utils.preprocess import *
from utils.make_optim import *
from utils.loss_func import AdjustedCrossEntropyLoss
from predict_model.model_TMC import TMC
from predict_model.model_CPM import CPM
from predict_model.CPM_GAN import CPM_GAN
from dataset.UCI_dataset import UCI_Dataset
from dataset.UKB_dataset import UKB_Dataset
from dataset.multi_view_dataset import Multiview_Dataset
from dataset.UKB_ad_dataset import UKB_AD_Dataset
from dataset.UKB_all_dataset import UKB_ALL_Dataset
from dataset.UKB_balanced_dataset import UKB_BALANCED_Dataset


# python中函数参数直接传地址，不涉及形参问题，所有的args都是一样的！

def parse_args():
    # argparse 模块是 Python 内置的一个用于命令项选项与参数解析的模块，argparse 模块可以让人轻松编写用户友好的命令行接口。
    # 通过在程序中定义好我们需要的参数，然后 argparse 将会从 sys.argv 解析出这些参数。
    # argparse 模块还会自动生成帮助和使用手册，并在用户给程序传入无效参数时报出错误信息。

    # 创建一个解析器——创建 ArgumentParser() 对象
    parser = argparse.ArgumentParser()

    # 添加参数——调用 add_argument() 方法添加参数
    # Model
    # default: 默认值，如果没有输入该变量则直接设为默认值
    parser.add_argument('--model', type = str, default = "CPM_GAN", choices = ["CPM", "CPM_GAN", "TMC"])
    # Training
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'mymodel_40/')
    parser.add_argument('--batch_size', type = int, default = 200)  # 200
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--max_epoch', type = int, default = 400)
    parser.add_argument('--lr_base', type = float, default = 0.0003)
    parser.add_argument('--lr_decay', type = float, default = 0.005)
    parser.add_argument('--lr_decay_times', type = int, default = 2)
    parser.add_argument('--grad_norm_clip', type = float, default = -1)
    parser.add_argument('--early_stop', type = int, default = 3)
    parser.add_argument('--seed', type = int, default = random.randint(0, 9999999))
    parser.add_argument('--loop_times', type = int, default = 1)
    parser.add_argument('--GAN_start', type = int, default = 200)
    parser.add_argument('--sp', type = int, default = 1)
    # Dataset
    parser.add_argument('--dataset', type = str,
                        choices = ['Caltech101_7', 'Caltech101_20', 'Reuters', 'NUSWIDEOBJ', 'MIMIC', 'UCI', 'UKB',
                                   'UKB_AD', 'UKB_All', 'UKB_Balanced'],
                        default = 'UCI')
    parser.add_argument('--missing_rate', type = float, default = 0,
                        help = 'view missing rate [default: 0]')
    # record
    parser.add_argument('--log', type = bool, default = False)
    parser.add_argument('--save_net', type = bool, default = False)

    # 解析参数——使用 parse_args() 解析添加的参数
    return parser.parse_args()


if __name__ == '__main__':
    # change working directory to the project root
    if os.getcwd().endswith("src"):
        os.chdir("../")
    # "./"：代表目前所在的目录
    # " . ./"代表上一层目录
    # "/"：代表根目录

    # Base on args given, compute new args
    args = compute_args(parse_args())
    # parse_args: 刚定义，return args
    # compute_args: utils.compute_args当中的函数, 根据数据集设置dataloader、pred_func和loss_func

    # Seed 保证每次运行网络的时候相同输入的输出是固定的,保证可复现性
    torch.manual_seed(args.seed)  # 设置CPU生成随机数的种子，方便下次复现实验结果
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True  # 每次返回的卷积算法将是确定的，即默认算法
    torch.backends.cudnn.benchmark = False  # 方便复现、提升训练速度

    # Dataset
    # eval: 返回传入字符串的表达式的结果
    train_dset = eval(args.dataloader)('train', args)  # args.dataloader类的构造函数
    eval_dset = eval(args.dataloader)('valid', args)

    # Generate missing views
    dataroot = os.path.join(os.getcwd() + '/data' + '/ukb_data')
    if args.dataset == 'UKB_All':
        missing_index = pickle.load(open(dataroot + "/missing_index_all.pkl", "rb"))
        print("missing_rate = " + str(sum(sum(missing_index))/(missing_index.shape[0] * missing_index.shape[1])))
    else:
        missing_index = get_missing_index(args.views, args.num, args.missing_rate)

    # Preprocess data with missing views and load data
    missing_data_process(args, train_dset, eval_dset, missing_index)

    # DataLoader: DataSet的打包
    train_loader = DataLoader(train_dset, args.train_batch_size, num_workers = args.num_workers, shuffle = True,
                              pin_memory = False)
    eval_loader = DataLoader(eval_dset, args.valid_batch_size, num_workers = args.num_workers, pin_memory = False)
    # batch_size:一次运行多少个sample
    # shuffle是打乱顺序: True两次顺序不同，False两次顺序相同，默认为false
    # num_workers: 采用多进程进行加载，默认为0，即逐进程；>0在windows下会出现错误
    # drop_last: 总sample除以batch_size除不尽的时候True舍弃
    # pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
    # ctrl+p可以查看参数

    # Net
    net = eval(args.model)(args)  # choices = ["CPM", "CPM_GAN", "TMC"]
    net.to(args.device)  # 是否用GPU，将模型加载到相应的设备中

    # Loss function
    loss_fn = eval(args.loss_fn)(args)
    loss_fn.to(args.device)

    # Optimizer
    optim = eval(args.optim_cca)(net, args.lr_base)
    print("Total number of parameters : " + str(sum([p.numel() for p in net.parameters()]) / 1e3) + "k")

    # Create Checkpoint dir
    if not os.path.exists(os.path.join(args.output, args.name)):
        os.makedirs(os.path.join(args.output, args.name))

    # Run training
    if args.model == "TMC":
        train1(net, loss_fn, optim, train_loader, eval_loader, args)
    elif args.model == "CPM":
        train2(net, optim, train_loader, eval_loader, args)
    elif args.model == "CPM_GAN":
        train(net, optim, train_loader, eval_loader, args)
