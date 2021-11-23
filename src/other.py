# -*- coding: utf-8 -*-
# @Time    : 2021/11/19 13:26
# @Author  : nieyuzhou
# @File    : other.py
# @Software: PyCharm
import argparse

from torch.utils.data import DataLoader

from utils.pred_func import *
from utils.preprocess import get_missing_index, missing_data_process


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type = str, default = 'ckpt/')
    parser.add_argument('--name', type = str, default = 'exp0/')
    parser.add_argument('--sets', nargs = '+', default = ["valid", "test"])
    parser.add_argument('--num_workers', type = int, default = 0)
    parser.add_argument('--index', type = int, default = 99)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    model_path = "ckpt/mymodel/best123UKB.pkl"
    # Load original args
    state = torch.load(model_path)
    args = state['args']
    # 设置seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Dataset
    # eval: 返回传入字符串的表达式的结果
    train_dset = eval(args.dataloader)('train', args)  # args.dataloader类的构造函数
    eval_dset = eval(args.dataloader)('valid', args)

    # Generate missing views
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