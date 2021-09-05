import torch


def compute_args(args):
    # DataLoader
    if args.dataset == "Caltech101_7" or args.dataset == "Caltech101_20" \
            or args.dataset == "Reuters" or args.dataset == "NUSWIDEOBJ":
        args.dataloader = 'Multiview_Dataset'  # 类的名字！直接用eval()可以避免复制代码！
        args.pred_func = "accuracy_count"
    elif args.dataset == "MIMIC":
        args.dataloader = 'Mimic_Dataset'
    elif args.dataset == "UCI":
        args.dataloader = "UCI_Dataset"
        args.pred_func = "accuracy_count"
    # cuda 电脑的 GPU 能否被 PyTorch 调用
    if torch.cuda.is_available():
        args.device = torch.device("cuda:0")
    else:
        args.device = torch.device("cpu")
    # Loss function to use
    if args.dataset == "Caltech101_7" or args.dataset == "Caltech101_20" \
            or args.dataset == "Reuters" or args.dataset == "NUSWIDEOBJ" \
            or args.dataset == "UCI":
        args.loss_fn = "AdjustedCrossEntropyLoss"
        args.lambda_epochs = 50
    # 模型选择
    if args.model == "CPM":
        args.lsd_dim = 128
        args.optim = "MixAdam"
    elif args.model == "TMC":
        args.optim = "Adam"

    return args
