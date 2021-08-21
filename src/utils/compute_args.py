import torch


def compute_args(args):
    # DataLoader
    if args.dataset == "Caltech101_7" or args.dataset == "Caltech101_20" \
            or args.dataset == "Reuters" or args.dataset == "NUSWIDEOBJ":
        args.dataloader = 'Multiview_Dataset'
        args.pred_func = "accuracy_count"
    elif args.dataset == "MIMIC":
        args.dataloader = 'Mimic_Dataset'
    elif args.dataset == "UCI":
        args.dataloader = "UCI_Dataset"
        args.pred_func = "accuracy_count"
    # cuda
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
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
        args.lsd_dim = 150
        args.optim = "MixAdam"
    elif args.model == "TMC":
        args.optim = "Adam"

    return args
