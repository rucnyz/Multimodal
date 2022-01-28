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
    elif args.dataset == "UKB":
        args.dataloader = "UKB_Dataset"
        args.pred_func = "accuracy_count"
    elif args.dataset == "UKB_AD":
        args.dataloader = "UKB_AD_Dataset"
        args.pred_func = "accuracy_count"
    elif args.dataset == "UKB_All":
        args.dataloader = "UKB_ALL_Dataset"
        args.pred_func = "accuracy_count"
    elif args.dataset == "UKB_Balanced":
        args.dataloader = "UKB_BALANCED_Dataset"
        args.pred_func = "accuracy_count"
    # cuda 电脑的 GPU 能否被 PyTorch 调用
    # args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = torch.device("cpu")
    # Loss Function to use
    if args.dataset == "Caltech101_7" or args.dataset == "Caltech101_20" \
            or args.dataset == "Reuters" or args.dataset == "NUSWIDEOBJ" \
            or args.dataset == "UCI" or args.dataset == "UKB" or args.dataset == "UKB_AD"\
            or args.dataset == "UKB_All" or args.dataset == "UKB_Balanced":
        args.loss_fn = "AdjustedCrossEntropyLoss"
        args.lambda_epochs = 50
    # 模型选择
    if args.model == "CPM":
        args.lsd_dim = 128
        args.optim_cca = "MixAdam"
    elif args.model == "TMC":
        args.optim_cca = "Adam"
    elif args.model == "CPM_GAN":
        args.lsd_dim = 128
        args.optim_cca = "GAN_Adam"
    # 训练集补充缺失数据，-1代表不补充，则设置GAN_start>max_epoch即可
    if args.GAN_start == -1:
        args.GAN_start = args.max_epoch + 1
    return args
