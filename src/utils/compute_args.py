import torch


def compute_args(args):
    # DataLoader

    if args.dataset == "Caltech101_7" or args.dataset == "Caltech101_20" \
            or args.dataset == "Reuters" or args.dataset == "NUSWIDEOBJ":
        args.dataloader = 'Multiview_Dataset'
    elif args.dataset == "MIMIC":
        args.dataloader = 'Mimic_Dataset'
    elif args.dataset == "UCI":
        args.dataloader = "UCI_Dataset"
    # Loss function to use
    # if args.dataset == 'MOSEI' and args.task == 'sentiment':
    #     args.loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum")
    # if args.dataset == 'MOSEI' and args.task == 'emotion':
    #     args.loss_fn = torch.nn.BCEWithLogitsLoss(reduction = "sum")
    # if args.dataset == 'MELD':
    #     args.loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum")

    # if args.dataset == 'MOSEI':
    #     args.pred_func = "amax"
    # if args.dataset == 'MOSEI' and args.task == "emotion":
    #     args.pred_func = "multi_label"
    # if args.dataset == 'MELD':
    #     args.pred_func = "amax"
    # if args.dataset == "UCI":
    #     args.pred_func = "amax"

    return args
