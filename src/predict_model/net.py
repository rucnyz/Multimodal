import torch
from torch import nn


class FeatConcat(nn.Module):
    def __init__(self, args):
        super(FeatConcat, self).__init__()
        # initialize parameter
        self.view_num = args.views
        self.layer_size = sum(args.classifier_dims)
        self.lsd_dim = 128  # args.lsd_dim = 128  # lsd: latent space data
        self.lamb = 1
        self.num = args.num
        self.fc = nn.Sequential(
            nn.Linear(self.layer_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, args.classes),
            nn.Softmax(dim = 1)
        )

    def forward(self, X):
        concat_X = torch.tensor([])
        for i in range(self.view_num):
            concat_X = torch.concat((concat_X, X[i]), dim = 1)
        return self.fc(concat_X)
