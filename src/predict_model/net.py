from torch import nn


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_size, classes):
        super(MultiLayerPerceptron, self).__init__()
        # initialize parameter
        self.lsd_dim = 128  # args.lsd_dim = 128  # lsd: latent space data
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, classes),
        )

    def forward(self, x):
        return self.fc(x)
