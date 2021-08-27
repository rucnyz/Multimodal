import torchvision
import torch
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Ha(nn.Module):
    def __init__(self):
        super(Ha, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
            # 再经过softmax才是概率
        )

    def forward(self, x):
        x = self.model1(x)
        return x


h = Ha()
# input = torch.ones((64, 3, 32, 32))
# output = h(input)
# print(output.shape)
#
# writer = SummaryWriter("logs_seq")
# writer.add_graph(h, input)
# writer.close()

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(h.parameters(), lr=0.01) # 学习速率不宜过大（不稳定），不宜过小（太慢），可以前面大，后面小
for epoch in range(20): # epoch: 一轮
    running_loss=0.0
    for data in dataloader: # 只对数据进行了一轮学习
        imgs, targets = data
        outputs = h(imgs)
        # print(outputs)
        # print(targets)
        result_loss = loss(outputs, targets)
        optim.zero_grad() # 梯度一定要调0，否则下一次循环会出问题
        result_loss.backward() # 计算出grad梯度
        optim.step() # 对模型参数进行调优
        running_loss += result_loss
    print(running_loss)

# grad: 梯度

