import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from learn.learn_model import * # learn_model与lear_train要在同一个文件夹底下

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="../data", train=True, transform=torchvision.transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.CIFAR10(root="../data", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)

# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为: {}".format(train_data_size))
print("测试数据集的长度为: {}".format(test_data_size))

# dataloader
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
tudui = Tudui()

# loss function
loss_fn = nn.CrossEntropyLoss() # 分类问题 交叉熵

# optimizer
# 0.01=1e-2=1*(10)^(-2)
learning_rate = 0.01
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_train") # 不能../

for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i+1))
    # 训练步骤开始
    # tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数: {}, loss: {}".format(total_train_step, loss.item())) # item: 将数据类型转换成真的数字
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试步骤开始
    # tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad(): # 没有梯度，保证不会对梯度调优，只需测试不需优化
        for data in test_dataloader:
            imgs, targets = data
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试上的loss: {}".format(total_test_loss))
    print("整体测试上的正确率: {}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    # 保存模型
    torch.save(tudui, "tudui_{}.pth".format(i)) # 保存每一轮的结果，区分文件名，否则每次会被覆盖
    # 第二种保存方式（官方推荐）: torch.save(tudui.state_dict(), "tudui_{}.pth".format(i))

writer.close()

# tudui.train() 网络中有特殊的层才需要调用
# tudui.eval()
