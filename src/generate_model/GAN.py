# -*- coding: utf-8 -*-
# @Time    : 2021/9/12 10:25
# @Author  : nieyuzhou
# @File    : GAN.py
# @Software: PyCharm
import torch
from torch import nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    # noinspection PyTypeChecker
    def __init__(self, in_channels = 3, channels = 32):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(channels * 2),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 4, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(channels * 4),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(channels * 4, channels * 8, kernel_size = 4, stride = 2, padding = 1),
            nn.BatchNorm2d(channels * 8),
            nn.LeakyReLU(0.2, inplace = True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1, 256 * 6 * 6)
        out = self.fc(out)
        return out


class Generator(nn.Module):
    def __init__(self, feature_size = 100, in_channels = 1024, channels = 128, out_channels = 3):
        super(Generator, self).__init__()
        self.in_channels = in_channels
        self.linear = nn.Linear(feature_size, in_channels * 6 * 6)

    def forward(self, x):
        out = self.linear(x)
        out = out.view(-1, self.in_channels, 6, 6)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


def train(discriminator, generator, criterion, d_optim, g_optim, epochs, dataloader, print_every = 10):
    iter_count = 0
    for epoch in range(epochs):
        for real_inputs in dataloader:
            real_inputs = real_inputs.to(device)  # 真图片
            fake_inputs = generator(torch.randn(real_inputs.size(0), 100).to(device))  # 生成假图片
            real_labels = torch.ones(real_inputs.size(0)).to(device)  # 真标签
            fake_labels = torch.zeros(real_inputs.size(0)).to(device)  # 假标签

            # 训练判别器
            d_output_real = discriminator(real_inputs).view(-1)  # 鉴别真图片
            d_loss_real = criterion(d_output_real, real_labels)  # 真图片损失
            d_output_fake = discriminator(fake_inputs.detach()).view(-1)  # 鉴别假图片
            d_loss_fake = criterion(d_output_fake, fake_labels)  # 假图片损失
            d_loss = d_loss_fake + d_loss_real  # 计算总损失
            d_optim.zero_grad()  # 判别器梯度清零
            d_loss.backward()  # 反向传播
            d_optim.step()  # 更新鉴别器参数

            # 训练判别器
            fake_inputs = generator(torch.randn(real_inputs.size(0), 100).to(device))  # 生成假图片
            g_output_fake = discriminator(fake_inputs).view(-1)  # 鉴别假图片
            g_loss = criterion(g_output_fake, real_labels)  # 假图片损失
            g_optim.zero_grad()  # 生成器梯度清零
            g_loss.backward()  # 反向传播
            g_optim.step()  # 更新鉴别器参数
            if iter_count % print_every == 0:
                print('Epoch:{}, Iter:{}, D:{:.4}, G:{:.4}'.format(epoch, iter_count, d_loss.item(), g_loss.item()))
            iter_count += 1
        torch.save(generator.state_dict(), 'g_' + str(epoch))


if __name__ == '__main__':
    # 测试一下GAN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = Discriminator().apply(weights_init).to(device)  # 定义鉴别器
    g = Generator().apply(weights_init).to(device)  # 定义生成器
    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(d.parameters(), lr = 0.0003)
    g_optimizer = torch.optim.Adam(g.parameters(), lr = 0.0003)
