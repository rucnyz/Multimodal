import torch
from torch.utils.tensorboard import SummaryWriter
from torch import nn


# writer = SummaryWriter("logs")
# # writer.add_image()
#
# for i in range(100):
#     writer.add_scalar("y=x",i,i)
#
# writer.close()
# # --port=6007(打开新的端口)

class Demo(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


demo = Demo()
x = torch.tensor(1.0)
out = demo(x) # 不需要使用forward，只要在实例化一个对象中传入对应的参数就可以自动调用 forward 函数
print(out)
