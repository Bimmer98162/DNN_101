#VGG视觉几何组(Visual Geometry Group)
import torch
from torch import nn


def vgg_block(num_conv: int, in_channels: int, out_channels: int) -> nn.Module:
    """VGG块构建函数"""
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch: list) -> nn.Module:
    """ VGG网络构建函数 """
    conv_block = []
    in_channels = 1
    # 卷积部分
    for (num_conv, out_channels) in conv_arch:
        conv_block.append(vgg_block(num_conv, in_channels, out_channels))
        in_channels = out_channels

    # 全连接部分
    model = nn.Sequential(*conv_block,
                          nn.Flatten(),

                          nn.Linear(7 * 7 * out_channels, 4096),
                          nn.ReLU(),
                          nn.Linear(4096, 4096),
                          nn.ReLU(),
                          nn.Linear(4096, 10))
    return model

def count_parameters(model: nn.Module):
    """ 计算模型参数量函数 """
    return sum(p.numel() for p in model.parameters())

#print(vgg_block(5,1,3))

conv_arch_16 = [(2, 64), (2, 128), (3, 256), (3, 512), (3, 512)]
x = torch.rand(16, 1, 224, 224)
model = vgg(conv_arch_16)
output = model(x)
#print(output.shape)
#print(count_parameters(model))
#print(model)