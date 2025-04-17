import torch
from torch import nn
import torchvision.transforms as transforms


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        #卷积层
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=1),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2),#池化

                                  nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2),
                                  nn.ReLU(),
                                  nn.MaxPool2d(kernel_size=3, stride=2),

                                  nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1),
                                  nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
                                  nn.MaxPool2d(kernel_size=3, stride=2),

                                  nn.Flatten())
        #全连接层
        self.fc =nn.Sequential(nn.Linear(in_features=256 * 5 * 5, out_features=4096),
                               nn.ReLU(),
                               nn.Linear(in_features=4096, out_features=4096),
                               nn.ReLU(),
                               nn.Linear(in_features=4096, out_features=4096),
                               nn.ReLU(),
                               nn.Linear(in_features=4096, out_features=10))
    def forward(self, x):
        x_flatten = self.conv(x)
        return self.fc(x_flatten)

x = torch.randn(16, 1, 224, 224)
model = AlexNet()
y = model(x)
print(y.shape)

trans = transforms.Compose([transforms.Resize((224, 224))])
x = torch.randn(16, 1, 28, 28)
print(trans(x).shape)