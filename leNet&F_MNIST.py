import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((32, 32))])
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download = True)
print(len(mnist_train), len(mnist_test))


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5),
                                  nn.Sigmoid(),
                                  nn.AvgPool2d(kernel_size=2, stride=2),

                                  nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
                                  nn.Sigmoid(),
                                  nn.AvgPool2d(kernel_size=2, stride=2))
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(nn.Linear(in_features=16 * 5 * 5, out_features=120),
                                nn.Sigmoid(),
                                nn.Linear(in_features=120, out_features=84),
                                nn.Sigmoid(),
                                nn.Linear(in_features=84, out_features=10))

    def forward(self, x):
        feat_map = self.conv(x)
        output = self.flatten(feat_map)
        return self.fc(output)

x = torch.randn(16, 1, 32, 32)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = LeNet().to(device)
y = model(x.to(device))
print(y.shape)

def count_parameters(model: nn.Module):
    """ 计算模型参数量函数 """
    return sum(p.numel() for p in model.parameters())
print(count_parameters(LeNet()))


def train_model(model, train_dataloader, loss_fn, optimizer):
    """模型训练函数"""
    model.train()
    train_loss = 0
    for x, y in train_dataloader:
        y_hat = model(x.to(device))
        loss = loss_fn(y_hat, y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_dataloader)

def test_model(model, test_dataloader, loss_fn):
    """模型测试函数"""
    model.eval()
    total_loss = 0
    for x, y in test_dataloader:
        y_hat = model(x.to(device))
        loss = loss_fn(y_hat, y.to(device))
        total_loss += loss.item()
        return total_loss / len(test_dataloader)
    return None


train_dataloader = DataLoader(mnist_train, batch_size=512, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=512, shuffle=False)
n_epochs = 10
train_loss_list = []
test_loss_list = []
for i in range(n_epochs):
    train_loss = train_model(model, train_dataloader,  loss_fn= nn.CrossEntropyLoss(), optimizer = torch.optim.Adam(model.parameters(), lr=0.001))
    test_loss = test_model(model, test_dataloader, loss_fn=nn.CrossEntropyLoss())
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    print(train_loss)

plt.figure(figsize=(12,8))
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test loss")
plt.title("Model loss")
plt.grid(True)
plt.legend()
plt.show()