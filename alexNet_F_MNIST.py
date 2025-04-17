import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lesson4ModernCNN.alexNet import model, AlexNet

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224, 224))])

mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download = True)
print(len(mnist_train), len(mnist_test))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)


train_dataloader = DataLoader(mnist_train, batch_size=640, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=640, shuffle=False)
print(len(train_dataloader), len(test_dataloader))

x, y = next(iter(train_dataloader))
print(x.shape, y.shape)

def count_parameters(model: nn.Module):
    """ 计算模型参数量函数 """
    return sum(p.numel() for p in model.parameters())
print(count_parameters(AlexNet()))

def train_model(model, train_dataloader, loss_fn, optimizer):
    """模型训练函数"""
    model.train()
    train_loss = 0
    for x, y in train_dataloader:
        # x: [bs, 1, 224, 224]
        # y: [batch_size]
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
    y_true = 0
    total_loss = 0
    for x, y in test_dataloader:
        y_hat = model(x.to(device))
        loss = loss_fn(y_hat, y.to(device))

        y_true += (y.to(device) == torch.argmax(y_hat, dim=-1)).sum().item()

        total_loss += loss.item()
    print(f"Acc:{round(y_true / len(test_dataloader.dataset), 3)}")
    return total_loss / len(test_dataloader)

model = AlexNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5
train_loss_list = []
test_loss_list = []
for i in range(n_epochs):
    train_loss = train_model(model, train_dataloader, loss_fn, optimizer)
    test_loss = test_model(model, test_dataloader, loss_fn)
    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)
    print(train_loss)

plt.figure(figsize=[12,8])
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test loss")
plt.title("Model loss")
plt.grid(True)
plt.legend()
plt.show()