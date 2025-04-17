import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from VGG import vgg


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize((224, 224))])
mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download = True)
mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download = True)

train_dataloader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=64, shuffle=False)

conv_arch_16 = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]
model = vgg(conv_arch_16).to(device)

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

plt.figure(figsize=(12,8))
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test loss")
plt.title("VGG Model loss")
plt.grid(True)
plt.legend()
plt.show()