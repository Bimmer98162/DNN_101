#Fashion-MNIST图像识别任务（线性神经网络）
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from lesson3ConvolutionNN.fashionMNIST import mnist_train, mnist_test


class LinearClassifier(nn.Module):
    """基于线性神经网络的分类模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(784, 256),#全连接层
                                nn.ReLU(),
                                nn.Linear(256, 10),)
        
    def forward(self, x):
        """前向传播模型"""
        return self.fc(x)
    
model = LinearClassifier()

train_dataloader = DataLoader(mnist_train, batch_size=128, shuffle=True)
test_dataloader = DataLoader(mnist_test, batch_size=128, shuffle=False)

def train_model(model, train_dataloader, loss_fn, optimizer):
    """模型训练函数"""
    model.train()
    train_loss = 0
    for x, y in train_dataloader:
        #x:[batch_size, 1, 28, 28]
        #y:[batch_size]
        batch_size = x.shape[0]
        y_hat = model(x.view(batch_size, -1))
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    return train_loss / len(train_dataloader)

def test_model(model, test_dataloader, loss_fn):
    """模型测试函数"""
    model.eval()
    test_loss = 0
    for x, y in test_dataloader:
        y_hat = model(x.view(x.shape[0], -1))
        loss = loss_fn(y_hat, y)
        test_loss += loss.item()
        return test_loss / len(test_dataloader)
    return None


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