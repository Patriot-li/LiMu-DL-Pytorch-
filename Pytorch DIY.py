import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
X = torch.rand(2, 20)
Y = net(X)
print(X)
print(Y)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module): #自定义
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module #是一个顺序字典，继承自nn.Module

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X


net = MySequential(nn.Linear(20,256), nn.ReLU(), nn.Linear(256, 10))
Y = net(X)
print(X)
print(Y)

class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand(20, 20, requires_grad = False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight))
        X = self.linear(X)

        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

print("------------------")
X = torch.rand(2, 20)
net = FixedHiddenMLP()
print(net(X))

