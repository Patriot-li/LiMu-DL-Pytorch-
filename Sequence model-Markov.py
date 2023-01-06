import torch
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt

def load_array(X, batch_size, is_train = True):
    Data = data.TensorDataset(*X)  #Dataset相当于一个小仓库把数据（数据和标签）一对一对整理好
    return data.DataLoader(Data, batch_size, shuffle = is_train) #加载数据，与dataset配套使用 dataload是数据采集具有可迭代性

T = 1000
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,)) #,不可以省略
plt.plot(range(1, 1001), x)


##-----基于马尔科夫假设的序列预测模型-----##
tau = 4
features = torch.zeros((T -tau, tau))
for i in range(tau):
    features[:, i] = x[i:T - tau + i] #这行将序列样本转换成矩阵非常巧妙
labels = x[tau:].reshape((-1, 1))

batch_size, n_train = 16, 600
train_iter = load_array((features[:n_train,:], labels[:n_train]), batch_size, is_train=True)

def init_weights(m):
    '''初始化权重函数'''
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def get_net():
    '''一个简单的多层感知机'''
    net = nn.Sequential(nn.Linear(4, 10), nn.ReLU(), nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss(reduction='none')

def train(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(epochs):
        for X, y in train_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.sum().backward()
            trainer.step()
        print(f'epoch {epoch + 1}, 'f'loss: {l.sum()}')

net = get_net()
train(net, train_iter, loss, 10, 0.01)

'''用训练好的模型进行单步预测'''
onestep_preds = net(features).reshape(-1)
plt.plot(time[tau:], onestep_preds.detach().numpy())
plt.show()

'''用训练好的模型进行多步预测'''
multi_preds = torch.zeros(T)
multi_preds[ : n_train] = x[ : n_train]
for i in range(n_train, T):
    multi_preds[i] = net(multi_preds[i-tau : i])

plt.plot(range(1, 1001), x)
plt.plot(range(1, 1001), multi_preds.detach().numpy())
plt.show()

'''自定义预测步数'''
pred_step = 4
features = torch.zeros((T - tau - pred_step + 1, tau + pred_step)) #列数好确定先确定列数，然后倒推行数，行数由最后一个数往前推
for i in range(tau):
    features[:, i] = x[i:i + T - tau - pred_step + 1]
for i in range(tau, tau + pred_step):
    features[:, i] = net(features[:, i - tau : i]).reshape(-1)

multi_preds = torch.zeros(T - tau - pred_step + 1)
multi_preds = features[:, tau + pred_step - 1]
plt.plot(range(1, 1001), x)
plt.plot(time[tau + pred_step - 1:T], multi_preds.detach().numpy())
plt.show()

##-------------------------------##

