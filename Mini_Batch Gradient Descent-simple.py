'''Mini_Batch Gradient Descent'''


import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torch import nn

#生成数据集
def synthetic_dataset(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))    #输出一个tensor 每个元素都是从均值为0标准差为1的独立正态分布中提取，形状：行数为样本数、列数为参数个的矩阵
    y = torch.matmul(X, w) + b    #X和w矩阵乘法
    y += torch.normal(0, 0.01, y.shape)     #加入噪声
    return X, y.reshape((-1, 1))   #调整X,y的形状，注意-1的含义(其他维度先定-1所在维度最后再定)

ture_w = torch.tensor([2, -3.4])
ture_b = 4.2
features, labels = synthetic_dataset(ture_w, ture_b, 1000) #X是一个（1000，2）的tensor y是（1000，1）的矩阵

plt.scatter(features[:,(1)].detach().numpy(), labels.detach().numpy(), 1)
'''注意tensor的切片操作，【，，】中逗号分隔的是对不同维度的处理，从0到1到2
   [:,(1)]表示0维全部选取，1维只选取序号为1的那一列，
   注意标号是从0开始的
'''
plt.show()

#读取数据并提取mini_batch
def load_array(X, batch_size, is_train = True):
    Data = data.TensorDataset(*X)  #Dataset相当于一个小仓库把数据（数据和标签）一对一对整理好
    return data.DataLoader(Data, batch_size, shuffle = is_train) #加载数据，与dataset配套使用 dataload是数据采集具有可迭代性

data_batch = load_array((features, labels), batch_size= 10)
print(type(data_batch))

#定义网络(moudel)
network = nn.Sequential(nn.Linear(2, 1, bias = True))
#sequential就网络的一个框架壳子，这里我们在里面放入了一个线性层Linear（输入参数个数，输出参数个数，是否有bia）注意b不算做输入参数。

#初始化参数
network[0].weight.data.normal_(0,0.01)    #用正态分布填充
network[0].bias.data.fill_(0)   #用0填充

#定义损失函数
loss = nn.MSELoss() #已经自动完成求和 #给函数重命名
#定义优化方法(optimizer)
trainer = torch.optim.SGD(network.parameters(), lr = 0.03)
i = 0
I = []
L = []
epoch = 10
#训练过程
for n_epoch in range(epoch):
    for X, y in data_batch:
        l = loss(network(X), y) #network()相当于按模型框架进行一侧forward计算，可得到预测值
        trainer.zero_grad()
        l.backward() #自动求导
        trainer.step() #让模型进行更新,我们使基于一定的优化方法来更新参数，所以前面是optimizer
        I.append(i)
        i += 1
        L.append(l.detach().numpy())


    print(loss(network(features),labels))


plt.plot(I, L)
plt.show()
print(network[0].weight.data)