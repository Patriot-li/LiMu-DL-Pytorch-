'''Mini_Batch Gradient Descent'''

import random
import torch
import matplotlib.pyplot as plt

#生成数据集
def synthetic_dataset(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))    #输出一个tensor 每个元素都是从均值为0标准差为1的独立正态分布中提取，形状：行数为样本数、列数为参数个的矩阵
    y = torch.matmul(X, w) + b    #X和w矩阵乘法
    y += torch.normal(0, 0.01, y.shape)     #加入噪声
    return X, y.reshape((-1, 1))   #调整y的形状，注意-1的含义(其他维度先定-1所在维度最后再定)

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
def mini_batch(batch_size, features, labels):
    num_examples = len(features)     #只读取第一个维度参数的长度eg：tensor（1，2，3）len=1，通过此法可读取样本数量
    indices = list(range(num_examples))   #获得每个样本的序号
    random.shuffle(indices)   #将这个序号打乱，也就是将样本打乱,原本的数据集是没变的，只是在提取是的序号是随机的。
    for i in range(0, num_examples, batch_size):   #步长为batch_size确保不会重复提取某一个样本
        batch_indices = torch.tensor(indices[i : min(i + batch_size, num_examples)]) #样本的提取
        yield  features[batch_indices], labels[batch_indices] #tensor的索引操作
'''yelid的作用就是是函数和程序一起迭代，每次执行完yelid后就就会退出函数，下一次再从yelid之后运行。
也就是说，上面的函数中的for每次调用只会迭代一次。下一次调用时再接着迭代。        

for X, y in mini_batch(10, features, labels):
    print(X, '\n', y)
    break #只迭代看看数据即可
'''
#初始化模型参数
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True) #允许对其求导
b = torch.zeros(1, requires_grad = True)

#使用线性回归
def linreg(X, w, b):
    return torch.matmul(X, w) + b

#定义损失函数
def error(y_hat, y):
    return (y_hat - y)**2 / 2

#MBGD法
#训练过程
alpha = 0.01
num_epoch = 3
batch_size = 10
i = 0
I = []
Loss = []
for epoch in range(num_epoch):
    for X, y in mini_batch(batch_size, features, labels):
        l = error(linreg(X, w, b), y)    #注意这里是向量的运算，也就是说minibatch中的所有样本损失都已经并行计算完了
        l.sum().backward() #backward会对之前所有可求导参数求导，并存放在grad中
        loss = float(l.sum())
        i += 1
        I.append(i)
        Loss.append(loss)
        with torch.no_grad(): #参数更新不属于计算图中的步骤，所以从计算图中剔除
             for param in [w, b]:#对参数进行更新
                 param -= alpha*param.grad / batch_size
                 param.grad.zero_() #每次用完都要把梯度清零不然会叠加



print("w=", w, "损失函数=", loss)
plt.plot(I, Loss)
plt.show()