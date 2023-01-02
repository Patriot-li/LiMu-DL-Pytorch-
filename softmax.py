import torch
from IPython import display
import torchvision
from torchvision import transforms
from torch.utils import data
def load_data_fashion_mnist(batch_size, resize = None):
    trans = transforms.ToTensor()  # 使用totensor转变数据类型
    mnist_train = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=True,    #读取数据集
                                                    transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=False,
                                                   transform=trans, download=True)
    if resize:
        trans = [transforms.ToTensor()]
        trans.insert(0,transforms.Resize(resize)) #把resize操作插入到transform操作中。
                                                  #注意transform是支持多步骤操作的，把操作步骤放到一个list中就行
        trans = transforms.Compose(trans)
        mnist_train = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=True,
                                                        transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=False,
                                                       transform=trans, download=True)
    if resize == None:
        mnist_train = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=True,
                                                        transform=trans, download=True)
        mnist_test = torchvision.datasets.FashionMNIST(root="D:\Working Apps\projects\dataset", train=False,
                                                       transform=trans, download=True)

    return (data.DataLoader(mnist_train, batch_size , shuffle = True),
            data.DataLoader(mnist_test, batch_size , shuffle= False))

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10

W = torch.normal(0, 1, size = (num_inputs, num_outputs), requires_grad = True)
b = torch.zeros(num_outputs, requires_grad = True)

#计算softmax函数
def softmax(O):
    O_exp = torch.exp(O)
    partition = O_exp.sum(1, keepdim = True)
    return O_exp / partition

def net(X): #输出y_hat
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b) #rashape可以实现降维操作这里将

#定义损失函数
'''y = torch.tensor([0, 2]) 注意y中储存的不是样本的one-hot code，而是正确类标签所在总类别列表中的位置索引
                            此例表示第一个样本的时第一类，第二个样本时第三类
   y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
   y_hat[[0, 1]这是样本序号（行数）, y样本所对应的类别序号] 列表的花式索引
   有了这个思想我们就可以很简单的计算交叉熵
'''
#计算交叉熵
def cross_entropy(y_hat, y):
    return - torch.log(y_hat[[range(len(y_hat))], y]) #由于one-hot code 交叉熵公式虽然时连加但是只有一项不为0

#计算预测正确得数量
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis = 1)  #argmax可以对矩阵运算，需指定按哪一个轴进行运算
    cmp = y_hat.type(y.dtype) == y #cmp是一个布尔向量 #注意dype返回y中数据元素的类型，type返回y的类型
    return float(cmp.sum())

#定义一个累加器类
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n #注意这是对列表的扩展，不是对列表内值得乘法

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):  #实现list得索引功能
        return self.data[idx]

def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_gard():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel()) #numel返回数据个数

    return metric[0] /metric[1]

#训练
def train_epoch(net, train_iter, loss, optimizer):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(optimizer, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            optimizer.zero_grad()
            l.mean().backward()
            optimizer.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            optimizer(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]