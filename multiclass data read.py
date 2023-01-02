import torch
import torchvision #有关计算机视觉处理的库
from torch.utils import data
from torchvision import transforms

# trans = transforms.ToTensor() #使用totensor转变数据类型
# mnist_train = torchvision.datasets.FashionMNIST(root = "D:\Working Apps\projects\dataset", train = True, transform = trans, download = True)
# mnist_test = torchvision.datasets.FashionMNIST(root = "D:\Working Apps\projects\dataset", train = False, transform = trans, download = True)
# #这里的数据集都是dataset类型，一个样本包括数据和标签，因此都可用dataload加载
#
# def get_fashion_mnist_labels(labels): #将数据集中的数字labels装换成文本labels
#     text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
#                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
#     return [text_labels[int(i)] for i in labels]
#
# train_iter = data.DataLoader(mnist_train, bacth_size = 256, shuffle = True)
#
# for X, y in train_iter:
#     continue #此循环只用于迭代数据

#获取和读取Fashion-MNIST数据集。 这个函数返回训练集和验证集的数据迭代器
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

train_iter, test_iter = load_data_fashion_mnist(batch_size , resize = None) #做一个一次提取一个batchsize的迭代器

def get_fashion_mnist_labels(labels): #将数据集中的数字labels装换成文本labels
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

for X, y in train_iter:
    print(X.shape,X.dtype)
    break



