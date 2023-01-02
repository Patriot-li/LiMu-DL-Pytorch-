#Batch Gradient Descent
import numpy as np
import matplotlib.pyplot as plt


def y_hat(x):
    return x*w

def loss(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        cost = (y_hat(x) - y)**2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad = 2*x*(w*x - y)
    return grad / len(xs)

x_data = [1, 2, 3]
y_data = [2, 4, 6]
w = 1.0
a = 0.01
epoch_list = []
loss_list = []
for epoch in range(100):
    loss_scalar = loss(x_data, y_data)
    w = w - a*gradient(x_data, y_data)
    epoch_list.append(epoch)
    loss_list.append(loss_scalar)
    print("w=", w, end = '\n')
    print("loss=", loss_scalar, end = '\n')

plt.plot(epoch_list, loss_list)
plt.show()
