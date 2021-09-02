#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/9/2 8:45
# @Author : doFighter


import numpy as np
import matplotlib.pyplot as plt

# 生成数据，对数据随机分类并保存
# data = np.random.random([20, 2]) * 100
# res = np.round(np.random.random(20))
# data = np.c_[data, res]
# np.savetxt('data3.cvs', data)

# 对不同分类的数据使用不同的颜色画出来
data = np.loadtxt('data3.cvs')
index = data[:, 2]
data1 = data[index == 1][:, 0:2]
data2 = data[index == 0][:, 0:2]
plt.scatter(data1[:, 0], data1[:, 1], c='b')
plt.scatter(data2[:, 0], data2[:, 1], c='r')
# plt.show()

# 将数据读取并划分
x_data = data[:, 0:2]
y_data = data[:, 2]



def sigmoid(x):
    res = 1.0 / (1 + np.exp(-x))
    return res


def LossFunction(y, res):
    cost = (y-1) * np.log(1 - res) - y * np.log(res)
    return np.sum(cost) / len(y)


def LogisticRegression(x, y, iterate_max=1000):
    N = len(y)                  # 获取数据集大小
    x = np.c_[np.ones(N), x]    # 获取增广矩阵
    w = np.zeros(x.shape[1])    # 定义增广权重矩阵
    ita = 0.001                 # 定义学习率
    for i in range(iterate_max):
        res = sigmoid(np.dot(x, w))
        w = w + ita * np.dot((y - res), x) / N
        if i % 100 == 0:
            loss = LossFunction(y, res)
            print(loss)
    return w


w = LogisticRegression(x_data, y_data)
print(w)




