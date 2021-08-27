#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/23 10:29
# @Author : doFighter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
# x_data = np.random.random(10) * 100
# y_data = np.random.random(10) * 100
# dataframe = pd.DataFrame({'x_data': x_data, 'y_data': y_data})
# # header=False 由于只存数据，不需要表头
# dataframe.to_csv("data.cvs", header=False, index=False)

# 获取数据
data = np.genfromtxt("data.cvs", delimiter=',')
x_data = data[:, 0]
y_data = data[:, 1]


# 画出数据点的分布
# plt.scatter(x_data, y_data)
# plt.show()

class UnaryLinearRegression(object):
    def __init__(self):
        """
        初始化函数：初始化学习率，斜率a，截距b
        """
        self.lr = 0.0001
        self.a = 0
        self.b = 0

    def setLearningRate(self, lr):
        """
        设置学习率
        :param lr: 设定的学习率的值
        :return:
        """
        self.lr = lr

    def costFunction(self, x, y):
        """
        代价函数
        :param x: 数据点自变量x值
        :param y: 数据点实际y值
        :return: 代价值
        """
        cost = np.sum((y - self.b - self.a * x) ** 2)
        return cost / len(x) / 2

    def gradientDescent(self, x, y):
        """
        梯度下降算法
        :param x: 数据点自变量x值
        :param y: 数据点实际y值
        :return:
        """
        n = len(x)
        gradient_a = -(np.sum(x * (y - self.b - self.a * x)) / n)
        gradient_b = -np.sum(y - self.b - self.a * x) / n
        self.a -= self.lr * gradient_a
        self.b -= self.lr * gradient_b

    def regress(self, x, y, iterate=100):
        """
        回归
        :param x: 数据点自变量x值
        :param y: 数据点实际y值
        :param iterate: 最大迭代次数
        :return:
        """
        for i in range(iterate):
            print("损失数：", self.costFunction(x, y))
            self.gradientDescent(x, y)

            if i % 20 == 0:
                plt.scatter(x, y)
                plt.plot(x, self.a * x + self.b, 'r')
                plt.show()


regress = UnaryLinearRegression()
regress.setLearningRate(0.0001)
regress.regress(x_data, y_data, 100)
