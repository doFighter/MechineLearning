#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/20 15:37
# @Author : doFighter

import numpy as np

class S_neuron(object):
    """
    感知器类
    """
    def __init__(self):
        """
        初始化相关参数
        """
        self.w = np.zeros(3)
        self.lr = 0.1
        self.border = 0.5

    def setLearningRate(self, lr):
        """
        学习率设置函数
        :param lr: 学习率想要设置的值
        :return:
        """
        self.lr = lr

    def setBorder(self, border):
        """
        分类边界值
        :param border: 传入想要设定的 border
        :return:
        """
        self.border = border

    def output(self, num):
        """
        自定义的规则函数
        :param num: 传入需要计算的值
        :return:
        """
        delta = 1 / (1 + np.exp(-num))
        delta[delta > self.border] = 1
        delta[delta <= self.border] = 0
        return delta

    def train(self, x, target, train_max=1000):
        """
        训练函数
        :param x: 输入值
        :param target: 目标值
        :param train_max: 最大训练次数
        :return:
        """
        for j in range(train_max):
            y = self.output(self.w.dot(x.T))
            if (y == target).all():
                break
            else:
                self.w += np.dot(self.lr * (target - y), x)/len(x)

    def scatter(self, x):
        """
        预测函数
        :param x: 需要预测的值
        :return:
        """
        res = self.output(self.w.dot(x.T))
        print(res)
        print("w0=", self.w[0], "   w1=", self.w[1], "  w2=", self.w[2])


test_num = [[1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1]]

target_num = [0, 0, 0, 1]

test_num = np.array(test_num)
target_num = np.array(target_num)

Sneuron = S_neuron()

Sneuron.setBorder(0.3)
Sneuron.train(test_num, target_num)
Sneuron.scatter(test_num)
