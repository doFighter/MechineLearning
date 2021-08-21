#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/20 15:37
# @Author : doFighter


class perceptron(object):
    """
    感知器类
    """
    def __init__(self):
        """
        初始化相关参数
        """
        self.w1 = 0
        self.w2 = 0
        self.b = -0.6
        self.lr = 0.1

    def setLearningRate(self, lr):
        """
        学习率设置函数
        :param lr: 学习率想要设置的值
        :return:
        """
        self.lr = lr

    def output(self, num):
        """
        自定义的规则函数
        :param num: 传入需要计算的值
        :return:
        """
        if num <= 0:
            return 0
        else:
            return 1

    def train(self, x, target, train_max=1000):
        """
        训练函数
        :param x: 输入值
        :param target: 目标值
        :param train_max: 最大训练次数
        :return:
        """
        y = target.copy()
        for j in range(train_max):
            for i in range(len(x)):
                y[i] = self.output(self.w1 * x[i][0] + self.w2 * x[i][1] + self.b)
            if y == target:
                break
            else:
                for i in range(len(target)):
                    self.w1 = self.w1 + self.lr * (target[i] - y[i]) * x[i][0]
                    self.w2 = self.w2 + self.lr * (target[i] - y[i]) * x[i][1]

    def scatter(self, x):
        """
        预测函数
        :param x: 需要预测的值
        :return:
        """
        for i in range(len(x)):
            res = self.output(self.w1 * x[i][0] + self.w2 * x[i][1] + self.b)
            print(res)
        print("w1=", self.w1, "   w2=", self.w2, "  b=", self.b)


test_num = [[0, 0],
            [1, 0],
            [0, 1],
            [1, 1]]
target_num = [0, 0, 0, 1]

perceptron = perceptron()

perceptron.train(test_num, target_num)
perceptron.scatter(test_num)
