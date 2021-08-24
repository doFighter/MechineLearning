#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/24 12:47
# @Author : doFighter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
# x_data = np.random.random([10, 3]) * 100
# y_data = np.random.random(10) * 100
# dataframe = pd.DataFrame(x_data, y_data)
# # header=False 由于只存数据，不需要表头
# dataframe.to_csv("data1.cvs", header=False)

# 获取数据
data = np.genfromtxt("data1.cvs", delimiter=',')
x_data = data[:, 0:-1]
y_data = data[:, -1]
# 给数据增加一列，其值全为1，位置在第一列
x = np.c_[np.ones(x_data.shape[0]).T, x_data]


def RegressWithStandardEquation(X, Y):
    Theta = np.linalg.inv(np.dot(X.T, X)).dot(X.T).dot(Y)
    return Theta


res = RegressWithStandardEquation(x, y_data)
print(res)
# plt.scatter(x_data, y_data)
# plt.plot(x, res[1] * x + res[0], 'r')
# plt.show()

