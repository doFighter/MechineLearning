#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/26 11:48
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


def RidgeRegression(X, Y, lamda):
    # 给数据增加一列，其值全为1，位置在第一列
    X = np.c_[np.ones(X.shape[0]).T, X]
    I = np.eye(X.shape[1])
    Theta = np.linalg.inv(np.dot(X.T, X) + lamda * I).dot(X.T).dot(Y)
    return Theta


res = RidgeRegression(x_data, y_data, lamda=0.02)
print(res)
# plt.scatter(x_data, y_data)
# plt.plot(x, res[1] * x + res[0], 'r')
# plt.show()
