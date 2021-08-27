#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time : 2021/8/27 11:12
# @Author : doFighter


import pandas as pd
import numpy as np
# 必须从 sklearn 中引入 linear_model
from sklearn import linear_model

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
# 创建模型
model = linear_model.LassoCV()
model.fit(x_data, y_data)
# 输出结果
print(model.alpha_)
print(model.coef_)
