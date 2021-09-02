# LASSO回归

## 1 引言

LASSO是由1996年Robert Tibshirani首次提出，全称Least absolute shrinkage and selection operator。该方法是一种压缩估计。它通过构造一个惩罚函数得到一个较为精炼的模型，使得它压缩一些回归系数，即强制系数绝对值之和小于某个固定值；同时设定一些回归系数为零。因此保留了子集收缩的优点，和岭回归一样，是一种处理具有复共线性数据的有偏估计。

LASSO回归使用的是 $L_1$ 正则化代价函数，如下所示：

$$
J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^i)-y^i)^2+\lambda\sum_{j=1}^n|\theta_j|]\tag{1}
$$

> LASSO回归和岭回归的同和异：
>
>- 相同：
都可以用来解决标准线性回归的过拟合问题。
>- 不同：
$LASSO$ 可以用来做 feature selection，而岭回归不行。或者说，LASSO更容易使得权重变为 0，而岭回归更容易使得权重接近 0。
从贝叶斯角度看，LASSO（L1 正则）等价于参数 $w$ 的先验概率分布满足拉普拉斯分布，而 岭回归（L2 正则）等价于参数 $w$ 的先验概率分布满足高斯分布。

## 2 sklearn-LASSO算法

还是以之前的数据，使用 $sklearn$ 中的 LASSO 算法进行回归，代码如下：

```py
import numpy as np
# 必须从 sklearn 中引入 linear_model
from sklearn import linear_model
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
```
