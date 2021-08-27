# LASSO回归

## 1 引言

LASSO是由1996年Robert Tibshirani首次提出，全称Least absolute shrinkage and selection operator。该方法是一种压缩估计。它通过构造一个惩罚函数得到一个较为精炼的模型，使得它压缩一些回归系数，即强制系数绝对值之和小于某个固定值；同时设定一些回归系数为零。因此保留了子集收缩的优点，和岭回归一样，是一种处理具有复共线性数据的有偏估计。

LASSO回归使用的是 $L_1$ 正则化代价函数，如下所示：

$$
J(\theta)=\frac{1}{2m}[\sum_{i=1}^m(h_\theta(x^i)-y^i)^2+\lambda\sum_{j=1}^n|\theta_j|]\tag{1}
$$
