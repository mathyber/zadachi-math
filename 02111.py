# import numpy as np
# import math as math
# import matplotlib.pyplot as plt

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

x = np.array([0.5, 0.6, 0.8, 0.9, 1.2, 1.4, 1.5])
# y = np.array([0, 0.51, 0.84, 0.99, 0.96, 0.75, 0.36])
y = np.array([0.36, 0.51, 0.64, 0.75, 0.84, 0.91, 0.96])

A = np.vstack([x, np.ones(len(x))]).T

m, c = np.linalg.lstsq(A, y)[0]
k1 = np.linalg.lstsq(A, y)[1]
print("a = ", m)
print("b = ", c)
print(m, c)


import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, m * x + c, 'r', label='Fitted line')
# plt.legend()
plt.show()

from pylab import *
from scipy.linalg import *

# задаем вектор m = [x**2, x, E]
m = vstack((x ** 2, x, ones(7))).T
# находим коэффициенты при составляющих вектора m
s = lstsq(m, y)[0]

# на отрезке [-5,5]
x_prec = linspace(0, 2, 101)
# рисуем теоретическую кривую x<sup>2</sup>
# plot(x_prec,x_prec**2,'--',lw=2)
# рисуем точки
plt.plot(x, y, 'D')
# рисуем кривую вида y = ax<sup>2</sup> + bx + c, подставляя из решения коэффициенты s[0], s[1], s[2]
plt.plot(x_prec, s[0] * x_prec ** 2 + s[1] * x_prec + s[2], '-', lw=2)
plt.show()
print('a = ', s[0])
print('b = ', s[1])
print('c = ', s[2])
k2 = 0
def f(x):
    return s[0] * x ** 2 + s[1] * x + s[2]
n = 6
while n >= 0:
    k2 = k2 + (y[n] - f(x[n])) ** 2
    n = n - 1
print("k1 = ", k1[0])
print('k2 = ', k2)
