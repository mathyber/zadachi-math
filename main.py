import numpy as np
import math as math
import matplotlib.pyplot as plt

N = 1
tt2 = [1, 15]
tt3 = [1, 8, 15]
tt4 = [1, 4, 10, 15]

def ptichk(arr):
    matrix = np.zeros((len(arr), len(arr)))
    vector = np.zeros((len(arr)))
    n = 0
    while n < len(arr):
        vector[n] = math.sin(arr[n]/N) * math.exp(arr[n]/(2*N)) + (5 * math.exp(-arr[n]/(N+1)))
        # vector[n] = math.sin(arr[n]/5) * math.exp(arr[n]/10) + (5 * math.exp(-arr[n]/2))
        # vector[n] = math.sin(15) * math.exp(15) + (5 * math.exp(-15/2))
        k = 0
        while k < len(arr):
            matrix[n][k] = arr[n]**k
            k += 1
        n += 1
    return matrix, vector

m, v = ptichk(tt2)

M1 = np.array(m)
v1 = np.array(v)

c1 = np.linalg.solve(M1, v1)

print("Коэффициенты многочлена 1 степени:")
print(c1)
def f1(x):
    return c1[0]+c1[1]*x

m2, v2 = ptichk(tt3)

M2 = np.array(m2)
v2 = np.array(v2)

c2 = np.linalg.solve(M2, v2)
print("Коэффициенты многочлена 2 степени:")
print(c2)
def f2(x):
    return c2[0]+c2[1]*x+c2[2]*x*x

m3, v3 = ptichk(tt4)

M3 = np.array(m3)
v3 = np.array(v3)

c3 = np.linalg.solve(M3, v3)
print("Коэффициенты многочлена 3 степени:")
print(c3)
def f3(x):
    return c3[0]+c3[1]*x+c3[2]*x*x+c3[3]*x*x*x

def ff(x):
    return np.sin(x/N) * np.exp(x / (2*N)) + (5 * np.exp((-x) / (N+1)))

fig, ax = plt.subplots(figsize=(10,5))


ax.grid()
x = np.linspace(1,15)
ax.plot(x, f1(x), color='blue')
ax.plot(x, f2(x), color="red")
ax.plot(x, f3(x), color="green")
ax.plot(x, ff(x), color="black")
plt.show()
