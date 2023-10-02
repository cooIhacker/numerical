import math

import numpy as np
import matplotlib.pyplot as plt
import pylab

a = 2.1  # Объявляю переменые
b = 3.3
alf = 0.2
values=[]
#valuesikf=np.array()
wolfdiff=[]

def momentum(k):
    return ((b - a) ** (k + 1 - alf) / (k + 1 - alf))
def f(x):
    return 4.5 * np.cos(7 * x) * np.exp((-2 * x) / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3

def p(x):
    return 1 / (pow(x - a, alf))
                                                                    #Точность сумм Pимана 10^(-5)
count_points=[100,1000,10000,100000]
arr=[1,2,3,4]
const=[3.52048]*len(arr)
for n in count_points:
    integral_sum=0
    h = (b - a) / (n - 1)
    for i in range(1, n + 1):  # Через суммы римана
        integral_sum += f(a + (i - 0.5) * h) * p(a + (i - 0.5) * h)
    integral_sum *= h
    values.append(integral_sum)
    wolfdiff.append(math.log10(abs(integral_sum-3.52048)))


    # ikf_sum=0
    # u = []  # Формирование вектора и матрцы для решения СЛАУ
    # for i in range(n):
    #     u.append(momentum(i))
    # u = np.array(u).reshape((n, 1))
    # # print(u)
    # matrix = []
    # ikf_points = np.linspace(0, b - a, n)
    #
    # # print(ikf_points)
    # for i in range(n):
    #     matrix.append(ikf_points ** i)
    # matrix = np.array(matrix).reshape(n, n)
    # ans = np.linalg.solve(matrix, u)
    # # print(ans)
    # for i in range(n):
    #     ikf_sum += ans[i] * f(ikf_points[i])
    # valuesikf.append(ikf_sum)


print(values)
plt.plot(arr,wolfdiff, color='blue', marker="*", )  # Вывод графика
plt.title("График функции")
plt.legend(['func'], fontsize="x-large")
#plt.plot(arr,const,color='red')

plt.show()