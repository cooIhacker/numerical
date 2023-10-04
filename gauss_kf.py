import math
import numpy as np
import matplotlib.pyplot as plt

a = 2.1  # Объявляю переменые
b = 3.3
alf = 0.4
arr=[]
wolfdiff=[]
count_points=[2,3,5,7,10,12,100,150,200]
const=4.461512705
ctest=[12]
def momentum(k):
    return ((b - a) ** (k + 1 - alf) / (k + 1 - alf))
def f(x):
    return 4.5 * np.cos(7 * x) * np.exp((-2 * x) / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3

def p(x):
    return 1 / (pow(x - a, alf))
for n in count_points:
    arr.append(str(n))
    u,matrix=[],[]              #Решаем СЛАУ получаем коэффиценты узлового многочлена
    for i in range(2*n):
        u.append(momentum(i))
    v=-1*np.array(u[n::]).reshape(n,1)
    u=np.array(u).reshape(2*n,1)
    for i in range(n):
        for j in range(n): matrix.append(u[i+j])
    matrix = np.array(matrix).reshape(n, n)
    ans = np.linalg.solve(matrix, v).reshape(1,n)
    a1=[]
    for i in range(n):
        a1.append(ans[0][i])
    a1.append(1.)
    w=np.roots(a1[::-1])
    matrix1=[]
    for i in range(n):
        matrix1.append(w ** i)
    matrix1 = np.array(matrix1).reshape(n, n)
    u1=np.array(u[:n:]).reshape(n,1)
    ans1 = np.linalg.solve(matrix1, u1).reshape(n, 1)
    gauss_sum=0
    for i in range(n):
        gauss_sum += ans1[i] * f(a+w[i])
    print(sum(u1))
    #print(gauss_sum)
    wolfdiff.append(math.log10(abs(gauss_sum - const)))

plt.plot(arr,wolfdiff, color='blue', marker="*", )  # Вывод графика
plt.title("График функции")
plt.legend(['func'], fontsize="x-large")
plt.show()