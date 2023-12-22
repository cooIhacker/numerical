import math
import numpy as np
import matplotlib.pyplot as plt
import pylab
plt.style.use('ggplot')

a = 2.1  # Объявляю переменые
b = 3.3
alf = 0.4
arr=[]
values=[]
wolfdiff=[]
# count_points=[12,15,18,20,23,50,100]
const=4.461512705331193
count_points=[2,3,4]
sum_U=[]
sum_aU=[]
condm,condm1=[],[]

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
    condm.append(math.log10(np.linalg.cond(matrix)))
    for i in range(n):
        a1.append(ans[0][i])
    a1.append(1.)
    w=np.roots(a1[::-1])
    # print(w)
    # for i in range(len(w)):           #Попытка решить относительно вещественных узлов
    #     w[i]=float(w[i])
    # print(w)
    matrix1=[]
    for i in range(n):
        matrix1.append(w ** i)
    matrix1 = np.array(matrix1).reshape(n, n)
    u1=np.array(u[:n:]).reshape(n,1)
    condm1.append(math.log10(np.linalg.cond(matrix1)))
    ans1 = np.linalg.solve(matrix1, u1).reshape(n, 1)
    gauss_sum=0
    for i in range(n):
        gauss_sum += ans1[i] * f(a+w[i])
    sum_U.append((sum(ans1)))
    sU=[0]*n
    for i in (ans1):
        sU.append(abs(i.real))
    sum_aU.append(sum(sU))
    # print(ans1)
    print(gauss_sum)
    wolfdiff.append(math.log10(abs(gauss_sum - const)))
    values.append(gauss_sum)


plt.subplot(2,2,1)
plt.plot(arr,wolfdiff, color='blue',  )  # Вывод графика
plt.title("График функции")
plt.xlabel("Количество узлов")
plt.ylabel("Порядок погрешности")
plt.legend(['func'], fontsize="x-large")

plt.subplot(2,2,2)
plt.plot(arr,sum_U,color='red')
plt.title("Сумма коэффицентов кф")
plt.xlabel("Количество узлов")
plt.ylabel("Порядок суммы")


plt.subplot(2,2,3)
plt.plot(arr,sum_aU,color='green')
plt.title("Сумма модулей коэффицентов кф")
plt.xlabel("Количество узлов")
plt.ylabel("Порядок суммы")

plt.subplot(2,2,4)
plt.plot(arr,condm,color='purple')
plt.title("Вычесленное значение")
plt.xlabel("Количество узлов")
plt.ylabel("Порядок суммы")
plt.show()

