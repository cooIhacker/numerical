import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sys
sys.float_info

a = 2.1  # Объявляю переменые
b = 3.3
alf = 0.4
values=[]
wolfdiff=[]
sumU=[]
def momentum(k):
    return ((b - a) ** (k + 1 - alf) / (k + 1 - alf))
def f(x):
    return 4.5 * np.cos(7 * x) * np.exp((-2 * x) / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3

def p(x):
    return 1 / (pow(x - a, alf))

count_points=[10]
const=[4.461512705331193]*len(count_points)

for n in count_points:
    ikf_sum=0
    points = np.linspace(a, b, n)

    u = []  # Формирование вектора и матрцы для решения СЛАУ
    for i in range(n):
        u.append(momentum(i))
    u = np.array(u).reshape((n, 1))
    # print(u)
    matrix = []
    ikf_points = np.linspace(0, b - a, n)

    # print(ikf_points)
    for i in range(n):
        matrix.append(ikf_points ** i)
    matrix = np.array(matrix).reshape(n, n)
    ans = np.linalg.solve(matrix, u)
    print(matrix, u)
    sabs,nabs=0,0
    for i in ans:
        sabs+=abs(i)
        nabs+=i

    sumU.append(sabs)

    for i in range(n):
        ikf_sum += ans[i] * f(points[i])
    print(points, ans)
    print(ikf_sum)
    values.append(ikf_sum)
    wolfdiff.append(math.log10(abs(ikf_sum-4.461512705)))                      #Все работает вроде даже на 20 эн считает более менее норм

swrd=[]
for i in count_points: swrd.append(str(i))
plt.subplot(2,2,1)
plt.plot(swrd,wolfdiff, color='blue', marker="*", )  # Вывод графика
plt.title("График функции")
plt.legend(['func'], fontsize="x-large")

plt.subplot(2,2,2)
plt.plot(swrd,values, color='blue', marker="*", )

plt.subplot(2,2,3)
plt.plot(swrd,sumU, color='blue', marker="*", )
plt.title("Сумма коэфицентов")
plt.show()