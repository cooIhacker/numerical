import math
import numpy as np
import matplotlib.pyplot as plt

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

count_points=[10,15,25,35,50,100]
arr=[1,2,3,4,5,6]
const=[3.52048]*len(arr)

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
    # sabs=0
    # for i in u:
    #     sabs+=abs(i)
    # print(sabs)
    print(sum(u), n , sep='   ')
    for i in range(n):
        ikf_sum += ans[i] * f(points[i])
    wolfdiff.append(math.log10(abs(ikf_sum-3.52048)))                      #Все работает вроде даже на 20 эн считает более менее норм

swrd=[]
for i in count_points: swrd.append(str(i))
plt.plot(swrd,wolfdiff, color='blue', marker="*", )  # Вывод графика
plt.title("График функции")
plt.legend(['func'], fontsize="x-large")
#plt.plot(arr,const,color='red')
plt.annotate(r'$\lim_{x\to 0}\frac{\sin(x)}{x}= 1$', xy=[0,1],xycoords='data',
             xytext=[30,30],fontsize=16, textcoords='offset points', arrowprops=dict(arrowstyle="->",
             connectionstyle="arc3,rad=.2"))


plt.show()