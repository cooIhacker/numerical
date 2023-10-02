# f(x) = 4.5 cos(7x) exp(−2x/3) + 1.4 sin(1.5x) exp(−x/3) + 3
import numpy as np
import matplotlib.pyplot as plt

a = 2.1  # Объявляю переменые
b = 3.3
alf = 0.2
beta = 0
n = 20
h = (b - a) / (n - 1)
integral_sum = 0
ikf_sum = 0


def f(x):
    return 4.5 * np.cos(7 * x) * np.exp((-2 * x) / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3


def momentum(k):
    return ((b - a) ** (k + 1 - alf) / (k + 1 - alf))


def p(x):
    return 1 / (pow(x - a, alf))


points = np.linspace(a, b, n)  # Узлы и значения функции в узлах
values = []
pointsmid=[]
for i in range(1,len(points)):
    values.append(f(points[i]-0.5*h)*p(points[i]-0.5*h))
    pointsmid.append(points[i]-0.5*h)
#print((values))

for i in range(1, n + 1):  # Через суммы римана
    integral_sum += f(a + (i - 0.5) * h)*p(a + (i - 0.5) * h)
integral_sum *= h

plt.plot(pointsmid, values, color='blue', marker="*", )  # Вывод графика
plt.title("График функции")
plt.legend(['func'], title=integral_sum, fontsize="x-large")
plt.fill_between(pointsmid, values)
#plt.show()





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
# print(ans)
for i in range(n):
    ikf_sum += ans[i] * f(points[i])
print(ikf_sum)                          #Все работает вроде даже на 20 эн считает более менее норм


#КФ типа Гаусса

# ug = []  # Формирование вектора и матрцы для решения СЛАУ
# for i in range(2*n):
#     ug.append(momentum(i))
# ug = np.array(ug).reshape((2*n, 1))
#
# matg=[]
# for i in range(n):
#     matg.append(ikf_points ** i)
# matg = np.array(matg).reshape(n, n)
# ansg = np.linalg.solve(matg, ug)