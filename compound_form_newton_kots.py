import math
import numpy as np
import matplotlib.pyplot as plt

A = 2.5  # Объявляю переменые
B = 2.9
alf = 0.4
h=0.3
n=3
wolfdiff=[]
const=4.461512705

def momentum(k,b,a):
    return (b ** (k + 1 - alf)-a ** (k + 1 - alf) ) / (k + 1 - alf)
def f(x):
    return 4.5 * np.cos(7 * x) * np.exp((-2 * x) / 3) + 1.4 * np.sin(1.5 * x) * np.exp(-x / 3) + 3

def p(x):
    return 1 / (pow(x - A, alf))

ikf_sum=0
points1=np.linspace(A,B,math.ceil((B-A)/h))
# print(points1)
for j in range(len(points1)-1):
    prom_sum=0
    a=points1[j]
    b=points1[j+1]

    points = np.linspace(a, b, n)
    u = []  # Формирование вектора и матрцы для решения СЛАУ
    for i in range(n):
        u.append(momentum(i,b-A,a-A))
    u = np.array(u).reshape((n, 1))
    #print(u)
    matrix = []
    ikf_points = np.linspace(0, b - a, n)
    for i in range(n):
        matrix.append(ikf_points ** i)
    matrix = np.array(matrix).reshape(n, n)
    print(matrix,u)
    ans = np.linalg.solve(matrix, u)
    for i in range(n):
        ikf_sum += ans[i] * f(points[i])
        prom_sum += ans[i] * f(points[i])
    print(ans)
    print(prom_sum)
    print(points)
    # print(u)
    print(ikf_sum)