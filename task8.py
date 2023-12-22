#Явный метод Рунге-Кутты
import math
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


A,B,C=3,-3,3
c2=0.35

def cor_y(x):
    return np.array([
        math.e**(B*math.sin(x**2)),
        math.e**(math.sin(x**2)),
        C*math.sin(x**2)+A,
        math.cos(x**2)]
    )


print(cor_y(math.sqrt(math.pi/2)))


def f(x,y):
    return np.array([
        2*x*pow(y[1],1/B)*y[3],
        2*B*x*math.exp(B/C*(y[2]-A))*y[3],
        2*C*x*y[3],
        -2*x*math.log(y[1])]
    )

def runge_kutta_meth(x0,y0,f,h,num_steps):
    a21=c2
    b2=0.5/c2
    b1=1-b2

    for i in range(num_steps):
        K1=f(x0,y0)
        K2=f(x0+c2*h, y0+h*a21*K1)

        x0+=h
        y0+=h*(b1*K1+b2*K2)

    return y0

x0=0
y0=np.array([1,1,A,1])

