#!/usr/bin/env python
# Ethan Atkins
# Using NumPy, SciPy, matplotlib
# Approach:
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def ODE(x, y):
    return -(y) + np.log(x)

def RKF(x, y, h):
    newX, newY = 0.0, 0.0
    k = ODE(x, y)
    kk = ODE(x + h/2, y + ((h * k)/2))
    kkk = ODE(x + h/2, y + ((h * kk)/2))
    kkkk = ODE(x + h, y + ((h * kkk)))
    newY = y + (h * (1/6) * (k + (2 * kk) + (2 * kkk) + kkkk))
    newX = x + h
    return newX, newY

n, x, y, h = 0, 2.0, 1.0, 0.3
results = []
xR = []
yR = []

while n <= 1000:
    results.append([n, x, y, ODE(x, y)])
    xR.append(x)
    yR.append(y)
    r = RKF(x, y, h)
    x, y = r[0], r[1]
    n += 1

for r in results:
    print(f"--------------------{r[0]}--------------------\n")
    print(f"x{r[0]} = {r[1]}")
    print(f"y{r[0]} = {r[2]}")
    print(f"Solution = {r[3]}\n")

t_span = (1, 300)
ini_con = [0]
sol = solve_ivp(ODE, t_span, ini_con, t_eval=np.linspace(1, 300, 300))
solT = sol.t
solY = sol.y[0]

plt.figure(figsize = (10,6))
plt.plot(xR, yR, 'bo-', label = 'RKF')
plt.plot(solT, solY, 'g^-', label = 'solve_ivp')
plt.title('-y + ln(x)')
plt.legend()
plt.grid(True)
plt.show()
