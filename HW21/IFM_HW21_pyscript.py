# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 15:44:17 2025

@author: coled
"""

#IFM HW21: Fanno Flow
import numpy as np
import matplotlib.pyplot as plt


gam =1.4 
M1 = 3
d = 0.01
f = 0.04

def G_M(M):
    num1 = 1- M**2
    den1 = gam*M**2
    t1 = num1/den1
    num2 = (1+gam)*M**2
    den2 = 2 + (gam-1)*M**2
    c = (1+gam)/(2*gam)
    t2 = c *np.log(num2/den2)
    
    return t1+t2

n = 100
M = np.linspace(3,1,n)

x = (d/f)*G_M(M)
L_choke = x[-1]
x = x[-1]-x

plt.plot(x, M)
plt.xlabel("x in m")
plt.ylabel("Mach number, M")
plt.title("Fanno flow plot")

plt.show()
    