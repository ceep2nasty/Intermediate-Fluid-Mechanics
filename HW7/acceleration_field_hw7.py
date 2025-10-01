# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 18:50:02 2025

@author: coled_agkeohi
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def velocity(x):
    x1, x2, x3 = x
    v1 = -x1 -4*x2
    v2 = 2*x1-x1*x2
    v3 = -x3-x1
    return np.array([v1, v2, v3], dtype=float)

#accel field

def acceleration(x):
    x1, x2, x3 = x
    a1 = -7*x1 + 4*x2 + 4*x1*x2
    a2 = -2*x1 - 8*x2 + 4*x2**2 + x1*x2 - 2*x1**2 + x1**2*x2
    a3 = 2*x1 + 4*x2 + x3
    return np.array([a1, a2, a3], dtype=float)

x1_vals = np.linspace(-3,3,20)
x2_vals = np.linspace(-3,3,30)
x3_vals = np.linspace(-3,3,20)

X1,X2,X3 = np.meshgrid(x1_vals, x2_vals, x3_vals)

A1 = np.zeros_like(X1)
A2 = np.zeros_like(X2)
A3 = np.zeros_like(X3)
A4 = np.zeros_like(X1)
A5 = np.zeros_like(X2)
A6 = np.zeros_like(X3)
for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        for k in range(X1.shape[2]):
            a1 = acceleration([X1[i,j,k], X2[i,j,k], X3[i,j,k]])
            A1[i,j,k], A2[i,j,k], A3[i,j,k] = a1
            a2 = acceleration([X1[i,j,k], X2[i,j,k], 0.0])
            A4[i,j,k],A5[i,j,k],A6[i,j,k] = a2
            
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(111, projection="3d")
ax1.quiver(X1, X2, X3, A1, A2, A3, length=0.2, normalize=True, color = "darkorange")

ax1.set_xlabel("x1")
ax1.set_ylabel("x2")
ax1.set_zlabel("x3")
ax1.set_title("3D acceleration field")

plt.figure(figsize=(6,6))
plt.quiver(X1,X2, A4, A5, angles="xy", color = "darkorange")
plt.axis("equal"); plt.grid(True)
plt.show()

