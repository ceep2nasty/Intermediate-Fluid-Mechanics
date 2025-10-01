# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 12:15:42 2025

@author: coled_agkeohi
"""

# IFM HW6 Problem 3.6

import numpy as np
import matplotlib.pyplot as plt

A = np.array([[-1, 4],
              [-4, -1]])

x0 = np.array([-2,1])



def x_of_t(t):
    c, s = np.cos(4*t), np.sin(4*t)
    R = np.array([[c, s], 
                  [-s, c]]) *np.exp(-t)
    
    return R @ x0

def v_field(x):
    return A @ x

def a_field(x):
    return A @ x_of_t(t)

def v_of_t(t):
    return A @ x_of_t(t)

def a_of_t(t):
    return A @ v_of_t(t)

L = 3.0
n = 21
xs = np.linspace (-L, L, n)
ys = np.linspace(-L,L, n)
XX, YY = np.meshgrid(xs, ys)

U = -XX +4*YY
V = -4*XX - YY

plt.figure(figsize=(6,6))
plt.quiver(XX, YY, U, V, color='k', angles='xy', scale_units='xy', scale=18, width=0.003)
plt.title('Velocity Vector Field  v(x) = A x')
plt.xlabel('x1'); plt.ylabel('x2'); plt.axis('equal'); plt.xlim(-L,L); plt.ylim(-L,L)
plt.grid(alpha=0.3)
plt.show()

# acceleration field
A2 = A @ A
U_a = A2[0,0]*XX + A2[0,1]*YY
V_a = A2[1,0]*XX + A2[0,1]*YY


plt.figure(figsize=(6,6))
plt.quiver(XX, YY, U_a, V_a, color='k', angles='xy', scale_units='xy', scale=100, width=0.003)
plt.title('Acceleration Vector Field  a(x) = A * Ax')
plt.xlabel('x1'); plt.ylabel('x2'); plt.axis('equal'); plt.xlim(-L,L); plt.ylim(-L,L)
plt.grid(alpha=0.3)
plt.show()

# pathline

T = 5.0 
t = np.linspace(0,T,400)
X = np.vstack([x_of_t(tt) for tt in t])
plt.figure(figsize=(6,6))
plt.quiver(XX, YY, U, V, color='0.85', angles='xy', scale_units='xy', scale=18, width=0.002)  # faint field
plt.plot(X[:,0], X[:,1], lw=2)
plt.plot([x0[0]], [x0[1]], 'o', label='t=0')
plt.title('Pathline from x(0)=(-2, 1)^T')
plt.xlabel('x1'); plt.ylabel('x2'); plt.axis('equal'); plt.xlim(-L,L); plt.ylim(-L,L)
plt.grid(alpha=0.3); plt.legend()
plt.show()

# ---------- 4) Speed and acceleration magnitude vs time ----------
V_path = np.vstack([v_of_t(tt) for tt in t])
A_path = np.vstack([a_of_t(tt) for tt in t])
speed = np.linalg.norm(V_path, axis=1)
amagn = np.linalg.norm(A_path, axis=1)

plt.figure(figsize=(7,4))
plt.plot(t, speed, lw=2)
plt.title('Speed |v(t)|'); plt.xlabel('t'); plt.ylabel('|v|'); plt.grid(alpha=0.3)
plt.show()

plt.figure(figsize=(7,4))
plt.plot(t, amagn, lw=2)
plt.title('Acceleration Magnitude |a(t)|'); plt.xlabel('t'); plt.ylabel('|a|'); plt.grid(alpha=0.3)
plt.show()
