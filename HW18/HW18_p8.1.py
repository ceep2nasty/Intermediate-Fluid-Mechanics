# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 20:03:39 2025

@author: coled
"""

# IFM HW 18


import numpy as np
import matplotlib.pyplot as plt

#8.1

# complex potential
L = 1.5
N = 401
x = np.linspace(-L, L, N)
y =  np.linspace(-L, L, N)
X, Y = np.meshgrid( x, y, indexing="xy")


Z = X + 1j*Y
W = Z **4

phi = np.real(W)
psi = np.imag(W)

u = 4*X**3 - 12*X*Y**2
v = 4*Y**3 - 12*X**2*Y

dx = x[1] - x[0]
dy = y[1] - y[0]
ux, uy = np.gradient(u, dx, dy, edge_order = 2)
vx, vy = np.gradient(v, dx, dy, edge_order = 2)

# plots



fig, axs = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)

# 1) Equipotential + streamlines (contours)
cs1 = axs[0].contour(X, Y, phi, levels=15)
cs2 = axs[0].contour(X, Y, psi, levels=15, linestyles="--")
axs[0].clabel(cs1, inline=True, fontsize=8)
axs[0].clabel(cs2, inline=True, fontsize=8)
axs[0].set_title(r"Equipotentials $\phi$ (solid) & Streamlines $\psi$ (dashed) & Isobars & Velocity Vectors")
axs[0].set_xlabel("x"); axs[0].set_ylabel("y"); axs[0].set_aspect("equal", "box")

# 2) Velocity quiver
skip = 12
axs[1].quiver(X[::skip, ::skip], Y[::skip, ::skip],
              u[::skip, ::skip], v[::skip, ::skip], pivot="mid", scale=100)
axs[1].set_title("Velocity field $(u,v)$")
axs[1].set_xlabel("x"); axs[1].set_ylabel("y"); axs[1].set_aspect("equal", "box")


plt.show()