# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 17:37:59 2025

@author: coled_agkeohi
"""

import numpy as np
import matplotlib.pyplot as plt

# Domain
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

# Stream function
psi = X*Y + Y**2

# Velocity field
u = X + 2*Y
v = -Y

# Acceleration field
a_x = X
a_y = Y



# --- Plot Velocity vector field over streamlines ---
plt.figure(figsize=(6,5))
plt.streamplot(X, Y, u, v, density=1, color=np.hypot(u,v), cmap='plasma')
plt.title("Velocity Field")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(0, 0, 'ro', label="Stagnation point")  # stagnation point
plt.legend()
plt.gca().set_aspect('equal')
plt.savefig("IFM_HW30_fig1.png")
plt.show()

# --- Plot Acceleration field over streamlines ---
skip = 10
plt.figure(figsize=(6,5))
plt.contour(X, Y, psi, levels=30, cmap='viridis')
plt.quiver(X[::skip, ::skip], Y[::skip, ::skip], a_x[::skip, ::skip], a_y[::skip, ::skip], scale = 20)
plt.title("Acceleration Field  a = (x, y)")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(0, 0, 'ro', label="Zero acceleration")  # same as stagnation point
plt.legend()
plt.gca().set_aspect('equal')
plt.savefig("IFM_HW30_fig2.png")
plt.show()


