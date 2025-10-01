# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 21:16:02 2025

@author: coled_agkeohi
"""

# IFM HW10 Problem 5.3

# Write an equation for the temperature field

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# create symbols


x, y = sp.symbols('x y')
T = x*(x-1)*y*(y-1)
k = sp.Matrix([[2,1],
               [1,2]])

#symbolic derivatives

Tx = sp.diff(T,x)
Ty = sp.diff(T,y)

# gradient of T and heat flux vector

gradT = sp.Matrix([[Tx],
                   [Ty]])

q = -k * gradT
qx, qy = q[0], q[1]
q_N = sp.diff(qx, x) + sp.diff(qy, y)


# irreversibility rate

I = qx * gradT[0] + qy * gradT[1]
# convert using lambdify

f_T = sp.lambdify((x,y), T, 'numpy')
f_Tx = sp.lambdify((x,y), Tx, 'numpy')
f_Ty = sp.lambdify((x,y), Ty, 'numpy')
f_qx = sp.lambdify((x,y), q[0, 0], 'numpy')
f_qy = sp.lambdify((x,y), q[1, 0], 'numpy')
f_bigQ = sp.lambdify((x,y), q_N, 'numpy')
f_I = sp.lambdify((x,y), I, 'numpy')


# establish some domain

nx = ny = 101

x = np.linspace(-0.2, 1.2, nx)
y = np.linspace(-0.2, 1.2, ny)
XX, YY = np.meshgrid(x,y)

# find relevant values over those points

TT = f_T(XX,YY)
TX = f_Tx(XX,YY)
TY = f_Ty(XX, YY)
QX = f_qx(XX,YY)
QY = f_qy(XX,YY)
bigQ = f_bigQ(XX,YY)
II = f_I(XX,YY)

# plot helper funcs

def add_quiver(ax, step=4, scale=75):
    ax.quiver(XX[::step, ::step], YY[::step, ::step])
# plot the temperature field

fig1, ax1 = plt.subplots()
c1 = ax1.contourf(XX,YY, TT, levels=25)
plt.colorbar(c1,ax=ax1, label="T")
ax1.contour(XX,YY,TT, levels=[0], linewidths=1)
ax1.set_title("Temperature field")
ax1.set_xlabel("x"); ax1.set_ylabel("y")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')

# Plot temp field as surface instead of contour
surf = ax4.plot_surface(XX, YY, TT,
                        cmap="viridis",    # color map
                        edgecolor="none",  # cleaner surface
                        alpha=0.9)

fig4.colorbar(surf, ax=ax4, shrink=0.5, aspect=10, label="T")

ax4.set_title("Temperature field T(x,y)")
ax4.set_xlabel("x")
ax4.set_ylabel("y")
ax4.set_zlabel("T")


# heat flux as quiver plot

fig2, ax2 = plt.subplots()
ax2.set_title("Heat flux vector field", fontsize = 20, fontweight = "bold")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
step = 4
ax2.quiver(XX[::step, ::step], YY[::step, ::step], QX[::step, ::step], QY[::step, ::step], scale = 20, scale_units = "xy")


# heat source field

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection = '3d')
surf2 = ax3.plot_surface(XX, YY, bigQ,
                         cmap = "viridis",
                         edgecolor = "none",
                         alpha = 0.9)
fig3.colorbar(surf2, ax=ax3, shrink = 0.5, aspect=10, label="Q")
ax3.set_title("Heat source field", fontsize = 20, fontweight = "bold")
ax3.set_xlabel("x")
ax3.set_ylabel("y")


# irreversibility production rate field as a contour plot
fig5, ax5 = plt.subplots()

# filled contour for I-dot
c5 = ax5.contourf(XX, YY, II, levels=25, cmap="viridis")
fig5.colorbar(c5, ax=ax5, label="I dot")

# quiver overlay
step = 6
ax5.quiver(XX[::step, ::step], YY[::step, ::step],
            QX[::step, ::step], QY[::step, ::step],
            scale=10, scale_units="xy", angles="xy",
            color="black", width=0.003)

# titles and labels
ax5.set_title("Irreversibility production rate", fontsize=16, fontweight="bold")
ax5.set_xlabel("x")
ax5.set_ylabel("y")
ax5.set_aspect("equal", adjustable="box")


plt.show()