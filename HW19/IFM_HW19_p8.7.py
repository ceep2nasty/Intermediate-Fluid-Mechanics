# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:05:23 2025

@author: coled_agkeohi
"""

# IFM HW19 Problem 8.6/7
import numpy as np
import matplotlib.pyplot as plt


# W(z) = 1/z^2
def phi_field(x,y): 
    num = x**2 - y**2
    den = (x**2 + y**2)**2
    return num/den


def u(x,y):
    num = 2*x**3 - 6*x*y**2
    den = (x**2 + y **2)**3
    return num/den

def v(x,y):
    num = 2*y**3 - 6*x**2*y
    den = (x**2 + y **2)**3
    return num/den

def p_field(u, v):
    return -(1/2)*(u**2 + v**2)

def a_x(x, y):
    return -12*x / ((x**2 + y**2)**4)   # -12 x / r^8

def a_y(x, y):
    return -12*y / ((x**2 + y**2)**4)   # -12 y / r^8

def invar_field(x, y):
    r2 = x**2 + y**2
    return 36.0 / (r2**4)               # J2 = 36 / r^8  (positive)

# grid
L1 = -2
L2 = 2
N = 1000

xx = np.linspace(L1, L2, 400)
yy = np.linspace(L1, L2, 400)
XX, YY = np.meshgrid(xx, yy, indexing = 'xy')
R2 = (XX**2 + YY**2)

# mask any origin singularities
with np.errstate(divide = 'ignore', invalid = 'ignore'):
    UU = u(XX, YY)
    VV = v(XX,YY)
    PHI = phi_field(XX, YY)
    P = p_field(UU, VV)
    A_X = a_x(XX, YY)
    A_Y = a_y(XX, YY)
    I = invar_field(XX, YY)
    
    
# ----- mask singular region + any non-finite values -----
r0 = 0.50
mask = ((R2 < r0**2) | ~np.isfinite(UU) | ~np.isfinite(VV) | ~np.isfinite(PHI))       

r02 = 0.9
mask2 = ((R2 < r02**2) | ~np.isfinite(P)| ~np.isfinite(A_X) | ~np.isfinite(A_Y))

U_plot   = np.where(mask,  np.nan, UU)
V_plot   = np.where(mask,  np.nan, VV)
PHI_plot = np.where(mask,  np.nan, PHI)
P_plot   = np.where(mask2, np.nan, P)
A_X_plot = np.where(mask2, np.nan, A_X)
A_Y_plot = np.where(mask2, np.nan, A_Y)
I_plot   = np.where(mask, np.nan, I)


# ----- plot: quiver + equipotentials on left; filled phi on right -----
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
# ----- invariant over velocity vectors -----
fig2, ax2 = plt.subplots(1,1, figsize=(8,8))




# Left: quiver  + equipotential contours
step = 12
axs[0].quiver(XX[::step, ::step], YY[::step, ::step],
              U_plot[::step, ::step], V_plot[::step, ::step], scale=100)
cs = axs[0].contour(XX, YY, PHI_plot, levels=20, colors='k', linewidths=1)
axs[0].clabel(cs, inline=True, fontsize=7, fmt="%.2f")
axs[0].set_title("Velocity vectors + equipotentials")
axs[0].set_aspect('equal', 'box')
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')

#right: acceleration + isobar contours
step = 12
axs[1].quiver(XX[::step, ::step], YY[::step, ::step],
              A_X_plot[::step, ::step], A_Y_plot[::step, ::step], scale=250)
cs = axs[1].contour(XX, YY, P_plot, levels=20, colors='k', linewidths=1)
axs[1].clabel(cs, inline=True, fontsize=7, fmt="%.2f")
axs[1].set_title("Acceleration vectors + isobars")
axs[1].set_aspect('equal', 'box')
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')



ax2.quiver(XX[::step, ::step], YY[::step, ::step],
              U_plot[::step, ::step], V_plot[::step, ::step], scale=200)
cs2 = ax2.contour(XX, YY, I_plot, levels=20, colors='k', linewidths=1, linestyle = '--')
plt.show()