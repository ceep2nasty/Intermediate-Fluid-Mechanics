# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 20:05:23 2025

@author: coled_agkeohi
"""

# IFM HW19 Problem 8.6/7
import numpy as np
import matplotlib.pyplot as plt

# -------- fields --------
# W(z) = 1/z^2  ->  phi = (x^2 - y^2)/(x^2+y^2)^2
def phi_field(x, y):
    r2 = x**2 + y**2
    return (x**2 - y**2) / (r2**2)

def u(x, y):
    # keep your u,v; (note: u is the negative of ∂phi/∂x if you derived from phi)
    r2 = x**2 + y**2
    return (2*x**3 - 6*x*y**2) / (r2**3)

def v(x, y):
    r2 = x**2 + y**2
    return (2*y**3 - 6*x**2*y) / (r2**3)

def p_field(u, v):
    # Bernoulli (rho=1): p = -1/2 |u|^2
    return -0.5 * (u**2 + v**2)

def a_x(x, y):
    # irrotational shortcut: a = ∇(1/2 |u|^2) = ∇(2/r^6) = -(12/r^8) [x, y]
    r2 = x**2 + y**2
    return -12.0 * x / (r2**4)   # -12 x / r^8

def a_y(x, y):
    r2 = x**2 + y**2
    return -12.0 * y / (r2**4)   # -12 y / r^8

def invar_field(x, y):
    # J2 = 1/2 D:D = 36 / r^8  (positive)
    r2 = x**2 + y**2
    return 36.0 / (r2**4)

# -------- grid --------
L1, L2 = -2.0, 2.0
N = 400
xx = np.linspace(L1, L2, N)
yy = np.linspace(L1, L2, N)
XX, YY = np.meshgrid(xx, yy, indexing='xy')
R2 = XX**2 + YY**2

# -------- evaluate & mask --------
with np.errstate(divide='ignore', invalid='ignore'):
    UU   = u(XX, YY)
    VV   = v(XX, YY)
    PHI  = phi_field(XX, YY)
    P    = p_field(UU, VV)
    A_X  = a_x(XX, YY)
    A_Y  = a_y(XX, YY)
    I    = invar_field(XX, YY)

# single consistent exclusion radius for all plots
r0 = 0.50
mask = (R2 < r0**2)

U_plot   = np.where(mask, np.nan, UU)
V_plot   = np.where(mask, np.nan, VV)
PHI_plot = np.where(mask, np.nan, PHI)
P_plot   = np.where(mask, np.nan, P)
A_X_plot = np.where(mask, np.nan, A_X)
A_Y_plot = np.where(mask, np.nan, A_Y)
I_plot   = np.where(mask, np.nan, I)

# -------- plots --------
fig, axs = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

# Left: velocity quiver + equipotential contours
step = 12
axs[0].quiver(XX[::step, ::step], YY[::step, ::step],
              U_plot[::step, ::step], V_plot[::step, ::step], scale=200)
cs0 = axs[0].contour(XX, YY, PHI_plot, levels=20, colors='k', linewidths=1)
axs[0].clabel(cs0, inline=True, fontsize=7, fmt="%.2f")
axs[0].set_title("Velocity vectors + equipotentials")
axs[0].set_aspect('equal', 'box')
axs[0].set_xlabel('x'); axs[0].set_ylabel('y')

# Right: acceleration quiver + isobar contours
axs[1].quiver(XX[::step, ::step], YY[::step, ::step],
              A_X_plot[::step, ::step], A_Y_plot[::step, ::step], scale=250)
cs1 = axs[1].contour(XX, YY, P_plot, levels=20, colors='k', linewidths=1)
axs[1].clabel(cs1, inline=True, fontsize=7, fmt="%.2f")
axs[1].set_title("Acceleration vectors + isobars")
axs[1].set_aspect('equal', 'box')
axs[1].set_xlabel('x'); axs[1].set_ylabel('y')

# Invariant over velocity vectors — USE MASKED INVARIANT + LOG LEVELS
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))

finite = np.isfinite(I_plot)
# cap extreme core to keep contour spread sane
cap_val = np.nanpercentile(I_plot[finite], 99.5)
I_cap = np.where(I_plot > cap_val, cap_val, I_plot)

# log-spaced levels on positive field
lo = np.nanpercentile(I_cap[finite], 10)
hi = np.nanmax(I_cap)
# guard against degeneracy
lo = max(lo, 1e-12)
levels = np.geomspace(lo, hi, 12)

cs2 = ax2.contour(XX, YY, I_cap, levels=levels, linewidths=0.8)
ax2.clabel(cs2, inline=True, fontsize=7, fmt="%.2g")

ax2.quiver(XX[::step, ::step], YY[::step, ::step],
           U_plot[::step, ::step], V_plot[::step, ::step], scale=200)

ax2.set_aspect('equal', 'box')
ax2.set_title(r'Contours of $J_2=\tfrac12 D\!:\!D$ over velocity field')
ax2.set_xlabel('x'); ax2.set_ylabel('y')

plt.show()
