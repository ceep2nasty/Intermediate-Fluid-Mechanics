# -*- coding: utf-8 -*-
"""
IFM HW25 – density plots for the homentropic Euler (γ=3) similarity solution
"""

import numpy as np
import matplotlib.pyplot as plt

def rho(x, t):
    A  = t**2 + 1.0
    s2 = A - x**2
    r  = np.zeros_like(x, float)       # vacuum by default
    m  = s2 >= 0.0                     # support: |x| <= sqrt(A)
    r[m] = np.sqrt(s2[m]) / A          # sqrt(A - x^2)/A
    return r

def u(x, t):
    A = t**2 + 1.0
    return (t * x) / A

# --- left subplot: rho(x) at several times ---
x1     = np.linspace(-2.5, 2.5, 2001)
times  = [0.0, 0.5, 1.0, 2.0]

# --- right subplot data: (x,t) grid and rho(x,t) field ---
x2 = np.linspace(-2.5, 2.5, 401)
t2 = np.linspace(0.0,  2.0, 401)
X, T = np.meshgrid(x2, t2)

A   = T**2 + 1.0
S2  = A - X**2
RHO = np.zeros_like(X, float)
mask = S2 >= 0.0
RHO[mask] = np.sqrt(S2[mask]) / A[mask]

# --- figure with two subplots ---
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

# Left: rho(x) profiles
for t in times:
    ax[0].plot(x1, rho(x1, t), label=f"t={t:g}")
ax[0].set_xlabel("x")
ax[0].set_ylabel(r"$\rho(x,t)$")
ax[0].set_title(r"Density profiles $\rho(x,t)$ at selected times")
ax[0].grid(True, alpha=0.3)
ax[0].legend()

# Right: contour of rho(x,t)
cf = ax[1].contourf(X, T, RHO, levels=30)
ax[1].contour(X, T, RHO, levels=10, linewidths=0.6, colors="k", alpha=0.35)

# vacuum boundary x = ±sqrt(t^2+1)
xb = np.sqrt(t2**2 + 1.0)
ax[1].plot(+xb, t2, ls="--", lw=1.2, color="w")
ax[1].plot(-xb, t2, ls="--", lw=1.2, color="w")

ax[1].set_xlabel("x")
ax[1].set_ylabel("t")
ax[1].set_title(r"Contour of $\rho(x,t)$ with vacuum boundary $x=\pm\sqrt{t^2+1}$")

# Colorbar for the contour (attach to right subplot)
cbar = fig.colorbar(cf, ax=ax[1], label=r"$\rho$")

plt.show()

