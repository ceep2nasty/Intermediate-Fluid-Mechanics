# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:31:22 2025

@author: coled
"""

# IFM HW29 P1: generalization of Stokes' First Problem

import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt

def u_fun(y, t):
    # t can be scalar or array, but must be > 0
    return 0.5 * (special.erf((y + 1) / np.sqrt(4.0 * t)) -
                  special.erf((y - 1) / np.sqrt(4.0 * t)))

# spatial grid
N = 200
yy = np.linspace(-5, 5, N)

# time grid: avoid t = 0 to prevent division by zero
M = 200
tt = np.linspace(0.01, 2.0, M)

# --- 1) Plot u(y,t) for a few fixed times ---

plt.figure()

t_samples = [0.01, 0.05, 0.1, 0.5, 1.0]  # pick whatever you want
for t in t_samples:
    u_profile = u_fun(yy, t)
    plt.plot(yy, u_profile, label=f"t = {t:.2f}")

plt.xlabel("y")
plt.ylabel("u(y,t)")
plt.title("Velocity profiles for different times")
plt.legend()
plt.grid(True)
plt.savefig("IFM_HW29_fig1.png", dpi = 300, bbox_inches="tight")
# --- 2) 2D contour plot in (y,t) ---

YY, TT = np.meshgrid(yy, tt)
U = u_fun(YY, TT)

plt.figure()
cp = plt.contourf(YY, TT, U, cmap = 'viridis', levels=50)
plt.colorbar(cp, label="u(y,t)")
plt.xlabel("y")
plt.ylabel("t")
plt.title("Generalized Stokes' problem: u(y,t)")
plt.savefig("IFM_HW29_fig2.png", dpi = 300, bbox_inches="tight")
plt.show()