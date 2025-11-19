# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:45:43 2025

@author: coled
"""

# IFM HW29 P3: X momentum pulse in viscous flowfield

import numpy as np
import matplotlib.pyplot as plt  # <-- fix typo: pyplot

def u_fun(y, z, t):
    # y, z can be scalars or arrays; t > 0
    r2 = y**2 + z**2
    return 1.0 / (4.0 * np.pi * t) * np.exp(-r2 / (4.0 * t))

N = 200
ystart, ystop = -5.0, 5.0
zstart, zstop = -5.0, 5.0

y = np.linspace(ystart, ystop, N)
z = np.linspace(zstart, zstop, N)

Y, Z = np.meshgrid(y, z)  # 2D grid in (y,z)

t_array = [1/10, 1/2]

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, t in zip(axes, t_array):
    U = u_fun(Y, Z, t)
    cp = ax.contourf(Y, Z, U, levels=50)
    fig.colorbar(cp, ax=ax, label="u(y,z,t)")
    ax.set_xlabel("y")
    ax.set_ylabel("z")
    ax.set_title(f"t = {t:.2f}")

plt.tight_layout()
plt.savefig("IFM_HW29_fig3.png", dpi = 300, bbox_inches="tight")
plt.show()
    