# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 18:06:42 2025

@author: coled_agkeohi
"""

# IFM HW16 P3

import numpy as np
import matplotlib.pyplot as plt


# grid
x = np.linspace(-3, 3, 200)
y = np.linspace(-3, 3, 200)
XX, YY = np.meshgrid(x, y, indexing='xy')

# --- mask near the singularity ---
r0 = 0.20                                 # radius of the masked disk
r2 = XX**2 + YY**2
mask = r2 < r0**2                          # True inside the disk

# safe r^2 for division 
r2_safe = np.where(mask, np.nan, r2)

# compute u,v using r2_safe 
u = 1 + 2*YY + 2*YY/r2_safe
v = -2*XX/r2_safe


fig, ax = plt.subplots()

step = 6
ax.quiver(XX[::step, ::step], YY[::step, ::step], u[::step, ::step], v[::step, ::step], scale = 200)
ax.set_aspect('equal', 'box')
ax.axline((0,0), slope=0, color='red', linestyle='--', lw=1.5, label="slip line")
ax.set_title("velocity vector field")
plt.show()

plt.savefig("two_vortices_fields.png", dpi=300, bbox_inches="tight"