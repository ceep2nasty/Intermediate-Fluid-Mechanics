# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 17:27:00 2025

@author: coled_agkeohi
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# --------------------------
# Velocity induced by two vortices
# --------------------------
def velocity(x, y, x01, y01, x02, y02, Gamma=1.0):
    R1 = (x - x01)**2 + (y - y01)**2
    R2 = (x - x02)**2 + (y - y02)**2
    u1 = -(Gamma/(2*np.pi)) * (y - y01)/R1
    v1 =  (Gamma/(2*np.pi)) * (x - x01)/R1
    u2 = -(Gamma/(2*np.pi)) * (y - y02)/R2
    v2 =  (Gamma/(2*np.pi)) * (x - x02)/R2
    return (u1 + u2), (v1 + v2), (u1, v1), (u2, v2)


# --------------------------
# Potential (multi-valued via atan2)
# --------------------------
def potential(x, y, x01, y01, x02, y02, Gamma=1.0):
    phi1 = (Gamma/(2*np.pi)) * np.arctan2(y - y01, x - x01)
    phi2 = (Gamma/(2*np.pi)) * np.arctan2(y - y02, x - x02)
    return phi1 + phi2

# --------------------------
# Core velocities for equal like-signed pair (rigid rotation)
# Positions at (-a,0), (+a,0) with a = (x02 - x01)/2
# --------------------------
def core_velocities(x01, y01, x02, y02, Gamma=1.0):
    a = 0.5 * abs(x02 - x01)
    Omega = Gamma / (4.0 * np.pi * a * a)
    U = Omega * a
    # For Gamma > 0 and cores at (-a,0), (+a,0):
    # left core moves DOWN, right core moves UP
    vx1, vy1 = 0.0, -U
    vx2, vy2 = 0.0, +U
    return (vx1, vy1), (vx2, vy2)

# --------------------------
# dphi/dt using the robust identity: dφ_k/dt = - v_core_k · u^(k)
# --------------------------
def dphi_dt(x, y, x01, y01, x02, y02, Gamma=1.0):
    # per-vortex velocity fields (u1,v1),(u2,v2)
    _, _, (u1, v1), (u2, v2) = velocity(x, y, x01, y01, x02, y02, Gamma)
    (vx1, vy1), (vx2, vy2) = core_velocities(x01, y01, x02, y02, Gamma)
    return -(vx1 * u1 + vy1 * v1) - (vx2 * u2 + vy2 * v2)

# --------------------------
# Speed^2 and pressures
# --------------------------
def speed2(x, y, x01, y01, x02, y02, Gamma=1.0):
    u, v, *_ = velocity(x, y, x01, y01, x02, y02, Gamma)
    return u*u + v*v

def p_fixed(x, y, x01, y01, x02, y02, Gamma=1.0):
    return -0.5 * speed2(x, y, x01, y01, x02, y02, Gamma)

def p_free(x, y, x01, y01, x02, y02, Gamma=1.0):
    return dphi_dt(x, y, x01, y01, x02, y02, Gamma) - 0.5 * speed2(x, y, x01, y01, x02, y02, Gamma)

# --------------------------
# Mask near cores to avoid singular arrows/colors
# --------------------------
def core_mask(X, Y, cores, r0):
    m = np.zeros_like(X, dtype=bool)
    for (xc, yc) in cores:
        m |= (X - xc)**2 + (Y - yc)**2 < r0**2
    return m

# --------------------------
# MAIN
# --------------------------
# Grid
x = np.linspace(-3.0, 3.0, 200)
y = np.linspace(-3.0, 3.0, 200)
XX, YY = np.meshgrid(x, y, indexing='xy')

# Vortex positions and strength
x01, y01 = -1.0, 0.0
x02, y02 =  1.0, 0.0
Gamma = 1.0
cores = [(x01, y01), (x02, y02)]
r0 = 0.08  # mask radius

# Fields
u, v, *_ = velocity(XX, YY, x01, y01, x02, y02, Gamma)
phi = potential(XX, YY, x01, y01, x02, y02, Gamma)
P_fixed = p_fixed(XX, YY, x01, y01, x02, y02, Gamma)
P_free  = p_free (XX, YY, x01, y01, x02, y02, Gamma)

# Apply mask
mask = core_mask(XX, YY, cores, r0)
for A in (P_fixed, P_free, u, v):
    A[mask] = np.nan

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True, sharey=True)
titles = ["Free-moving vortices", "Vortices held fixed"]
Ps = [P_free, P_fixed]

# Shared pressure levels for fair comparison
finite_vals = np.concatenate([np.ravel(P_free[np.isfinite(P_free)]),
                              np.ravel(P_fixed[np.isfinite(P_fixed)])])
vmin, vmax = np.percentile(finite_vals, [2, 98])
levels = np.linspace(vmin, vmax, 40)

# Downsample quiver
step = 10
Xs, Ys = XX[::step, ::step], YY[::step, ::step]
us, vs = u[::step, ::step], v[::step, ::step]

for ax, title, P in zip(axes, titles, Ps):
    cf = ax.contourf(XX, YY, P, levels=levels, extend='both')
    cs = ax.contour(XX, YY, phi, colors='k', linewidths=0.6, levels=16, alpha=0.65)
    qv = ax.quiver(Xs, Ys, us, vs, pivot='mid', scale=45, width=0.0022)

    # draw masked core circles for reference
    for (xc, yc) in cores:
        ax.add_patch(Circle((xc, yc), r0, edgecolor='k', facecolor='none', lw=1))
        ax.plot(xc, yc, 'ro', ms=5)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(y.min(), y.max())
    ax.set_xlabel('x'); ax.set_title(title)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Pressure p')

axes[0].set_ylabel('y')
fig.suptitle('Two Equal Point Vortices (Γ=1): pressure (filled), potential (contours), velocity (quiver)', y=1.02, fontsize=13)

# plt.savefig("two_vortices_fields.png", dpi=300, bbox_inches="tight")
plt.show()
