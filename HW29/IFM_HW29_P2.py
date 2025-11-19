# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 16:45:43 2025

@author: coled
"""

# IFM HW29 P3: X momentum pulse in viscous flowfield

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def u_fun(y, z, t):
    # y, z can be scalars or arrays; t > 0
    r2 = y**2 + z**2
    return 1.0 / (4.0 * np.pi * t) * np.exp(-r2 / (4.0 * t))

# ----------------------------------------------------
# Part 1: contour plots of u(y,z,t)
# ----------------------------------------------------
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
plt.savefig("IFM_HW29_fig3.png", dpi=300, bbox_inches="tight")

# ----------------------------------------------------
# Part 2: Cauchy viscous stress quadric at y=z=t=1, rho=nu=M=1
# ----------------------------------------------------
rho = nu = M = y0 = z0 = t = 1.0
mu  = rho * nu
u0  = M/(4*np.pi*nu*t) * np.exp(-(y0**2 + z0**2)/(4*nu*t))
dudy = -y0/(2*nu*t) * u0
dudz = -z0/(2*nu*t) * u0

tau = np.array([
    [0.0,      mu*dudy, mu*dudz],
    [mu*dudy,  0.0,     0.0    ],
    [mu*dudz,  0.0,     0.0    ]
])

# eigen-decomposition: tau = Q diag(lam) Q^T
lam, Q = np.linalg.eigh(tau)

print("eigenvalues:", lam)
print("eigenvectors (columns):\n", Q)

# Separate the ~zero eigenvalue from the two nonzero ones
eps = 1e-10
idx_zero = np.argmin(np.abs(lam))
idx_nz = [i for i in range(3) if i != idx_zero]

lam_nz = lam[idx_nz]          # two non-zero eigenvalues
e_nz   = Q[:, idx_nz]         # corresponding eigenvectors
e_zero = Q[:, idx_zero]       # axis of the cylinder

# Label them as lambda_plus > 0 (tension), lambda_minus < 0 (compression)
if lam_nz[0] > 0:
    lam_plus,  lam_minus  = lam_nz[0], lam_nz[1]
    e_plus,    e_minus    = e_nz[:,0], e_nz[:,1]
else:
    lam_plus,  lam_minus  = lam_nz[1], lam_nz[0]
    e_plus,    e_minus    = e_nz[:,1], e_nz[:,0]

# ----------------------------------------------------
# 2a. Quadric in eigenbasis (principal coordinates)
# ----------------------------------------------------
C = 1.0   # sets the overall size of the quadric

# Hyperbola in (xi2, xi3) plane:
# lam_plus * xi2^2 + lam_minus * xi3^2 = C
a = np.sqrt(C / lam_plus)
b = np.sqrt(C / abs(lam_minus))

xi1_vals = np.linspace(-1.0, 1.0, 20)   # along axis (zero eigenvalue direction)
s = np.linspace(-2.0, 2.0, 200)         # parameter along hyperbola

xi2 = a * np.cosh(s)
xi3 = b * np.sinh(s)

Xi1, S = np.meshgrid(xi1_vals, s)
Xi2 = np.tile(xi2, (xi1_vals.size, 1)).T
Xi3 = np.tile(xi3, (xi1_vals.size, 1)).T

fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')

ax2.plot_surface(Xi1, Xi2, Xi3, alpha=0.6, linewidth=0)

ax2.set_xlabel(r'$\xi_1$ (zero-stress axis)')
ax2.set_ylabel(r'$\xi_2$')
ax2.set_zlabel(r'$\xi_3$')
ax2.set_title("Cauchy viscous stress quadric in principal-stress coordinates")
fig2.savefig("IFM_HW29_fig4.png", dpi=300, bbox_inches="tight")
# ----------------------------------------------------
# 2b. Rotate quadric into lab coordinates (x,y,z)
# x = Q * xi  (Q columns are eigenvectors)
# ----------------------------------------------------
# stack eigenbasis coordinates as 3 x N array and rotate
Xi_stack = np.stack([Xi1.ravel(), Xi2.ravel(), Xi3.ravel()], axis=0)  # shape (3, N)
X_lab = Q @ Xi_stack                                                  # shape (3, N)

Xx = X_lab[0,:].reshape(Xi1.shape)
Xy = X_lab[1,:].reshape(Xi1.shape)
Xz = X_lab[2,:].reshape(Xi1.shape)

fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection='3d')
ax3.plot_surface(Xx, Xy, Xz, alpha=0.6, linewidth=0)

ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_title("Cauchy viscous stress quadric in lab coordinates")
fig3.savefig("IFM_HW29_fig5.png", dpi=300, bbox_inches="tight")


plt.show()

