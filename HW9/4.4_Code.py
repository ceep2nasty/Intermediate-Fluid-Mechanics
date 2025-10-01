import numpy as np
import matplotlib.pyplot as plt

# Domain
x1_min, x1_max = -2.5, 4.5
x2_min, x2_max = -3.0, 3.0
N = 81
x1 = np.linspace(x1_min, x1_max, N)
x2 = np.linspace(x2_min, x2_max, N)
X1, X2 = np.meshgrid(x1, x2)

# Fields
V1, V2 = X1 + X2, X1 - X2                   # velocity
A1, A2 = 2.0 * X1, 2.0 * X2                 # acceleration
S1, S2 = np.full_like(X1, 2.0), np.zeros_like(X2)  # ∇·T

# Helper: uniform downsampling and safe normalization
def downsample(M, step):
    return M[::step, ::step]

def normalize(U, V, eps=1e-12):
    mag = np.sqrt(U*U + V*V)
    Um = U / np.maximum(mag, eps)
    Vm = V / np.maximum(mag, eps)
    return Um, Vm

skip = 3                      # fewer = denser arrows
scale_quiver = 10            # smaller -> longer arrows with scale_units='xy'

# 1) Acceleration
Xq = downsample(X1, skip); Yq = downsample(X2, skip)
Uq = downsample(A1, skip); Vq = downsample(A2, skip)
Uu, Vu = normalize(Uq, Vq)    # show direction clearly
plt.figure(figsize=(6,5))
plt.quiver(Xq, Yq, Uu, Vu, angles='xy', scale_units='xy', scale=scale_quiver, pivot='mid')
plt.axis('equal'); plt.xlim(x1_min, x1_max); plt.ylim(x2_min, x2_max)
plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
plt.title(r"Acceleration field")
plt.tight_layout()

# 2) Velocity
Uq = downsample(V1, skip); Vq = downsample(V2, skip)
Uu, Vu = normalize(Uq, Vq)
plt.figure(figsize=(6,5))
plt.quiver(Xq, Yq, Uu, Vu, angles='xy', scale_units='xy', scale=scale_quiver, pivot='mid')
plt.axis('equal'); plt.xlim(x1_min, x1_max); plt.ylim(x2_min, x2_max)
plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
plt.title(r"Velocity field")
plt.tight_layout()

# 3) Surface-stress force density ∇·T
Uq = downsample(S1, skip); Vq = downsample(S2, skip)
Uu, Vu = normalize(Uq, Vq)
plt.figure(figsize=(6,5))
plt.quiver(Xq, Yq, Uu, Vu, angles='xy', scale_units='xy', scale=scale_quiver, pivot='mid')
plt.axis('equal'); plt.xlim(x1_min, x1_max); plt.ylim(x2_min, x2_max)
plt.xlabel(r"$x_1$"); plt.ylabel(r"$x_2$")
plt.title(r"Surface-stress force density")
plt.tight_layout()

plt.show()
