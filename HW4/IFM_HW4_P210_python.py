import numpy as np
import matplotlib.pyplot as plt

# Define the Jacobians
Js = [
    np.array([[2, 1, 1],
              [-1, 2, 0],
              [1, 0, 2]]),
    
    np.array([[10, 1, 1],
              [-1, 2, 0],
              [1, 0, 2]]),
    
    np.array([[10, 1, 1],
              [-1, 10, 0],
              [1, 0, 2]])
]

labels = ["J1", "J2", "J3"]

def ellipsoid_axes(J):
    """Compute ellipsoid semi-axes from Jacobian J."""
    M = J.T @ J
    vals, vecs = np.linalg.eigh(M)   # eigenvalues, eigenvectors
    axes = 1 / np.sqrt(vals)         # ellipsoid semi-axes lengths
    return axes, vecs, np.linalg.det(J)

# Create sphere mesh
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 50)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
sphere = np.stack([xs, ys, zs], axis=0)

fig = plt.figure(figsize=(15,5))

for i, J in enumerate(Js):
    axes, vecs, detJ = ellipsoid_axes(J)
    
    # scale and rotate the sphere
    ellipsoid = vecs @ np.diag(axes) @ sphere.reshape(3, -1)
    X, Y, Z = ellipsoid.reshape(3, *xs.shape)
    
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.plot_surface(X, Y, Z, color="royalblue", alpha=0.7, edgecolor="k", linewidth=0.2)
    
    # set equal limits based on ellipsoid size
    max_range = np.max([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]) / 2
    mid_x = (X.max()+X.min()) / 2
    mid_y = (Y.max()+Y.min()) / 2
    mid_z = (Z.max()+Z.min()) / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.set_title(f"{labels[i]}: det={detJ:.2f}\naxes={np.round(axes,2)}")
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")

plt.tight_layout()
plt.show()
