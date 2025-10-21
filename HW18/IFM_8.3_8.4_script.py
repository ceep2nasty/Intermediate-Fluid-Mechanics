import numpy as np
import matplotlib.pyplot as plt

# ---- domain & constants ----
L, N = 3.0, 801
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
X, Y = np.meshgrid(x, y, indexing="xy")
R = np.hypot(X, Y)
TH = np.arctan2(Y, X)

a = 1.0       # cylinder radius
U = 1.0       # freestream speed
rho = 1.0

mask_cyl = R <= a
mask_origin = (R < 1e-6)

def fields(Gamma0=0.0):
    r = R.copy(); th = TH.copy()
    # polar velocities
    ur = U*(1.0 - a**2/np.where(r==0, np.inf, r**2))*np.cos(th)
    ut = -U*(1.0 + a**2/np.where(r==0, np.inf, r**2))*np.sin(th) + Gamma0/(2*np.pi*np.where(r==0, np.inf, r))
    # to Cartesian
    u = ur*np.cos(th) - ut*np.sin(th)
    v = ur*np.sin(th) + ut*np.cos(th)
    # potential & streamfunction
    with np.errstate(divide='ignore', invalid='ignore'):
        phi = U*(r + a**2/np.where(r==0, np.inf, r))*np.cos(th)
        psi = U*(r - a**2/np.where(r==0, np.inf, r))*np.sin(th) + (Gamma0/(2*np.pi))*np.log(np.where(r==0, 1.0, r))
    # pressure (p∞=0, ρ=1)
    q2 = u**2 + v**2
    p = 0.5*(U**2 - q2)

    for A in (phi, psi, u, v, p):
        A[mask_cyl] = np.nan
        A[mask_origin] = np.nan
    return phi, psi, u, v, p

def stagnation_points(Gamma0):
    g = Gamma0/(4*np.pi*U)
    pts = []
    if abs(g) <= 1:
        th = np.arcsin(g)
        pts += [(np.cos(th), np.sin(th)), (np.cos(np.pi - th), np.sin(np.pi - th))]
    else:
        s = abs(Gamma0)/(2*np.pi*U)
        r = 0.5*(s + np.sqrt(s**2 - 4.0))
        th = np.sign(Gamma0)*np.pi/2
        pts += [(r*np.cos(th), r*np.sin(th)), (-r*np.cos(th), -r*np.sin(th))]
    return pts

def composite_plot(Gamma0, levels_stream=25, levels_p=31, skip=10):
    phi, psi, u, v, p = fields(Gamma0)

    fig, ax = plt.subplots(figsize=(6.0, 6.0), constrained_layout=True)

    # 1) background = isobars
    cs_pressure = ax.contour(X, Y, p, levels=levels_p)  # pressure fill

    # 2) streamlines (solid) and equipotentials (dashed)
    cs_psi = ax.contour(X, Y, psi, levels=levels_stream, colors='k', linewidths=0.8)
    cs_phi = ax.contour(X, Y, phi, levels=levels_stream, linestyles='--', colors='k', linewidths=0.7)

    # 3) velocity vectors (subsampled)
    skip = 20
    ax.quiver(X[::skip,::skip], Y[::skip,::skip], u[::skip,::skip], v[::skip,::skip],
              pivot="mid", scale=30)

    # 4) cylinder boundary and stagnation points
    ax.add_artist(plt.Circle((0,0), a, color='k', fill=False, lw=1.2))
    spts = stagnation_points(Gamma0)
    if spts:
        ax.plot([sx for sx,sy in spts], [sy for sx,sy in spts], 'ro', ms=5)

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title(f"Gamma0 = {Gamma0/np.pi:.1f}*pi")

    return fig, ax

# -------- make one figure per case, each with all overlays --------
for G in [0.0, -2*np.pi, -4*np.pi, -9*np.pi/2]:
    composite_plot(G)
plt.show()
