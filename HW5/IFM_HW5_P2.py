# ==============================================================
# Pathline r(t) = (t, t^4): pretty plots with annotations
# ==============================================================

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------- Styling ----------
mpl.rcParams.update({
    "figure.figsize": (9, 6),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "font.size": 12,
})

# ---------- Kinematics ----------
def r(t):  return np.column_stack((t, t**4))
def v(t):  return np.column_stack((np.ones_like(t), 4*t**3))
def a(t):  return np.column_stack((np.zeros_like(t), 12*t**2))
def speed(t): return np.sqrt(1 + 16*t**6)

def T_hat(t):
    V = v(t)
    S = speed(t)[:, None]
    return V / S

def curvature(t):
    x_p, y_p = 1.0, 4*t**3
    x_pp, y_pp = 0.0, 12*t**2
    return np.abs(x_p*y_pp - y_p*x_pp) / (x_p**2 + y_p**2)**(1.5)

def N_hat(t):
    h = 1e-5
    Tp = (T_hat(t + h) - T_hat(t - h)) / (2*h)
    return Tp / np.linalg.norm(Tp, axis=1)[:, None]

# ---------- Data & special points ----------
t = np.linspace(-1.5, 1.5, 400)
R = r(t)

t_star = 56**(-1/6)  # positive max curvature
R_star = r(np.array([t_star]))[0]
T_star = T_hat(np.array([t_star]))[0]
N_star = N_hat(np.array([t_star]))[0]
k_star = curvature(np.array([t_star]))[0]

# ---------- Figure 1: Pathline + vectors ----------
fig1, ax = plt.subplots()

ax.plot(R[:,0], R[:,1], lw=2.5, color="#d18f00", label=r"pathline $r(t)=(t,t^4)$")

# Velocity/acceleration arrows
t_s = np.linspace(-1.25, 1.25, 13)
R_s, V_s, A_s = r(t_s), v(t_s), a(t_s)
v_scale, a_scale = 0.35, 0.030

ax.quiver(R_s[:,0], R_s[:,1], v_scale*V_s[:,0], v_scale*V_s[:,1],
          angles="xy", scale_units="xy", scale=1, width=0.004,
          color="#1f77b4", alpha=0.9, label=r"velocity $v$")

ax.quiver(R_s[:,0], R_s[:,1], a_scale*A_s[:,0], a_scale*A_s[:,1],
          angles="xy", scale_units="xy", scale=1, width=0.004,
          color="#d62728", alpha=0.9, label=r"acceleration $a$")

# Unit T and N at max curvature point
def add_unit_arrow(ax, base, vec, text, color):
    ax.annotate("", xy=(base[0]+vec[0], base[1]+vec[1]), xytext=(base[0], base[1]),
                arrowprops=dict(arrowstyle="-|>", lw=2, color=color))
    ax.text(base[0]+1.05*vec[0], base[1]+1.05*vec[1], text,
            color=color, fontsize=12, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.7))

u_scale = 0.6
add_unit_arrow(ax, R_star, u_scale*T_star, r"$\hat{T}$", "#2ca02c")
add_unit_arrow(ax, R_star, u_scale*N_star, r"$\hat{N}$", "#9467bd")

ax.scatter(*R_star, s=60, c="k", zorder=3)
ax.annotate(r"max curvature point",
            xy=R_star, xytext=(R_star[0]+0.3, R_star[1]+0.6),
            arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")
ax.set_title("Pathline with Velocity (blue) and Acceleration (red)\n"
             r"Unit Tangent/Normal at $t_\ast=56^{-1/6}$")
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper left")
fig1.tight_layout()

# ---------- Figure 2: Curvature vs t ----------
fig2, ax2 = plt.subplots()
K = curvature(t)
ax2.plot(t, K, lw=2.5, color="#d18f00", label=r"$\kappa(t)$")
ax2.scatter([0, t_star, -t_star], curvature(np.array([0, t_star, -t_star])), c="k")
ax2.annotate(r"min $\kappa=0$ at $t=0$", xy=(0, 0),
             xytext=(0.15, K.max()*0.08),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11)
ax2.annotate(r"max $\kappa$", xy=( t_star, k_star),
             xytext=(t_star+0.22, k_star*0.55),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11)
ax2.annotate(r"max $\kappa$", xy=(-t_star, k_star),
             xytext=(-t_star-0.6, k_star*0.55),
             arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=11)
ax2.set_xlabel(r"$t$")
ax2.set_ylabel(r"curvature $\kappa(t)$")
ax2.set_title("Curvature vs parameter $t$")
ax2.legend(loc="upper right")
fig2.tight_layout()

fig1.savefig("pathline_vectors.png", dpi=300, bbox_inches="tight")
fig2.savefig("curvature_vs_t.png", dpi=300, bbox_inches="tight")


plt.show()

