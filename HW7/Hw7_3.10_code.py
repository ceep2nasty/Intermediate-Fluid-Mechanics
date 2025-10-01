import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

# ---- Define the system ----
def f(t, x):
    x1, x2, x3 = x
    return np.array([
        -x1 - 4*x2,
        2*x1 - x1*x2,
        -x3 - x1
    ], dtype=float)

# ---- Runge–Kutta 4 integrator ----
def rk4_step(fun, t, x, h):
    k1 = fun(t, x)
    k2 = fun(t + 0.5*h, x + 0.5*h*k1)
    k3 = fun(t + 0.5*h, x + 0.5*h*k2)
    k4 = fun(t + h,     x + h*k3)
    return x + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def integrate_rk4(fun, t0, x0, t1, h):
    n = int(np.ceil(abs(t1 - t0)/h))
    h_signed = np.sign(t1 - t0) * abs(h)
    t = t0
    x = np.array(x0, dtype=float)
    T = [t]; X = [x.copy()]
    for _ in range(n):
        x = rk4_step(fun, t, x, h_signed)
        t += h_signed
        T.append(t); X.append(x.copy())
    return np.array(T), np.vstack(X)

# ---- Initial condition and integration ----
x0 = [1.0, 1.0, 1.0]   # starting point
T, X = integrate_rk4(f, 0.0, x0, 6.0, 0.01)  # integrate to t=6 with step size 0.01

# ---- Make the 3D plot ----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(X[:,0], X[:,1], X[:,2], linewidth=2, label="Pathline")
ax.scatter([x0[0]], [x0[1]], [x0[2]], color='red', s=50, label="Initial point")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("3D Pathline through (1,1,1)")
ax.legend()
plt.show()

