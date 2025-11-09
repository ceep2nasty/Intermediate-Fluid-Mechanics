#IFM HW25

import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(0, 1, 1000)

# Case 1: shock at x = 0.5  (H(-x) IC at t=1)
u1 = (x < 0.5).astype(float)  # 1 for x<0.5, 0 for x>=0.5

# Case 2: rarefaction (H(x) IC at t=1)
u2 = np.piecewise(
    x,
    [x <= 0, (0 < x) & (x < 1), x >= 1],
    [0.0, lambda z: z, 1.0]
)

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

ax1.plot(x, u1, drawstyle="steps-post")
ax1.set_ylabel("u1(x,1)")
ax1.set_title("shock at x=0.5 (from H(-x))")
ax1.grid(True)

ax2.plot(x, u2)
ax2.set_xlabel("x")
ax2.set_ylabel("u2(x,1)")
ax2.set_title("rarefaction (from H(x))")
ax2.grid(True)

plt.tight_layout()
plt.show()
