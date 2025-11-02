# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 13:57:17 2025

@author: coled_agkeohi
"""

# IFM HW22: Rayleigh Flow

#symbolically solve rayleigh flow equations to plot both branches of mach number


import sympy as sp

x, u, p, rho, h = sp.symbols('x u p rho h', real = True)
gamma, rho1, u1, p1, h1, qw, d = sp.symbols('gamma rho1 u1 p1 h1 qw d', positive=True)

Qprime = 4*qw/(rho1*u1*d)  #dQ/dx term
Qx = Qprime*x

# equations to be solved

eq_cont = sp.Eq(rho*u, rho1*u1)
eq_mom = sp.Eq(rho*u**2 + p, rho1*u1**2+p1)
eq_h = sp.Eq(h, gamma/(gamma-1)*p/rho)
eq_eng = sp.Eq(h + u**2/2, h1+u1**2/2 + Qx)

#elim rho, p
rho_expr = sp.solve(eq_cont, rho)[0]
p_expr = sp.solve(eq_mom.subs(rho,rho_expr), p)[0]

#sub into h and energy

h_expr   = eq_h.rhs.subs({rho: rho_expr, p: p_expr})

#energy as a function of u

eq_u = sp.simplify(sp.Eq(h_expr + u**2/2, h1 + u1**2/2 +Qx))

#solve for u(x) as two branches

u_branches = sp.solve(sp.together(eq_u), u)

eq_u_poly = sp.expand((h_expr + u**2/2) - (h1 + u1**2/2 + Qx)) # bring everything to LHS
poly_u = sp.Poly(eq_u_poly, u)
alpha = poly_u.coeff_monomial(u**2)
beta = -poly_u.coeff_monomial(u)
C_x = poly_u.coeff_monomial(1)
Delta = sp.simplify(beta**2 - 4*alpha*C_x) # discriminant of polynomial
x_star_solutions = sp.solve(sp.Eq(Delta, 0), x)
x_star = x_star_solutions[0]

#Convert each u-branch into Mach number M(x) 
def M_of(u_expr):
    rho_x = sp.simplify(rho_expr.subs(u, u_expr))                # rho(u(x))
    p_x   = sp.simplify(p_expr.subs(u, u_expr))                  # p(u(x))
    a_x   = sp.sqrt(gamma * p_x / rho_x)                         # speed of sound
    return sp.simplify(u_expr / a_x)                             # Mach

M_branches = [M_of(ub) for ub in u_branches]

# now plug in data, plot

R = 287.0
T1 = 300.0
M1 = 2.0
gamma_val = 1.4

a1 = (gamma_val*R*T1)**0.5
u1_num = M1*a1
rho1_num = 1.0e5/(R*T1)                       
cp = gamma_val*R/(gamma_val-1)
h1_num = cp*T1

subs_numeric = {
    gamma: gamma_val, p1: 1.0e5, u1: u1_num, rho1: rho1_num,
    h1: h1_num, qw: 1.0e4, d: 0.01
}

# Evaluate x* and turn the symbolic M(x) into fast numeric callables
xstar = float(sp.N(x_star.subs(subs_numeric)))    # choke location

M_plus = sp.lambdify(x, sp.N(M_branches[1].subs(subs_numeric)), 'numpy')
M_minus = sp.lambdify(x, sp.N(M_branches[0].subs(subs_numeric)), 'numpy')
xstar = float(sp.N(x_star.subs(subs_numeric)))

import numpy as np
xgrid = np.linspace(0.0, 0.999*xstar, 400)  # a hair further from xstar

M_upper = np.real_if_close(M_plus(xgrid))
M_lower = np.real_if_close(M_minus(xgrid))


import matplotlib.pyplot as plt

# pick inlet-matching branch
M1 = 2.0
M0 = [float(M_plus(0.0)), float(M_minus(0.0))]
idx_phys = int(np.argmin([abs(M - M1) for M in M0]))
M_phys = [M_plus, M_minus][idx_phys]
M_comp = [M_plus, M_minus][1 - idx_phys]

M_phys_vals = np.real_if_close(M_phys(xgrid))
M_comp_vals = np.real_if_close(M_comp(xgrid))

plt.figure()
plt.plot(xgrid, M_phys_vals, label="Branch given M1=2")
plt.plot(xgrid, M_comp_vals, "--", label="Subsonic pal")
plt.axvline(xstar, linestyle=":", label="Choke line")
plt.xlabel("x [m]"); plt.ylabel("Mach"); plt.legend(); plt.tight_layout(); plt.show()


