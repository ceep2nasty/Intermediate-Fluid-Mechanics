# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 08:39:48 2025

@author: coled
"""

# IFM HW22: Rayleigh/Huogoniot Solve wit Van der Waals gas

import math
from typing import List, Tuple, Dict

import sympy as sp
import numpy as np

# given gas constants

R = 14.4843
cv = 1131.588
a = 22.3889
b = 0.0007244

# upstream state

v1 = 0.002700
T1 = 656.0
U = 37.89  # note that this is the shock velocity; will transform into shock-frame

# reference values
T0 = 300.0
v0 = 1.0

    # note the omission of s0 (0) and e0 (0)
    

# thermo functions

# EOS

def p_vdw(T: float, v:float)->float:
    return R*T/(v-b) - a / (v**2)

# energy given constant cv
def e_vdw(T: float, v:float) -> float:
    return cv*(T-T0) - a*(1.0/v - 1.0/v0)

def s_vdw(T:float, v:float):
    return cv*sp.log(T/T0) + R*sp.log((v-b)/(v0-b))

def h_vdw (T:float, v:float) -> float:
    return e_vdw(T,v) + p_vdw(T, v)*v

# upstream properties
rho1 = 1.0/v1
p1 = p_vdw(T1, v1)
h1 = h_vdw(T1, v1)
s1 = float(sp.N(s_vdw(T1, v1)))

# mass flux across shock in shock-fixed frame
m = rho1*U

# symbolic expressions for solver
v2, T2 = sp.symbols('v2 T2', positive = True)
p_rayleigh = p1 - m**2*(v2-v1) # rayleigh p2
p_eos = R*T2/(v2-b) - a/v2**2 #p2 from EOS
e2_expr = cv*(T2-T0) - a*(1.0/v2 - 1.0/v0)
h2_expr = e2_expr + p_rayleigh*v2    # rayleigh p2 inside h2
hugoniot = h2_expr - h1 - sp.Rational(1,2)*(p_rayleigh-p1)*(v1+v2)
eos_eq = p_rayleigh - p_eos

# lambidfy the residuals
F1 = sp.lambdify((v2, T2), hugoniot, 'numpy')
F2 = sp.lambdify ((v2, T2), eos_eq, 'numpy')

# find the sound speed
rho = sp.symbols('rho', positive=True)
v_rho = 1/rho
p_rhoT = R*T2/(v_rho - b) - a/v_rho**2
s_rhoT = cv*sp.log(T2/T0) + R*sp.log((v_rho - b)/(v0 - b))
dpdrho = sp.diff(p_rhoT, rho)
dpdT   = sp.diff(p_rhoT, T2)
dsdrho = sp.diff(s_rhoT, rho)
dsdT   = sp.diff(s_rhoT, T2)
dTdrho_s = -dsdrho/dsdT
a2_rhoT  = sp.simplify(dpdrho + dpdT*dTdrho_s)
a2_fun   = sp.lambdify((rho, T2), a2_rhoT, 'numpy')

def sound_speed(T: float, v: float) -> float:
    return float(a2_fun(1.0/v, T))**0.5

# root solver

def solve_states(
        seeds_v: List[float],
        seeds_T: List[float],
        tol: float = 1e-12,
        ) -> List[Tuple[float,float]]:
    roots = []
    keys = set()
    for gv in seeds_v:
        for gT in seeds_T:
            try:
                sol = sp.nsolve([hugoniot, eos_eq], [v2, T2], [gv, gT],
                                tol = tol, maxsteps=200, prec=50)
                vv, TT = float(sol[0]), float(sol[1])
                key = (round(vv, 10), round(TT, 6))
                if key not in keys:
                    keys.add(key)
                    roots.append((vv,TT))
            except Exception:
                pass
    return roots

def summarize_root(vv: float, TT: float) -> Dict[str, float]:
    rho2 = 1.0/vv
    p2 = p_vdw(TT,vv)
    u2 = m/rho2
    a1 = sound_speed(T1, v1)
    a2 = sound_speed(TT, vv)
    M1 = U/a1
    M2 = u2/a2
    s2 = float(sp.N(s_vdw(TT, vv)))
    ds = s2 - s1
    dpdv_T = float(sp.diff(R*T2/(v2-b)-a/v2**2, v2).subs({v2:vv, T2: TT}))
    
    return dict(v2=vv, T2=TT, p2=p2, u2=u2, a2=a2, M1=M1, M2=M2, ds=ds, dpdv_T=dpdv_T)

def main():
    # Build coarse mesh for seed generation
    v_mesh = np.linspace(max(b*(1+1e-3), 0.4*v1), 2.5*v1, 80)
    T_mesh = np.linspace(T1-200, T1+250, 60)
    
    # Evaluate residuals on the grid (using lambdified F1,F2)
    V, T = np.meshgrid(v_mesh, T_mesh, indexing='ij')
    R1 = F1(V, T)
    R2 = F2(V, T)
    
    # Select candidate seeds where both residuals are small
    
    
    # finite masks to avoid all-NaN/Inf cases
    finite1 = np.isfinite(R1)
    finite2 = np.isfinite(R2)
    if not finite1.any() or not finite2.any():
        print("No finite residuals in coarse scan; adjust mesh ranges.")
        return
    
    tau1 = np.percentile(np.abs(R1[finite1]), 5)
    tau2 = np.percentile(np.abs(R2[finite2]), 5)

    mask = (np.abs(R1) < tau1) & (np.abs(R2) < tau2)
    cand_vs = V[mask]
    cand_Ts = T[mask]
    
    # Now solve on these candidates

    roots = solve_states(cand_vs, cand_Ts)
    if not roots:
        print("No roots found bozo")
        return

    # Sort by v2
    roots = sorted(roots, key=lambda x: x[0])
    print("Found {} roots:".format(len(roots)))
    for i,(vv,TT) in enumerate(roots, start=1):
        S = summarize_root(vv, TT)
        print(f"[{i}] v2 = {S['v2']:.9f} m^3/kg, T2 = {S['T2']:.6f} K, p2 = {S['p2']:.3f} Pa")
        print(f"    u2 = {S['u2']:.6f} m/s, a2 = {S['a2']:.6f} m/s,  M1 = {S['M1']:.6f},  M2 = {S['M2']:.6f}")
        print(f"    Δs = {S['ds']:.6e} J/(kg·K),  (dp/dv)_T = {S['dpdv_T']:.3e}  (<0 stable)")

main()
    


    
        
                