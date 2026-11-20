"""
Laminar pipe flow of a Carreau fluid.

This is Appendix A of Chandra, Shankar & Das, JFM 885, A47 (2020):

    div v = 0                                          (A 1)
    -grad P + div tau = 0                              (A 2)
    eta = eta_inf + (eta_0 - eta_inf) [1+(lambda*gamma)^2]^((n-1)/2)   (A 3)

For fully developed pipe flow the stress is linear in radius. With
r* = r/R and Gamma* = gamma / (4V/R), their balance becomes

    (eta/eta_0) Gamma* = r* f Re / 16                  (A 4, with the solvent plateau kept)

The flow rate fixes f. When n = 1 or lambda = 0 this returns f = 16/Re.
No neural net is used here: each radius is a scalar Newton solve, and f
is found by bisection.
"""

from __future__ import annotations

import numpy as np

RHO = 998.0
ETA_SOLVENT = 0.90e-3


def wi_pipe(Re: float, lam_s: float, eta0: float, D_m: float) -> float:
    """lambda * (8V/D), the Carreau number based on the Newtonian wall shear."""
    if lam_s <= 0.0 or D_m <= 0.0:
        return 0.0
    return lam_s * 8.0 * Re * eta0 / (RHO * D_m**2)


def _gamma_star(r: np.ndarray, f: float, Re: float, n: float, beta: float, Wi: float) -> np.ndarray:
    """Newton solve of (A 4) at every radius. Gamma* = 0 at the centre."""
    rhs = r * (f * Re / 16.0)
    G = np.maximum(rhs, 1e-16)
    if Wi <= 0.0 or abs(n - 1.0) < 1e-12:
        return rhs
    p = 0.5 * (n - 1.0)
    for _ in range(50):
        s = 1.0 + (Wi * G) ** 2
        thin = np.power(s, p)
        eta = beta + (1.0 - beta) * thin
        dthin = p * np.power(s, p - 1.0) * (2.0 * Wi * Wi * G)
        deta = (1.0 - beta) * dthin
        g = eta * G - rhs
        gp = eta + G * deta
        gp = np.where(np.abs(gp) < 1e-14, np.sign(gp) * 1e-14 + 1e-14, gp)
        step = np.clip(g / gp, -0.5 * G, 0.5 * G)
        G = np.maximum(G - step, 1e-16)
        if np.max(np.abs(step)) < 1e-10:
            break
    return G


def _flow_integral(G: np.ndarray, r: np.ndarray) -> float:
    """int_0^1 Gamma* r*^2 dr*. Equals 1/4 when the mean velocity is V."""
    return float(np.trapezoid(G * r**2, r))


def f_carreau(Re: float, n: float, beta: float, Wi: float, n_r: int = 120) -> float:
    """
    Fanning friction factor of laminar Carreau pipe flow at this Re.

    Re uses the zero-shear viscosity. beta = eta_inf / eta_0.
    Wi = lambda * 8V/D.
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive")
    beta = float(np.clip(beta, 1e-4, 1.0))
    n = float(n)
    if Wi <= 1e-8 or abs(n - 1.0) < 1e-10 or beta > 0.999:
        return 16.0 / Re

    r = np.linspace(0.0, 1.0, n_r)

    def integral(f: float) -> float:
        return _flow_integral(_gamma_star(r, f, Re, n, beta, Wi), r) - 0.25

    f_hi = 16.0 / Re
    f_lo = beta * f_hi * 0.5
    i_hi = integral(f_hi)
    i_lo = integral(f_lo)
    # Shear thinning raises the shear rate at fixed f, so the root is below 16/Re.
    if i_hi < 0.0:
        return f_hi
    guard = 0
    while i_lo > 0.0 and guard < 12:
        f_lo *= 0.5
        i_lo = integral(f_lo)
        guard += 1
    for _ in range(50):
        f_mid = 0.5 * (f_lo + f_hi)
        i_mid = integral(f_mid)
        if i_mid > 0.0:
            f_hi = f_mid
        else:
            f_lo = f_mid
        if abs(f_hi - f_lo) < 1e-8 * f_hi:
            break
    return 0.5 * (f_lo + f_hi)


def f_carreau_curve(Re: np.ndarray, n: float, beta: float, lam_s: float, eta0: float, D_m: float) -> np.ndarray:
    out = np.empty(len(Re))
    for i, R in enumerate(Re):
        Wi = wi_pipe(float(R), lam_s, eta0, D_m)
        out[i] = f_carreau(float(R), n, beta, Wi)
    return out
