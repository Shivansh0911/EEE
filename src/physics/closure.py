"""
LBClosure: a fast float64 emulator of Vedant's exact synthesis.

Same equations, same search, same root choice as `pierce_synthesis` in the
vendored solver (Vaughan 1981 Sections III-A..F with the Yang 2006 |ln R|
amendment and the Greenwood 2016 relativistic factor), but vectorised over
designs in NumPy instead of one-at-a-time in 30-digit mpmath. What differs:

  * float64 instead of mpmath (relative effect ~1e-12);
  * kf is interpolated from `models/kf_table.json` (relative error ~1e-7)
    instead of re-integrated per call;
  * bisection runs a fixed 64 halvings (float64 machine precision) instead of
    to 1e-18 in mpmath;
  * the Yang-iteration cross-check is not run: it only produces warnings.

Gate G1 (tests/test_closure_vs_exact.py) checks it against his solver.

Two angles are kept DISTINCT throughout:
  theta_raw -- the analysis angle that solves Eq.(10) = Eq.(14). All geometry
               (Rc, Ra, z_ac, z_a, throw) is computed from it.
  theta_T   -- the aberration-corrected angle, sin(theta_raw) = 0.905 sin(theta_T)
               (Sec. III-F). The cathode is BUILT with it (RcT = 0.905 Rc), and it
               is the angle Panahi et al. Table 2 tabulates, so it is what the ANN
               and the measured guns are compared on.
"""

import numpy as np

from . import kf as kfmod
from . import lb

IN_TO_MM = 25.4
_DEG = np.pi / 180.0
_LO = 0.05 * _DEG
_HI = 89.0 * _DEG
N_HI = 400        # his window scan
N_SCAN = 300      # his root scan
N_BISECT = 64
C174 = 0.17409    # his constant, = lb.SPREAD_CONST to 5 figures

# Above -alpha ~ 7.94 (Rc/Ra ~ 15) the vendored y_from_a, capped at 20 Newton
# iterations, no longer converges (it does with 200) and the angle it returns is
# wrong. The emulator reproduces that faithfully -- its job is to match him --
# so designs beyond this are flagged `alpha_ok = False` and kept out of the
# synthetic envelope and the optimiser. Reported in INTEGRATION_DESIGN.md.
ALPHA_NEWTON_OK = 7.9


def _state(theta, rc, rw, muP, kf):
    """(a, y, R) at theta. Vaughan Eq.(3), Eq.(5) inverted, Eqs.(8, 11)."""
    a = np.sqrt(lb.VAUGHAN_CONE_CONST * kf * (1.0 - np.cos(theta)) / muP)
    y = lb.y_from_a(a)
    return a, y, rc * np.exp(-y) / rw


def _residual(theta, rc, rw, muP, kf):
    """|tan phi2| - |tan phi1|: Eq.(10) vs Eq.(14) with Yang's |ln R| (Eq. 9')."""
    a, y, R = _state(theta, rc, rw, muP, kf)
    tp1 = np.sin(theta) * (1.0 - lb.LENS_COEFF * lb.p_poly(y) / a)
    tp2 = C174 * np.sqrt(muP * np.abs(np.log(R)))
    return np.abs(tp2) - np.abs(tp1)


def _solve_theta(rc, rw, muP, kf):
    """Vectorised `solve_theta`: window, scan, bisect, keep largest root with R > 1.

    Inputs are 1-D arrays of equal length. Returns (theta, ok, why) where why
    is 0 ok, 1 empty window, 2 no root, 3 no physical root.
    """
    n = rc.size
    col = lambda v: v[:, None]

    # Window: first grid point where -alpha reaches the series limit.
    i = np.arange(1, N_HI + 1)
    t = _LO + (_HI - _LO) * i / N_HI                       # (400,)
    a = np.sqrt(lb.VAUGHAN_CONE_CONST * col(kf) * (1.0 - np.cos(t)) / col(muP))
    hit = a >= lb.A_MAX
    first = np.where(hit.any(axis=1), hit.argmax(axis=1), -1)
    theta_hi = np.where(first < 0, _HI,
                        np.where(first == 0, _LO, _LO + (_HI - _LO) * first / N_HI))
    why = np.where(theta_hi <= _LO, 1, 0)

    # Scan for sign changes.
    j = np.arange(N_SCAN + 1)
    ts = _LO + col(theta_hi - _LO) * j / N_SCAN              # (n, 301)
    fs = _residual(ts, col(rc), col(rw), col(muP), col(kf))
    fa, fb = fs[:, :-1], fs[:, 1:]
    exact_zero = fa == 0
    bracket = (fa * fb < 0) & ~exact_zero

    rows, cols = np.nonzero(bracket)
    a_ = ts[rows, cols]
    b_ = ts[rows, cols + 1]
    fa_ = fa[rows, cols]
    rr, rw_, rm, rk = rc[rows], rw[rows], muP[rows], kf[rows]
    for _ in range(N_BISECT):
        mid = 0.5 * (a_ + b_)
        fm = _residual(mid, rr, rw_, rm, rk)
        left = fa_ * fm <= 0
        b_ = np.where(left, mid, b_)
        a_ = np.where(left, a_, mid)
        fa_ = np.where(left, fa_, fm)
    roots_b = 0.5 * (a_ + b_)
    zr, zc = np.nonzero(exact_zero)
    root_row = np.concatenate([rows, zr])
    root_val = np.concatenate([roots_b, ts[zr, zc]])

    # Physical roots only (R > 1); take the largest per design, as he does.
    theta = np.full(n, np.nan)
    has_root = np.zeros(n, dtype=bool)
    has_root[root_row] = True
    if root_row.size:
        _, _, R = _state(root_val, rc[root_row], rw[root_row], muP[root_row], kf[root_row])
        phys = R > 1
        best = np.full(n, -np.inf)
        np.maximum.at(best, root_row[phys], root_val[phys])
        theta = np.where(np.isfinite(best), best, np.nan)
    why = np.where(why > 0, why,
                   np.where(~has_root, 2, np.where(np.isnan(theta), 3, 0)))
    return theta, why == 0, why


REASONS = {1: "empty theta window (series limit at the first step)",
           2: "no theta satisfies Eq.(10) = Eq.(14)",
           3: "every root leaves R <= 1 (no downstream waist)",
           4: "rc <= rw: compression R <= 1 for every theta"}


def synthesize(V, I, rw_mm, Jc, aperture_factor=lb.APERTURE_FACTOR, kf_iter=10):
    """
    Vectorised emulation of `pierce_synthesis(V, I, rw, Jc)`.

    V volts, I amps, rw mm, Jc A/cm^2 (scalars or arrays). Returns a dict of
    arrays; lengths in mm; angles in degrees; `feasible` boolean; `why` code.
    """
    V, I, rw_mm, Jc = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float))
                                            for x in (V, I, rw_mm, Jc)))
    V, I, rw_mm, Jc = (x.ravel().copy() for x in (V, I, rw_mm, Jc))
    rc = np.sqrt(I / (np.pi * Jc)) / 2.54                # Eq.(1), inches
    rw = rw_mm / IN_TO_MM
    muP = 1e6 * I / V ** 1.5                             # Eq.(2)
    n = V.size

    ok0 = rc > rw
    idx = np.nonzero(ok0)[0]
    theta = np.full(n, np.nan)
    kfv = np.ones(n)
    why = np.where(ok0, 0, 4)

    if idx.size:
        r_, w_, m_, v_ = rc[idx], rw[idx], muP[idx], V[idx]
        k = np.ones(idx.size)
        # kf depends on Rc/Ra, which depends on theta: iterate as he does.
        for _ in range(kf_iter):
            th, good, _w = _solve_theta(r_, w_, m_, k)
            _, y, _ = _state(np.where(good, th, _LO), r_, w_, m_, k)
            k_new = np.where(good, kfmod.kf(v_, y), k)
            done = np.max(np.abs(k_new - k)) < 1e-12
            k = k_new
            if done:
                break
        th, good, w = _solve_theta(r_, w_, m_, k)
        theta[idx], kfv[idx], why[idx] = th, k, w

    feas = why == 0
    t = np.where(feas, theta, _LO)
    a, y, R = _state(t, rc, rw, muP, kfv)

    # Section III-E geometry (inches), from theta_raw.
    Rc = rc / np.sin(t)                                   # Eq.(15)
    Ra = Rc * np.exp(-y)                                  # Eq.(16)
    rb_za = rc * np.exp(-y)                               # Eq.(17)
    ra = aperture_factor * rb_za                          # Eq.(18)
    zac = Rc - Ra                                         # Eq.(19)
    disc = Ra ** 2 - ra ** 2
    za_valid = disc > 0
    za = np.where(za_valid, Rc - np.sqrt(np.where(za_valid, disc, 0.0)), Rc)  # Eq.(20)
    Rm1 = R - 1
    Rp = np.where(Rm1 > 0, Rm1, 0.0)
    Z = 11.0 * np.sqrt(Rp) + 1.32 * Rp + 0.0615 * Rp ** 2 - 0.00167 * Rp ** 3   # Eq.(12)
    zw = np.where(Rm1 > 0, zac + rw * Z / np.sqrt(muP), zac)                    # Eq.(21)

    # Section III-F aberration correction.
    s = np.sin(t) / lb.ABERRATION_K
    corr = s <= 1
    thetaT = np.where(corr, np.arcsin(np.minimum(s, 1.0)), t)
    RcT = np.where(corr, lb.ABERRATION_K * Rc, Rc)

    nan = lambda v: np.where(feas, v, np.nan)
    mm = lambda v: nan(v * IN_TO_MM)
    return dict(
        V=V, I=I, rw_mm=rw_mm, Jc=Jc, muP=muP, C=(rc / rw) ** 2,
        rc_mm=rc * IN_TO_MM,
        theta=nan(np.degrees(t)), thetaT=nan(np.degrees(thetaT)),
        Rc_mm=mm(Rc), Ra_mm=mm(Ra), zac_mm=mm(zac), za_mm=mm(za),
        rb_za_mm=mm(rb_za), ra_mm=mm(ra), zw_mm=mm(zw), RcT_mm=mm(RcT),
        R=nan(R), y=nan(y), alpha_neg=nan(a), rel_factor=nan(kfv),
        alpha_ok=feas & (a < ALPHA_NEWTON_OK),
        za_valid=feas & za_valid, correction_applied=feas & corr,
        feasible=feas, why=why,
    )


def from_pcr(P_uperv, rw_mm, C, V=1000.0, **kw):
    """Same synthesis addressed by the ANN's inputs (P, rw, C) at voltage V."""
    P_uperv, rw_mm, C, V = np.broadcast_arrays(*(np.atleast_1d(np.asarray(x, dtype=float))
                                                 for x in (P_uperv, rw_mm, C, V)))
    I = P_uperv * 1e-6 * V ** 1.5
    rc_cm = np.sqrt(C) * rw_mm / 10.0
    Jc = I / (np.pi * rc_cm ** 2)
    return synthesize(V, I, rw_mm, Jc, **kw)
