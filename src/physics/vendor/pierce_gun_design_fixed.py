# pierce_gun_design.py
# Interactive Pierce Gun Synthesis Tool
#
# Base method:
#   Vaughan, J.R.M. (1981), "Synthesis of the Pierce Gun",
#   IEEE Trans. Electron Devices ED-28(1), 37-41.
#
# Amendments from later literature:
#   Yang C., Jia B., Zhu Z. (2006), "Improvement in the synthesis method of the
#     Pierce electron gun", High Power Laser and Particle Beams 18(9).
#     -> Eq.(9'):  tan(phi2) uses |ln R|, so the residual survives R < 1
#     -> Eq.(10'): the update uses |tan(phi2)/tan(phi1)|, and the square root
#                  may be raised to a 4th root or higher for damping
#     Implemented in yang_iterate() as an independent cross-check on the
#     bracketed bisection in solve_theta().
#   Liao et al. (2003), via the same paper: initial angle 10*sqrt(muP*rc/rw)
#     rather than Vaughan's 30*sqrt(muP), used as yang_iterate()'s default start.
#
# NOT implemented, and worth adding if the paper can be obtained:
#   Sharma R.K., Sinha A.K., Joshi S.N. (2001), "An improved method for the
#   synthesis of anode aperture for Pierce guns", IEEE Trans. Electron Devices
#   48(2), 395-397. Replaces the empirical ra = 1.2*rb(za) used here (Eq.18),
#   which is this program's weakest step: above theta = 56.44 deg it makes
#   Eq.(20) insoluble, so za cannot be computed for high-convergence guns.
#   The paper's abstract states the approach needs no empirical factor and
#   agrees better with real guns; the equation itself is behind IEEE paywall
#   and has deliberately not been guessed at.
#
# Focus electrode (diagram stage):
#   Built as a straight face starting a small gap (k1 x Rc) beside the cathode
#   rim, at angle psi to the beam edge, fixed at 67.5 deg (Pierce's angle), and
#   running toward the anode until a gap alpha short of its face. The angle
#   actually built is measured from the drawn points and printed. Previously
#   the face was a chord of the cathode sphere, which tied its angle to its
#   length and gave ~75 deg at the default inputs.
#
# Validated against:
#   Kant & Nagaraju, IJAMT 6(2) 2021 (8 MW S-band klystron gun): all ten
#     tabulated outputs within 0.51%.
#   Ardebili et al., Results in Engineering (2025), Table 1: nine guns,
#     perveance 1.90-3.67 uP, convergence ratio 5.33-306; mean deviation
#     0.26 deg on the aberration-corrected half angle.
#
# Usage : python pierce_gun_design.py
# Units : beam voltage entered in kV; all lengths in cm; current in A; A/cm²
#         beam waist radius rw entered in mm everywhere

import math
import numpy as np

from mpmath import mp, mpf, sin, cos, sqrt, exp, log, radians, degrees, pi, asin
import csv, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

mp.dps = 30          # 30 significant digits is far beyond physical precision;
                     # was 50 — reducing gives ~30-40 % speedup on mpmath calls
CM       = 2.54      # 1 inch = 2.54 cm
IN_TO_MM = CM * 10   # 1 inch = 25.4 mm — used throughout instead of '* CM * 10'

# ─────────────────────────────────────────────────────────────────────────────
# Module-level mpf constants — avoids re-parsing string literals on every call
# ─────────────────────────────────────────────────────────────────────────────

# a_from_y (Eq. 5) coefficients
_A1 = mpf("0.3");     _A2 = mpf("0.075");   _A3 = mpf("0.01432")
_A4 = mpf("0.00216"); _A5 = mpf("0.00035")   # Vaughan: 0.00027 -> 0.00035 extends
#                                              Eq.(5) to within 1% for Rc/Ra up to 20

# y_from_a initial-guess coefficients
_Y0 = mpf("0.275");  _Y1 = mpf("0.06");    _Y2 = mpf("0.006")

# Derivative denominator coefficients (also used for P in pierce_synthesis)
_D1 = mpf("0.6");    _D2 = mpf("0.225");   _D3 = mpf("0.05728")
_D4 = mpf("0.0108"); _D5 = mpf("0.00162")

# pierce_synthesis constants
_C14_67 = mpf("14.67");   _C0_417 = mpf("0.417");   _C0_174 = mpf("0.17409")
_CP1    = mpf("0.0573");   _CP2    = mpf("0.0021")    # P-polynomial (Eq. 10)
_C1_2   = mpf("1.2")
_C11_0  = mpf("11.0");    _C1_32  = mpf("1.32")
_C0_061 = mpf("0.0615");  _C0_001 = mpf("0.00167")
_C30    = mpf("30");      _C0_905 = mpf("0.905")
_C1e6   = mpf("1e6")

_NEWTON_TOL  = mpf("1e-20")
_MAX_NEWTON  = 20   # was 60; Newton converges quadratically — rarely needs > 10

# Pierce's focus electrode angle to the beam edge (deg), fixed.
PIERCE_ANGLE = 67.5
# Outer wall (pts 7, 8) and focus electrode tip (pt 6) heights, as multiples of
# H2 = height of pt 2 (the cathode rim), which equals the cathode disc radius rc
RMAX_FACTOR          = 1.75
ELECTRODE_TOP_FACTOR = 1.5

# ── Validity limits (previously unenforced — see solve_theta) ────────────────
_A_MAX      = mpf("9.6")     # Eq.(5) is valid within 1% to Rc/Ra = 20 with the improved
                            # last coefficient, i.e. -alpha up to about 9.63. The older
                            # 2.8 limit belongs to Eq.(6), the approximate inverse, which
                            # is only used here as a Newton starting guess.
_THETA_MAX  = radians(mpf("89.0"))   # hard geometric limit on the theta search
_THETA_ZA   = radians(mpf("56.44"))  # above this Eq.(20) has no real root, because
                                     # ra = 1.2*rb(za) exceeds Ra. This limits the
                                     # anode-aperture formula ONLY, not the theta
                                     # solve: real high-convergence guns run at
                                     # 57-70 deg (see Yang 2006, Sharma et al.).
_R_MIN      = mpf("1.001")   # compression must exceed unity
_BISECT_TOL = mpf("1e-18")


class GunInfeasible(ValueError):
    """Raised when no valid Pierce gun exists for the requested inputs."""
    pass


def output_dir():
    """Return the folder next to the running .exe (frozen) or .py (script)."""
    if getattr(sys, 'frozen', False):
        # Running as a PyInstaller executable
        return os.path.dirname(sys.executable)
    else:
        # Running as a normal Python script
        return os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe mpmath → float
# ─────────────────────────────────────────────────────────────────────────────

def to_float(x):
    """Convert an mpf to float, refusing to silently discard an imaginary part.

    The old body was `float(mp.re(x))`, which quietly threw away the imaginary
    component whenever an out-of-range value produced a complex result (e.g.
    sqrt of a negative in Eq.12 or Eq.14) and returned a plausible-looking but
    meaningless number.
    """
    im = mp.im(x)
    if im != 0 and abs(im) > abs(mp.re(x)) * mpf("1e-20"):
        raise GunInfeasible(f"a complex intermediate appeared ({mp.nstr(x, 8)}); "
                            "inputs are outside the model's valid range.")
    return float(mp.re(x))


# ─────────────────────────────────────────────────────────────────────────────
# Langmuir-Blodgett series (Eq. 5) and Newton inversion (Eq. 6)
# ─────────────────────────────────────────────────────────────────────────────

def a_from_y(y):
    """Eq.(5): forward series — computes (-α) from y = ln(Rc/Ra)."""
    y2 = y * y
    return y + y2 * (_A1 + y * (_A2 + y * (_A3 + y * (_A4 + y * _A5))))


def y_from_a(a):
    """Invert Eq.(5) by Newton iteration. Valid to 1% for (-α) ≤ 2.8."""
    y = a - _Y0*a**2 + _Y1*a**3 - _Y2*a**4
    for _ in range(_MAX_NEWTON):
        deriv = 1 + y * (_D1 + y * (_D2 + y * (_D3 + y * (_D4 + y * _D5))))
        step  = (a_from_y(y) - a) / deriv
        y -= step
        if abs(step) < _NEWTON_TOL:
            break
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Relativistic space-charge-limited current of the spherical diode
# (Greenwood, Zhang, Lau et al., Phys. Plasmas 23, 072101 (2016))
# ─────────────────────────────────────────────────────────────────────────────
#
# Vaughan's Region I uses the classical Langmuir-Blodgett current of concentric
# spheres. At high voltage the electrons gain mass and the diode passes less
# current. Here Poisson's equation for the converging spherical diode is
# integrated with the relativistic charge density (Greenwood's formulation,
# W = phi^(3/2), phi = eV/mc^2), and the same integrator run classically, so the
# ratio of the two currents is free of discretisation error to first order.
# The classical run reproduces Langmuir-Blodgett to 0.04%. Typical size of the
# correction at Rc/Ra = 2.5: -0.19% at 14 kV, -1.9% at 150 kV, -5.9% at 500 kV.

MC2_EV = 510998.95                     # electron rest energy, eV


def _diode_anode_phi(Istar, x, rel, n=600):
    """Normalised anode potential eV/mc^2 of a space-charge-limited concentric
    spherical diode, cathode radius 1 and anode radius 1/x, for normalised
    current Istar. RK4 in t = ln(1 - r), started from the Taylor solution
    W = A (1 - r)^2 at the cathode."""
    s1 = 1.0 - 1.0 / x
    s0 = min(1e-6, s1 * 1e-4)
    A  = 9.0 * Istar / (16.0 * math.pi * math.sqrt(2.0))
    c3 = 3.0 * Istar / (8.0 * math.pi)
    def rhs(t, W, Q):
        s_ = math.exp(t); r_ = 1.0 - s_
        if W < 1e-300:
            W = 1e-300
        if rel:
            p_ = W ** (2.0 / 3.0)
            g_ = (p_ + 1.0) / math.sqrt(p_ + 2.0)
        else:
            g_ = 1.0 / math.sqrt(2.0)
        Wpp = c3 * g_ / (r_ * r_) + Q * Q / (3.0 * W) - 2.0 * Q / r_
        return -s_ * Q, -s_ * Wpp
    t0, t1 = math.log(s0), math.log(s1)
    h = (t1 - t0) / n
    W, Q, t = A * s0 * s0, -2.0 * A * s0, t0
    for _ in range(n):
        k1 = rhs(t, W, Q)
        k2 = rhs(t + h / 2, W + h / 2 * k1[0], Q + h / 2 * k1[1])
        k3 = rhs(t + h / 2, W + h / 2 * k2[0], Q + h / 2 * k2[1])
        k4 = rhs(t + h,     W + h * k3[0],     Q + h * k3[1])
        W += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        Q += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        t += h
    return W ** (2.0 / 3.0)


def relativistic_factor(V, x):
    """Relativistic / classical space-charge-limited current of the spherical
    diode at voltage V (volts) and Rc/Ra = x. Equal to 1 in the limit V -> 0."""
    x = max(float(x), 1.0 + 1e-9)
    T = float(V) / MC2_EV
    F = lambda L, rel: math.log(_diode_anode_phi(math.exp(L), x, rel)) - math.log(T)
    L_nr = 0.0
    L_nr -= 1.5 * F(L_nr, False)        # classically phi ~ I^(2/3): exact in one step
    L_nr -= 1.5 * F(L_nr, False)
    a_, fa = L_nr, F(L_nr, True)
    b_ = a_ - 1.5 * fa
    fb = F(b_, True)
    for _ in range(40):                 # secant on ln(current)
        if fb == fa or abs(fb) < 1e-13:
            break
        a_, b_, fa = b_, b_ - fb * (b_ - a_) / (fb - fa), fb
        fb = F(b_, True)
    return math.exp(b_ - L_nr)


# ─────────────────────────────────────────────────────────────────────────────
# Pierce gun synthesis — Vaughan (1981) Sections III-A through III-F
# ─────────────────────────────────────────────────────────────────────────────

def _theta_state(theta, rc, rw, muP, kf=1):
    """Return (a_neg, y, R) at a given cathode half-angle.

    kf is the relativistic factor of the cathode-anode spherical diode: its
    space-charge-limited current divided by the classical Langmuir-Blodgett
    value at the same voltage and Rc/Ra (Greenwood et al. 2016). It enters
    only Region I, Eq.(3): muP = 14.67 kf (1 - cos theta) / (-alpha)^2."""
    a = sqrt(_C14_67 * kf * (1 - cos(theta)) / muP)  # Eq.(3), relativistic
    y = y_from_a(a)                                  # Eq.(5) inverted
    R = rc * exp(-y) / rw                            # Eqs.(8,11)
    return a, y, R


def _tan_phis(theta, rc, rw, muP, kf=1):
    """Return (tan phi1, tan phi2, a, y, R) at a given half-angle.

    Eq.(14) uses |ln R| rather than ln R, following Yang, Jia & Zhu (2006),
    HPLPB 18(9), their Eq.(9'). Vaughan's form goes complex the moment the
    trial angle drives R below 1, which is the first of the two failure modes
    that paper identifies. With the absolute value the residual stays real and
    continuous across R = 1, so a root sitting just above R = 1 is not lost to
    window truncation.
    """
    a, y, R = _theta_state(theta, rc, rw, muP, kf)
    P   = 1 + y * (_D1 + y * (_D2 + y * (_CP1 + y * (_D4 + y * _CP2))))
    tp1 = sin(theta) * (1 - _C0_417 * P / a)           # Eq.(10)
    tp2 = -_C0_174 * sqrt(muP * abs(log(R)))           # Eq.(14) as amended, Eq.(9')
    return tp1, tp2, a, y, R


def _residual(theta, rc, rw, muP, kf=1):
    """|tan(phi2)| - |tan(phi1)|.  Zero at the Pierce design point."""
    tp1, tp2, a, y, R = _tan_phis(theta, rc, rw, muP, kf)
    return abs(tp2) - abs(tp1)


def yang_iterate(rc, rw, muP, theta0=None, max_iter=2000, tol=mpf("5e-4"), kf=1):
    """Improved Vaughan iteration of Yang, Jia & Zhu (2006), HPLPB 18(9).

    Their two amendments to Vaughan's loop:
      Eq.(9')  tan(phi2) = 0.17409 * sqrt(muP * |ln R|)  - survives R < 1
      Eq.(10') theta <- theta * |tan(phi2)/tan(phi1)|^(1/n)  - survives tan(phi1) < 0

    The paper notes the square root in Eq.(10') may be replaced by a fourth
    root or higher. That matters: with n = 2 this still fails on the very
    lowest-convergence guns, and escalating to n = 4 recovers them, so the
    damping order is escalated automatically here.

    Default start is the Liao (2003) initial value, theta0 = 10*sqrt(muP*rc/rw)
    degrees, which the same paper reports as extending Vaughan's original
    30*sqrt(muP) down to a convergence ratio of about 5. The method is in
    practice insensitive to the start; the paper demonstrates 0-70 degrees.

    Provided as an independent cross-check on solve_theta, not as the primary
    solver: an iteration can stall, a bracketed bisection cannot.
    """
    if theta0 is None:
        theta0 = 10 * sqrt(muP * rc / rw)              # Liao (2003)
        theta0 = min(max(float(theta0), 1.0), 70.0)
    for n_root in (2, 4, 8, 16):
        theta = radians(mpf(theta0))
        for it in range(max_iter):
            tp1, tp2, a, y, R = _tan_phis(theta, rc, rw, muP, kf)
            if tp1 == 0:
                break
            ratio = abs(tp2 / tp1)                      # Eq.(10')
            if abs(ratio - 1) <= tol:
                return theta, it, n_root
            theta *= ratio ** (mpf(1) / n_root)
            if theta >= radians(mpf("89.9")): theta = radians(mpf("89.9"))
            if theta <= radians(mpf("0.01")): theta = radians(mpf("0.01"))
    return None, max_iter, None


def solve_theta(rc, rw, muP, n_scan=300, kf=1):
    """Bracket-and-bisect solve for the cathode half-angle.

    Replaces the old `theta *= ratio**0.5` fixed-point update, which had no
    bracket, no bound and no failure path: whenever tan(phi1) approached zero
    the ratio blew up and theta diverged (values above 1e20 degrees were
    observed), and whenever no root existed the loop simply ran out of
    iterations and returned the last garbage state.

    Returns (theta, all_roots, theta_hi, warnings).
    Raises GunInfeasible when no design point exists.
    """
    warn = []
    Rmax = rc / rw                      # limiting compression as theta -> 0

    if Rmax <= 1:
        raise GunInfeasible(
            f"cathode radius rc = {float(rc)*IN_TO_MM:.4f} mm does not exceed beam "
            f"waist rw = {float(rw)*IN_TO_MM:.4f} mm, so compression R <= 1 for every "
            f"theta. Increase I or Jc, or reduce rw.")

    # ── Upper bound on theta: first violation of R > 1, a <= 2.8 or Eq.(20) ──
    lo = radians(mpf("0.05"))
    hi = _THETA_MAX
    theta_hi, limiter = hi, "geometry limit (89 deg)"
    prev = lo
    for i in range(1, 401):
        t = lo + (hi - lo) * mpf(i) / 400
        a, y, R = _theta_state(t, rc, rw, muP, kf)
        if a >= _A_MAX:
            theta_hi, limiter = prev, "Langmuir series limit (-alpha = 2.8)"; break
        prev = t
    if theta_hi <= lo:
        raise GunInfeasible("the valid theta window is empty; compression collapses "
                            "immediately. Check rw against rc.")

    # ── Scan for sign changes, then bisect each bracket ─────────────────────
    ts = [lo + (theta_hi - lo) * mpf(i) / n_scan for i in range(n_scan + 1)]
    fs = [_residual(t, rc, rw, muP, kf) for t in ts]

    roots = []
    for i in range(n_scan):
        fa, fb = fs[i], fs[i + 1]
        if fa == 0:
            roots.append(ts[i]); continue
        if fa * fb < 0:
            a_, b_ = ts[i], ts[i + 1]
            for _ in range(200):
                mid = (a_ + b_) / 2
                fm  = _residual(mid, rc, rw, muP, kf)
                if fa * fm <= 0:
                    b_ = mid
                else:
                    a_, fa = mid, fm
                if b_ - a_ < _BISECT_TOL:
                    break
            roots.append((a_ + b_) / 2)

    if not roots:
        raise GunInfeasible(
            f"no theta in (0, {float(degrees(theta_hi)):.2f} deg] satisfies Eq.(10) = Eq.(14) "
            f"(window closed by {limiter}). Microperveance {float(muP):.3f} is outside the "
            f"Pierce regime for this rc/rw = {float(Rmax):.2f}. Lower I, raise V, or reduce rw.")

    # With the wider window a second root can appear at larger theta where the
    # beam is already below the requested waist at the anode, so no waist forms
    # downstream. Keep only roots that give a real compression.
    physical = [t for t in roots if _theta_state(t, rc, rw, muP, kf)[2] > 1]
    if not physical:
        raise GunInfeasible(
            "every solution leaves the beam at the anode smaller than the requested waist, "
            "so no waist forms downstream. Reduce rw, or raise I or Jc.")
    roots = physical

    if len(roots) > 1:
        warn.append(
            f"{len(roots)} roots found ({', '.join(f'{float(degrees(t)):.2f}' for t in roots)} deg); "
            f"taking the largest. This happens only for rc/rw < 2.2 (here {float(Rmax):.2f}), "
            f"a very low-compression design — verify against Vaughan's worked example.")

    theta = roots[-1]
    a, y, R = _theta_state(theta, rc, rw, muP, kf)
    if R <= 1:
        raise GunInfeasible(
            f"the only root lies at R = {float(R):.4f} <= 1, i.e. the beam at the anode is "
            "already smaller than the requested waist, so no downstream waist exists. "
            "Reduce rw, or raise I or Jc.")
    if a > mpf("8.6"):
        warn.append(f"-alpha = {float(a):.3f} is close to the 9.6 series limit; "
                    "Eq.(5) inversion is losing accuracy.")
    return theta, roots, theta_hi, warn


def pierce_synthesis(V, I, rw_in, Jc_cm2, max_iter=50, aperture_factor=1.2):
    """
    Synthesise a Pierce gun from four input parameters.

    Parameters
    ----------
    V       : beam voltage            (Volts)
    I       : beam current            (Amperes)
    rw_in   : beam waist radius       (inches)
    Jc_cm2  : cathode current density (A/cm²)

    Returns a dict of all design parameters from Vaughan (1981) Sections III-E & III-F.
    """
    V, I, rw, Jc = mpf(V), mpf(I), mpf(rw_in), mpf(Jc_cm2)

    rc  = sqrt(I / (pi * Jc)) / CM    # Eq.(1): cathode disc radius [inches]
    muP = _C1e6 * I / V**1.5          # Eq.(2): microperveance

    # Region I with the relativistic diode current (Greenwood 2016). The factor
    # depends on Rc/Ra, which depends on theta, so solve, update, re-solve until
    # it settles; it changes slowly, so three or four passes suffice.
    kf = mpf(1)
    for _ in range(10):
        theta, roots, theta_hi, warnings = solve_theta(rc, rw, muP, kf=kf)
        a_neg, y, R = _theta_state(theta, rc, rw, muP, kf)
        kf_new = mpf(relativistic_factor(V, float(exp(y))))
        if abs(kf_new - kf) < mpf("1e-12"):
            kf = kf_new
            break
        kf = kf_new
    theta, roots, theta_hi, warnings = solve_theta(rc, rw, muP, kf=kf)
    a_neg, y, R = _theta_state(theta, rc, rw, muP, kf)
    converged = True                 # bisection cannot return a non-root
    it = len(roots)

    # Independent cross-check against Yang, Jia & Zhu (2006). Two solvers of
    # different character agreeing is worth the handful of extra evaluations.
    theta_y, _, n_root = yang_iterate(rc, rw, muP, kf=kf)
    if theta_y is None:
        cross_check = None
        warnings.append("Yang (2006) cross-check did not converge; bisection result "
                        "is unconfirmed by a second method.")
    else:
        cross_check = to_float(degrees(theta_y))
        gap = abs(cross_check - to_float(degrees(theta)))
        if gap > 0.1:
            warnings.append(
                f"solvers disagree: bisection {float(degrees(theta)):.3f} deg vs Yang "
                f"iteration {cross_check:.3f} deg (gap {gap:.3f} deg). Likely multiple "
                "roots; treat the geometry as provisional.")

    # ── Section III-E: complete gun geometry ─────────────────────────────────
    Rc        = rc / sin(theta)                              # Eq.(15)
    exp_neg_y = exp(-y)
    Ra        = Rc * exp_neg_y                               # Eq.(16)
    rb_za     = rc * exp_neg_y                               # Eq.(17)
    # Eq.(18). Vaughan's empirical clearance factor is 1.2. It is exposed here
    # because it only moves the aperture radius and the aperture plane za; the
    # half angle, Rc, Ra, the cathode-anode spacing and the throw are untouched.
    ra        = mpf(str(aperture_factor)) * rb_za
    zac       = Rc - Ra                                      # Eq.(19)
    disc      = Ra**2 - ra**2
    if disc > 0:
        za    = Rc - sqrt(disc)                              # Eq.(20)
        za_ok = True
    else:
        # theta > 56.44 deg: the empirical ra = 1.2*rb(za) exceeds Ra, so Eq.(20)
        # has no real root. The old code hid this with max(..., 0), silently
        # returning za = Rc. Everything else in the design is still valid.
        za, za_ok = Rc, False
        warnings.append(
            f"theta = {float(degrees(theta)):.2f} deg exceeds 56.44 deg, so Eq.(20) has no "
            "real root and za is not defined by the 1.2 empirical aperture factor. "
            "za is reported as Rc (a placeholder); use an improved anode-aperture "
            "formula for high-convergence guns.")
    Rm1       = R - 1
    if Rm1 > 0:
        Z     = (_C11_0*Rm1**mpf("0.5") + _C1_32*Rm1
                 + _C0_061*Rm1**2 - _C0_001*Rm1**3)          # Eq.(12)
        zw    = zac + rw * Z / sqrt(muP)                     # Eq.(21)
    else:
        Z, zw = mpf(0), zac
        warnings.append("R <= 1, so Eq.(12) has no real root and the throw zw is undefined.")
    if Rm1 > 30:
        # Eq.(12) is an empirical fit; its cubic term turns the curve over near
        # R-1 ~ 33, so Z stops increasing and then falls. Beyond that the throw
        # is extrapolation, not fit.
        warnings.append(f"R - 1 = {float(Rm1):.1f} is past the turning point of the Eq.(12) "
                        "fit (~33); the throw zw is an extrapolation.")

    # ── Section III-F: spherical-aberration correction ────────────────────────
    sin_over_k = sin(theta) / _C0_905
    if sin_over_k <= 1:
        thetaT, RcT, correction_applied = asin(sin_over_k), _C0_905 * Rc, True
    else:
        thetaT, RcT, correction_applied = theta, Rc, False

    return dict(
        V=to_float(V),          I=to_float(I),
        rw=to_float(rw),        Jc=to_float(Jc),
        muP=to_float(muP),      rc=to_float(rc),
        theta=to_float(degrees(theta)),
        Rc=to_float(Rc),        Ra=to_float(Ra),
        zac=to_float(zac),      za=to_float(za),
        rb_za=to_float(rb_za),  R=to_float(R),
        ra=to_float(ra),        zw=to_float(zw),
        thetaT=to_float(degrees(thetaT)),
        RcT=to_float(RcT),
        correction_applied=correction_applied,
        rel_factor=to_float(kf),
        aperture_factor=float(aperture_factor),
        iters=it, converged=converged,
        warnings=warnings, n_roots=len(roots), za_valid=za_ok,
        theta_crosscheck=cross_check, yang_damping=n_root,
        theta_hi=to_float(degrees(theta_hi)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Input helper
# ─────────────────────────────────────────────────────────────────────────────

def ask(prompt, default=None, lo=None, hi=None):
    """Prompt with an optional default value and optional bounds."""
    while True:
        suffix = f"  [default: {default}]" if default is not None else ""
        raw = input(f"  {prompt}{suffix}: ").strip()
        if raw == "" and default is not None:
            return float(default)
        if raw == "" and default is None:
            print("    ✗  Please enter a number."); continue
        try:
            v = float(raw)
        except ValueError:
            print("    ✗  Please enter a number."); continue
        if lo is not None and v < lo:
            print(f"    ✗  Must be ≥ {lo}."); continue
        if hi is not None and v > hi:
            print(f"    ✗  Must be ≤ {hi}."); continue
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Single-design display
# ─────────────────────────────────────────────────────────────────────────────

def print_results(r):
    W = 62

    def sep(title):
        pad = (W - len(title) - 2) // 2
        print("─"*pad + f" {title} " + "─"*(W - pad - len(title) - 2))

    def row(label, value, unit=""):
        print(f"  {label:<36} {value}  {unit}")

    print(f"\n{'═'*W}\n         PIERCE GUN SYNTHESIS RESULTS\n         Vaughan (1981), IEEE Trans. Electron Devices\n{'═'*W}")

    sep("Input Parameters")
    row("Beam voltage  V",             f"{r['V']/1000:.4g}",        "kV")
    row("Beam current  I",             f"{r['I']:.4g}",             "A")
    row("Waist radius  rw",            f"{r['rw_mm']:.5f}",         "mm")
    row("Cathode current density  Jc", f"{r['Jc']:.4g}",            "A/cm²")

    sep("Derived Constants")
    row("Microperveance  μP",          f"{r['muP']:.4f}",              "μP")
    row("Cathode disc radius  rc",     f"{r['rc']*IN_TO_MM:.5f}",      "mm")

    sep("Pierce Design")
    row("Convergence half-angle  θ",          f"{r['theta']:.3f}",         "deg")
    row("Cathode spherical radius  Rc",       f"{r['Rc']*IN_TO_MM:.5f}",   "mm")
    row("Anode sphere radius  Ra",            f"{r['Ra']*IN_TO_MM:.5f}",   "mm")
    row("Anode-cathode spacing  zac",         f"{r['zac']*IN_TO_MM:.5f}",  "mm")
    row("Anode plane position  za",           f"{r['za']*IN_TO_MM:.5f}",   "mm")
    row("Beam radius at anode  rb(za)",       f"{r['rb_za']*IN_TO_MM:.5f}","mm")
    row("Normalised compression  R",          f"{r['R']:.5f}",             "")
    row("Relativistic diode factor  kf",      f"{r['rel_factor']:.5f}",    "")
    row("Anode aperture radius  ra",          f"{r['ra']*IN_TO_MM:.5f}",   "mm")
    row("Throw (cathode vertex to waist) zw", f"{r['zw']*IN_TO_MM:.5f}",   "mm")

    sep("Spherical-Aberration Correction  (k = 0.905)")
    if r['correction_applied']:
        row("Corrected half-angle  θT",      f"{r['thetaT']:.3f}",           "deg")
        row("Corrected cathode radius  RcT", f"{r['RcT']*IN_TO_MM:.5f}",     "mm")
    else:
        print(f"  ⚠  Correction inapplicable — uncorrected θ > 64.8°\n"
              f"     Uncorrected values: θ = {r['theta']:.3f}°,  Rc = {r['Rc']*IN_TO_MM:.5f} mm")

    sep("Solver Diagnostics")
    row("Status",     "CONVERGED (bracketed bisection)" if r['converged'] else "FAILED")
    row("Roots found", str(r.get('n_roots', 1)))
    row("Search window", f"0 – {r.get('theta_hi', 0):.2f} deg")
    cc = r.get('theta_crosscheck')
    row("Cross-check (Yang 2006)", f"{cc:.3f} deg" if cc is not None else "did not converge")
    for w in r.get('warnings', []):
        print(f"  ⚠  {w}")
    print("═" * W + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — free design parameters, coordinate table, diagram
# ─────────────────────────────────────────────────────────────────────────────

def _simpson(f, a, b, n):
    """Composite Simpson rule (n even); same rule the browser version uses."""
    if b <= a:
        return 0.0
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += f(a + i * h) * (4 if i % 2 else 2)
    return s * h / 3


def beam_edge_table(rw, rb_za, za, zw, n=400):
    """Beam edge after the anode from Vaughan's spreading law, dr/dz = -0.17409 sqrt(muP ln(r/rw)).
    Returns [(s, r)], s = distance past the anode plane, from the anode (s = 0) to the waist."""
    u0 = math.sqrt(max(math.log(rb_za / rw), 0.0))
    if u0 <= 0:
        return [(0.0, rw)]
    f   = lambda t: math.exp(t * t)
    tot = _simpson(f, 0.0, u0, 400)
    out = []
    for i in range(n + 1):
        u = u0 * (1 - i / n)
        out.append(((zw - za) * _simpson(f, u, u0, 120) / tot, rw * math.exp(u * u)))
    return out


def drift_tube_clearance(table, L_waist, s12, ra, r12, s_max):
    """Smallest wall-radius / beam-radius ratio from the anode plane to the back wall.
    Wall: straight from (0, ra) to (s12, r12), then flat at r12. The beam is
    symmetric about the waist, so past the waist the table is mirrored."""
    def wall(s_):
        return ra + (r12 - ra) * s_ / s12 if s_ < s12 else r12
    worst = float('inf')
    for s_, rb in table:
        for sv in (s_, 2 * L_waist - s_):
            if 0 <= sv <= s_max:
                worst = min(worst, wall(sv) / rb)
    return worst


def get_design_params_mode1(r):
    """
    Prompt for the seven free design parameters.
    All lengths in mm.  Returns a dict with values in mm / radians.

    H is asked first (before all other parameters).
    Rmax and Zmax are specified as multiples of H beyond reference points:
      - Rmax = RMAX_FACTOR x H2 (1.75 H2, H2 = height of pt 2)   (pts 7 & 8 share this r-coordinate)
      - focus electrode tip (pt 6) at ELECTRODE_TOP_FACTOR x H2 (1.5 H2)
      - Zmax = d + 1.3 x the throw                       (pts 13 & 14 share this z-coordinate)
    """
    # Convert synthesis results from inches → mm
    Rc_mm  = r['Rc']    * IN_TO_MM
    Ra_mm  = r['Ra']    * IN_TO_MM
    rc_mm  = r['rc']    * IN_TO_MM
    ra_mm  = r['ra']    * IN_TO_MM
    za_mm  = r['za']    * IN_TO_MM
    zw_mm  = r['zw']    * IN_TO_MM
    rb_za_mm = r['rb_za'] * IN_TO_MM
    zac_mm = Rc_mm - Ra_mm
    theta_rad = math.radians(r['theta'])
    # Vaughan (1981), Sec. III-F: analyse with theta, Rc but BUILD the cathode with
    # the aberration-corrected thetaT, RcT. As in his Fig. 2, the cathode rim stays
    # at rc and the anode stays put relative to the rim; only the cathode curves
    # more deeply. With the vertex kept at z = d, everything from the anode on
    # moves downstream by Dz, the extra depth of the corrected cathode.
    thetaT_rad = math.radians(r['thetaT'])
    RcT_mm     = r['RcT'] * IN_TO_MM
    Dz         = RcT_mm * (1 - math.cos(thetaT_rad)) - Rc_mm * (1 - math.cos(theta_rad))

    print(f"\n  ── Geometry design parameters ──\n")
    print(f"  Reference:  Rc = {Rc_mm:.3f} mm   Ra = {Ra_mm:.3f} mm   "
          f"rc = {rc_mm:.3f} mm   za = {za_mm:.3f} mm\n")

    # ── H first — needed to express Rmax and Zmax in grid units ──────────────
    H_z = ask("H_z  — grid spacing along z-axis  [mm]",
              default=0.05, lo=0.001)
    H_r = ask("H_r  — grid spacing along r-axis  [mm]",
              default=0.05, lo=0.001)
    print()

    d  = ask("d    — cathode vertex offset from origin  [mm]",
             default=0.2, lo=0.01)
    # Default 0 puts pts 3 and 4 directly above pt 1, as in the professor's second deck
    l  = ask("l    — cathode body axial depth  [mm]  (0 = pts 3,4 directly above pt 1)",
             default=0.0, lo=0.0, hi=d - 0.001)

    # ── Focus electrode ─────────────────────────────────────────────────────────
    #
    # Pierce's condition: outside a Child-law beam the zero-volt equipotential
    # leaves the beam edge at 67.5 deg (V ~ z^(4/3) on the edge, and
    # 67.5 x 4/3 = 90). The focus (beam-forming) electrode, held at cathode
    # voltage, is placed along it: a straight face starting beside the cathode
    # rim and making angle psi with the beam edge, i.e. leaning (90 - psi) deg
    # forward of the cathode surface. Exact for a planar cathode; for a
    # converging gun it is the standard starting value that simulation refines
    # (a TRAK/MAGIC-refined 130 kV design moved to 60.5 deg).
    #
    # Construction (replaces the old one, where point 6 was forced onto the
    # cathode sphere, which tied the electrode's angle to its length and pushed
    # its start far out from the cathode rim):
    #   k1    : radial gap between cathode rim and electrode start, as a
    #           fraction of Rc. Keep it small - the electrode should hug the rim.
    #   psi   : electrode angle to the beam edge, fixed at PIERCE_ANGLE (67.5).
    #   beta  : anode sphere rim angle; follows from pt 9 lying below pt 8.
    #   g     : axial gap between pts 7 and 8 on the outer wall (stored as 'alpha').
    # The cathode as built is the corrected one, so the electrode works off thetaT, RcT.
    print(f"\n  Cathode built with the aberration-corrected values (Vaughan Sec. III-F):")
    print(f"  thetaT = {math.degrees(thetaT_rad):.4f} deg,  RcT = {RcT_mm:.4f} mm  "
          f"(rim stays at rc = {rc_mm:.4f} mm; anode moves downstream by {Dz:.4f} mm)")
    sin_theta = math.sin(thetaT_rad)
    th_deg    = math.degrees(thetaT_rad)
    k1_max    = round(0.99 - sin_theta, 4)
    print(f"\n  Note: k1 is the radial gap between cathode rim and focus electrode, as a fraction of RcT")
    print(f"  (gap = k1 x RcT). Keep it small; must be < {k1_max:.4f}")
    k1 = ask("k1   — focus electrode gap multiplier",
             default=0.02, lo=0.001, hi=k1_max)
    sin_delta = sin_theta + k1
    delta     = math.asin(sin_delta)
    print(f"  -> gap = {k1 * RcT_mm:.4f} mm")

    # psi must exceed theta (electrode rises as it runs forward) and stay
    # flatter than the cathode sphere at point 5 (psi < 90 + theta - delta),
    # otherwise the face is recessed behind the cathode surface.
    psi_lo = th_deg + 1.0
    psi_hi = min(89.5, 90.0 + th_deg - math.degrees(delta) - 0.5)
    if psi_lo >= psi_hi:
        raise ValueError(f"k1 = {k1} leaves no valid electrode angle; reduce k1.")
    psi = PIERCE_ANGLE
    if not (psi_lo <= psi <= psi_hi):
        if psi < psi_lo:
            raise ValueError(
                f"theta = {th_deg:.2f} deg is too large for a {psi} deg focus electrode: the face "
                "would not rise away from the beam. This gun is outside what the Pierce angle covers.")
        raise ValueError(
            f"k1 = {k1} starts the electrode too far round the cathode sphere for a {psi} deg face; "
            "reduce k1.")
    print(f"\n  Focus electrode angle to beam edge fixed at the Pierce value, {psi} deg")
    slope = math.tan(math.radians(psi) - thetaT_rad)         # dr/dz of the electrode face

    # ── Focus electrode tip, outer wall, pts 7-8 gap and anode rim ──────────
    # The electrode runs from pt 5 at the Pierce angle until it reaches
    # ELECTRODE_TOP_FACTOR x H2 (pt 6), H2 = height of pt 2 = rc. Pt 7 sits directly
    # above it on the outer wall, Rmax = RMAX_FACTOR x H2. Pt 8 is on the same wall
    # a gap g further along (the insulating gap between cathode and anode
    # potentials). Pt 9 sits directly below pt 8, on the anode sphere, so the
    # anode rim angle beta is no longer an input: it follows from g.
    z5_rel = RcT_mm * (1 - math.cos(delta))                  # z measured from cathode vertex
    r5     = RcT_mm * sin_delta
    H2     = RcT_mm * sin_theta                              # height of pt 2 (= rc)
    r6     = ELECTRODE_TOP_FACTOR * H2
    Rmax   = RMAX_FACTOR * H2
    if r6 <= r5:
        raise ValueError(f"The electrode would end at {r6:.4f} mm, below its own start at {r5:.4f} mm. "
                         f"Reduce k1 below {(ELECTRODE_TOP_FACTOR - 1) * H2 / RcT_mm:.4f}.")
    z6_rel = z5_rel + (r6 - r5) / slope
    zmax_rel = 1.3 * zw_mm                                   # Zmax = d + 1.3 x throw
    print(f"\n  H2 = height of pt 2 = {H2:.4f} mm")
    print(f"  Focus electrode ends at {ELECTRODE_TOP_FACTOR:g} x H2 = {r6:.4f} mm (pt 6); "
          f"outer wall Rmax = {RMAX_FACTOR:g} x H2 = {Rmax:.4f} mm (pts 7, 8)")
    print(f"  -> electrode runs from r = {r5:.3f} to {r6:.3f} mm, "
          f"length {math.hypot(z6_rel - z5_rel, r6 - r5):.3f} mm, tip at z = {d + z6_rel:.4f} mm")

    # Limits on beta: beyond the aperture (so the aperture sits on the sphere),
    # no further than 89.9 deg, and pt 9 must stay below the outer wall.
    if ra_mm < Ra_mm:
        beta_lo = math.degrees(math.asin(ra_mm / Ra_mm)) + 1.0
    else:
        beta_lo = 10.0                                       # za invalid (theta > 56.44)
    beta_hi = 89.9
    if Ra_mm * 0.98 < Rmax:
        pass
    else:
        beta_hi = min(beta_hi, math.degrees(math.asin(0.98 * Rmax / Ra_mm)))
    if beta_lo >= beta_hi:
        raise ValueError("The anode sphere reaches above the outer wall at 1.75 x H2 before it clears "
                         "the aperture; pt 9 cannot sit below pt 8.")
    def _z9(beta_deg):                                       # z of the sphere rim, from cathode vertex
        return Dz + Rc_mm - Ra_mm * math.cos(math.radians(beta_deg))
    def _beta_for(g):
        c = (Dz + Rc_mm - (z6_rel + g)) / Ra_mm
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def _cross(p1, p2, q1, q2):
        """True if segments p1-p2 and q1-q2 intersect."""
        def o(a, b, c):
            return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
        d1, d2 = o(q1, q2, p1), o(q1, q2, p2)
        d3, d4 = o(p1, p2, q1), o(p1, p2, q2)
        return (d1 * d2 < 0) and (d3 * d4 < 0)
    electrode = [((z5_rel, r5), (z6_rel, r6)), ((z6_rel, r6), (z6_rel, Rmax))]
    def _face_ok(g):
        b  = math.radians(_beta_for(g))
        P9 = (z6_rel + g, Ra_mm * math.sin(b))
        return not any(_cross(a, c, P9, (Dz + za_mm, ra_mm)) for a, c in electrode)

    gap_lo = max(_z9(beta_lo) - z6_rel, 1e-3)
    gap_hi = min(_z9(beta_hi) - z6_rel, zmax_rel - z6_rel - 1e-3)
    if gap_hi <= gap_lo:
        raise ValueError(f"The electrode tip (z = {d + z6_rel:.3f} mm) is too far downstream for pt 9 "
                         f"to sit on the anode sphere below pt 8.")
    step = (gap_hi - gap_lo) / 400
    while gap_lo < gap_hi and not _face_ok(gap_lo):
        gap_lo += step
    if gap_lo >= gap_hi:
        raise ValueError("No gap between pts 7 and 8 keeps the anode face clear of the electrode.")
    # Default the gap to the cathode-anode distance; if that does not fit the
    # allowed window, fall back to its midpoint and say so.
    alpha_default = round(zac_mm, 4)
    if not (gap_lo <= alpha_default <= gap_hi):
        alpha_default = round(gap_lo + 0.5 * (gap_hi - gap_lo), 4)
        print(f"\n  Note: the cathode-anode distance ({zac_mm:.4f} mm) lies outside the "
              f"allowed range for the pts 7-8 gap, so the default is the midpoint of that range instead.")
    print(f"\n  Note: the pts 7-8 gap must lie in ({gap_lo:.4f}, {gap_hi:.4f}) mm "
          f"(pt 9 is placed directly below pt 8, so the gap sets β)")
    alpha = ask("g    — axial gap between pts 7 and 8 (electrode wall to anode wall)  [mm]",
                default=alpha_default, lo=gap_lo, hi=gap_hi)
    beta     = _beta_for(alpha)
    beta_rad = math.radians(beta)
    r9       = Ra_mm * math.sin(beta_rad)
    print(f"  -> pt 8 at z = {d + z6_rel + alpha:.4f} mm; pt 9 directly below at r = {r9:.4f} mm, "
          f"anode rim angle β = {beta:.3f} deg")
    z11 = d + Dz + Rc_mm

    # ── Anode nose: pts 10, 11 and 12 ───────────────────────────────────────
    # Pt 11 is the anode aperture edge exactly as Vaughan defines it, (za, ra)
    # on the anode sphere, so the hole really is ra = 1.2 x rb(za) wide in the
    # anode plane. Pt 10 sits directly above it, a whole number of H_r steps
    # higher (whole steps so grid snapping keeps the gap exact). Pt 12: the
    # drift-tube wall leaves the aperture parallel to the beam edge, whose exit
    # slope after the anode lens is tan(phi2) = 0.17409 sqrt(muP ln R), Eq. (9).
    gap_max = int(math.floor((r9 - ra_mm) / H_r - 1e-9))
    if gap_max < 1:
        raise ValueError(
            f"Pt 9 (r = {r9:.4f} mm) is less than one grid step above the aperture "
            f"(ra = {ra_mm:.4f} mm), so pt 10 has no room. Increase the pts 7-8 gap.")
    gap_default = min(gap_max, max(1, int(round(0.5 * (r9 - ra_mm) / H_r))))
    print(f"\n  Pt 11 is the aperture edge (za, ra) = ({d + Dz + za_mm:.4f}, {ra_mm:.4f}) mm.")
    print(f"  Pt 10 sits directly above it; the gap is a whole number of H_r steps, 1 to {gap_max}")
    print(f"  (the top of the range is just below pt 9). Default: halfway to pt 9, "
          f"{gap_default} steps = {gap_default * H_r:.4f} mm.")
    nH_gap = int(ask("nH_gap — pts 11 to 10 gap, in H_r steps",
                     default=float(gap_default), lo=1.0, hi=float(gap_max)))
    r10 = ra_mm + nH_gap * H_r
    print(f"  -> r(pt 10) = {ra_mm:.4f} + {nH_gap} x {H_r} = {r10:.4f} mm")
    P10, P11 = (Dz + za_mm, r10), (Dz + za_mm, ra_mm)
    P9 = (z6_rel + alpha, r9)
    if any(_cross(a, c, P9, P10) or _cross(a, c, P10, P11) for a, c in electrode):
        raise ValueError("The anode face through pt 10 cuts the focus electrode; use a smaller pts 11-10 gap.")

    tan_phi2 = 0.17409 * math.sqrt(r['muP'] * abs(math.log(r['R'])))
    phi2_deg = math.degrees(math.atan(tan_phi2))
    r12      = ra_mm - (Rc_mm - za_mm) * tan_phi2
    if r12 <= 0:
        raise ValueError("A wall parallel to the beam edge reaches the axis before pt 12; this gun is too long for the rule.")
    rw_mm   = r['rw'] * IN_TO_MM
    table   = beam_edge_table(rw_mm, rb_za_mm, za_mm, zw_mm)
    s_max   = (d + 1.3 * zw_mm) - (d + Dz + za_mm)       # back wall, measured from the anode plane
    clear   = drift_tube_clearance(table, zw_mm - za_mm, Rc_mm - za_mm, ra_mm, r12, s_max)
    print(f"\n  Drift-tube wall leaves the aperture at the beam exit angle phi2 = {phi2_deg:.3f} deg")
    print(f"  -> r(pts 12, 13) = {ra_mm:.4f} - ({Rc_mm:.4f} - {za_mm:.4f}) x {tan_phi2:.5f} = {r12:.4f} mm")
    print(f"  Smallest wall / beam radius ratio from the anode to the back wall: {clear:.3f}")
    if clear <= 1.0:
        print("  WARNING: the drift-tube wall cuts into the predicted beam. Raise the aperture factor.")

    # Rmax is fixed at RMAX_FACTOR x Rc; make sure it really is the top of the model.
    r_others = max(r6, rc_mm + k1 * RcT_mm, r9, r10)
    if Rmax <= r_others:
        raise ValueError(f"Rmax = {RMAX_FACTOR:g} x H2 = {Rmax:.4f} mm is not above every other point "
                         f"({r_others:.4f} mm). Lower β so the anode rim (pt 9) sits lower.")
    print(f"\n  Rmax = {RMAX_FACTOR:g} x H2 = {Rmax:.4f} mm;  pts 6-7 gap = {Rmax - r6:.4f} mm")

    # ── Zmax: fixed at 1.3 x the throw ───────────────────────────────
    Zmax = d + 1.3 * zw_mm
    print(f"\n  Zmax is fixed at 1.3 x the throw:  {d} + 1.3 x {zw_mm:.4f} = {Zmax:.4f} mm")
    if Zmax <= z11:
        print(f"  WARNING: that is not beyond z(pt 12) = {z11:.4f} mm, so points 13 and 14 "
              "would sit ahead of point 12.")

    print()
    return dict(
        d=d, l=l, k1=k1, psi_target=psi, psi_deg=psi,
        nH_gap=nH_gap, r6=r6, Dz=Dz, r10=r10, r12=r12, phi2_deg=phi2_deg, clearance=clear,
        alpha=alpha, beta_deg=beta, beta_rad=beta_rad,
        Rmax=Rmax, Zmax=Zmax, H_z=H_z, H_r=H_r,
    )


def compute_gun_points_mm(r, dp):
    """
    Compute the 15 geometry points in mm (pts 0-14).
    r  : result dict from pierce_synthesis  (lengths in inches)
    dp : design parameter dict from get_design_params_mode1  (lengths in mm)

    Point 11: same z as pt 10, r = r(pt 10) - 3*H_r  (minimum 3H separation enforced).
    """
    # Convert synthesis results inches -> mm
    Rc    = r['Rc']    * IN_TO_MM
    Ra    = r['Ra']    * IN_TO_MM
    rc    = r['rc']    * IN_TO_MM
    ra    = r['ra']    * IN_TO_MM
    rb_za = r['rb_za'] * IN_TO_MM
    za    = r['za']    * IN_TO_MM
    # Cathode as built: aberration-corrected thetaT, RcT (Vaughan Sec. III-F)
    theta = math.radians(r['thetaT'])
    RcT   = r['RcT']   * IN_TO_MM
    Dz    = dp.get('Dz', 0.0)                 # anode side shifted by the extra cathode depth

    d     = dp['d'];      l     = dp['l']
    k1    = dp['k1'];     alpha = dp['alpha']
    beta  = dp['beta_rad']
    Rmax  = dp['Rmax'];   Zmax  = dp['Zmax']
    H_r   = dp['H_r']

    # Cache trig values — each function is called multiple times below
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_delta = sin_theta + k1
    if sin_delta >= 1.0:
        raise ValueError(f"sinTheta + k1 = {sin_delta:.4f} >= 1.  Reduce k1.")
    delta     = math.asin(sin_delta)
    cos_delta = math.cos(delta)
    sin_beta  = math.sin(beta)
    cos_beta  = math.cos(beta)

    psi_rad = math.radians(dp['psi_deg'])
    slope   = math.tan(psi_rad - theta)                   # focus electrode dr/dz

    # Point coordinates (mm)
    z0,  r0  = 0.0,                       0.0
    z1,  r1  = d,                         0.0
    z2,  r2  = d + RcT*(1 - cos_theta),  RcT*sin_theta
    z3,  r3  = d - l,                     rc
    z4,  r4  = d - l,                     rc + k1*RcT
    z5,  r5  = d + RcT*(1 - cos_delta),  RcT*sin_delta
    # Electrode ends at r6 = 1.5 Rc; pt 7 directly above on the outer wall;
    # pt 8 a gap alpha further along; anode face straight from pt 8 to pt 9.
    r6       = dp['r6']
    z6       = z5 + (r6 - r5) / slope
    z7,  r7  = z6,                        Rmax
    z8,  r8  = z7 + alpha,               Rmax
    z9,  r9  = z8,                        Ra*sin_beta   # directly below pt 8, on the anode sphere
    # Pt 11: the aperture edge (za, ra) on the anode sphere; pt 10 directly above it
    z11, r11 = d + Dz + za,               ra
    z10, r10 = z11,                       ra + dp['nH_gap']*dp['H_r']
    # Old pts 11, 12, 13 renumbered to 12, 13, 14
    # Drift-tube wall leaves pt 11 parallel to the beam edge (slope tan phi2)
    z12, r12 = d + Dz + Rc,                    dp['r12']
    z13, r13 = Zmax,                           dp['r12']
    z14, r14 = Zmax,                           0.0

    pts = [
        ( 0, "True origin",             z0,  r0 ),
        ( 1, "Cathode vertex",          z1,  r1 ),
        ( 2, "Cathode rim",             z2,  r2 ),
        ( 3, "Cathode body rear-inner", z3,  r3 ),
        ( 4, "Cathode body rear-outer", z4,  r4 ),
        ( 5, "Focus elec. base",        z5,  r5 ),
        ( 6, "Focus elec. tip",         z6,  r6 ),
        ( 7, "Focus elec. outer tip",   z7,  r7 ),
        ( 8, "Drift-tube front outer",  z8,  r8 ),
        ( 9, "Anode sphere rim",        z9,  r9 ),
        (10, "Anode face above aperture",z10, r10),
        (11, "Anode aperture edge",     z11, r11),
        (12, "Drift-tube inner start",  z12, r12),
        (13, "Drift-tube inner end",    z13, r13),
        (14, "Drift-tube back, on axis",z14, r14),
    ]

    # Focus electrode angle, measured from the drawn points rather than the
    # design formula, so the printout reports what was actually built.
    fz, fr   = z6 - z5, r6 - r5
    fn       = math.hypot(fz, fr)
    ez, er   = cos_theta, -sin_theta          # beam edge, rim -> centre of curvature
    psi_deg  = math.degrees(math.acos(max(-1.0, min(1.0, (fz*ez + fr*er) / fn))))

    meta = dict(
        psi_deg      = psi_deg,
        elec_len     = fn,
        psi_target   = dp.get('psi_target'),
        arc_centre_z = d + RcT,
        Rc_mm        = RcT,
        Ra_mm        = Ra,
        delta_deg    = math.degrees(delta),
        beta_deg     = dp['beta_deg'],
        theta_deg    = r['thetaT'],
    )
    return pts, meta


def print_gun_points(pts, meta, V=0.0, H_z=0.2, H_r=0.2, Rc_mm_val=0.0, rc_mm_val=0.0, d_val=0.0, txt_path=None):
    W = 125
    def sep(ch="─"): print(ch * W)

    arc_cz = meta['arc_centre_z']
    Rc_mm  = meta['Rc_mm']

    # code column: 0 = curved arc, -1 = empty space, 1 = straight line
    # Only arc 1->2 is curved; gaps are between pts 3->4 and pts 7->8
    arc_starts = {
        1: (arc_cz, 0.0, Rc_mm, "RcT"),
    }
    code_minus1 = {3, 7}

    def row_extras(pid):
        if pid in arc_starts:
            zc, rc_, rad, formula = arc_starts[pid]
            return 0, zc, rc_, rad, formula
        elif pid in code_minus1:
            return -1, 0.0, 0.0, 0.0, ""
        else:
            return 1, 0.0, 0.0, 0.0, ""

    def voltage_for(pid):
        return 0.0 if pid <= 7 else V

    sep("═")
    print("  GEOMETRY POINTS")
    sep("═")
    print(f"  {'Pt':<4}  {'Label':<30}  {'z (mm)':>11}  {'r (mm)':>11}  "
          f"{'Code':>5}  {'Voltage (V)':>12}  {'z_center':>11}  {'r_center':>10}  {'Radius (formula)':>20}")
    sep()
    for pid, label, z, r_val in pts:
        code, zc, rc_, rad, formula = row_extras(pid)
        volt = voltage_for(pid)
        rad_str = f"{rad:>10.5f} ({formula})" if formula else f"{rad:>10.5f}      "
        print(f"  {pid:<4}  {label:<30}  {z:>11.5f}  {r_val:>11.5f}  "
              f"  {code:>4}  {volt:>12.1f}  {zc:>11.5f}  {rc_:>10.5f}  {rad_str}")
    sep()

    print("\n  DERIVED ANGLES")
    sep()
    print(f"  {'thetaT — half-angle as built (corrected)':<48}  {meta['theta_deg']:>10.5f}  deg")
    print(f"  {'delta  = arcsin(sin(thetaT) + k1)':<48}  {meta['delta_deg']:>10.5f}  deg")
    print(f"  {'beta   — anode sphere angle':<48}  {meta['beta_deg']:>10.5f}  deg")
    tgt = meta.get('psi_target')
    tgt_s = f"   (Pierce value {tgt:g})" if tgt is not None else ""
    print(f"  {'psi    — focus electrode to beam edge':<48}  {meta['psi_deg']:>10.5f}  deg{tgt_s}")
    sep("═")
    print()

    # ── Plain text file: code, z, r, voltage, z_center, r_center, radius ──────
    if txt_path is None:
        txt_path = os.path.join(output_dir(), "pierce_gun_points.txt")

    def snap(v, h):
        """Round v to the nearest grid point (multiple of h)."""
        return round(round(v / h) * h, 10)

    def fmt(v, ndp=2):
        """Format a snapped value to ndp decimal places."""
        return f"{v:.{ndp}f}"

    # Derived snapped values for CATHODE section
    N        = sum(1 for row in pts if row[0] != 0)   # number of points excl pt 0
    d_snap   = snap(d_val,    H_z)
    Rc_snap  = snap(Rc_mm_val, H_r)
    rc_snap  = snap(rc_mm_val, H_r)
    # maxtr must be >= Rc / H_r; round up to nearest multiple of 50
    maxtr_threshold = Rc_mm_val / H_r
    maxtr = int(math.ceil(maxtr_threshold / 50.0)) * 50
    maxtr = max(maxtr, 50)   # never below 50
    # Emission step sizes, as the professor sets them: 3 x H_z and 2 x H_r
    DZ       = round(H_z * 3.0, 6)
    DX       = round(H_r * 2.0, 6)

    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            # ── Header ───────────────────────────────────────────────────────
            f.write(" 'PIERCE GUN TEST'\n")
            f.write(" &BOUND\n")
            f.write(f" H =   {H_z},{H_r}, MBP = {N}, NT = {N}, NGRD = 0, "
                    f"jMAG = 0, NMAG = 0, MAGORD = 2,\n")
            f.write(" /\n")
            f.write(" &CONTROL\n")
            f.write(" IOPT = 2, ICLOS = 0, IOS = 0, IORDR=1, IPONT = 5, "
                    "NMAX = 1000, EL =   0.0000050, \n")
            f.write(f" maxtr={maxtr}, lmax=20,LC=1,eqstep=0.1,1.,10.,\n")
            f.write(" /\n")
            f.write(" &PRINTS\n")
            f.write(" IPOT = 0, IFLDZ = 2, IFLDR = 2, ISPC = 1, IRAY = 2,\n")
            f.write(" /\n")
            f.write(" &PLOT\n")
            f.write(" ITRPL = 2, IPOTPL = 2, ICDPL = 2, ISCDPL = 2,\n")
            f.write(" / \n")

            # ── Point rows ───────────────────────────────────────────────────
            for pid, label, z, r_val in pts:
                if pid == 0:
                    continue
                code, zc, rc_, rad, formula = row_extras(pid)
                volt    = voltage_for(pid)
                # z: full precision for pt2, snapped for all others
                z_out   = f"{z:.8f}"              if pid == 2 else fmt(snap(z,    H_z))
                r_out   = fmt(snap(r_val, H_r))
                v_out   = fmt(snap(volt,  H_z),   ndp=4)
                zc_out  = fmt(snap(zc,    H_z),   ndp=6)
                rc_out  = fmt(snap(rc_,   H_r),   ndp=5)
                rad_out = fmt(snap(rad,   H_r),   ndp=4)
                f.write(f"{code:>2},  {z_out:<16}  {r_out:<8}  {v_out:<12}"
                        f"  {zc_out:<12}  {rc_out:<10}  {rad_out}\n")

            # ── Footer ───────────────────────────────────────────────────────
            f.write(" &GRID\n")
            f.write("  /\n")
            f.write(f" &CATHODE CSHAPE= \"SPHERE\", EMISSION=\"CHILD\", "
                    f"NSEG = 1, OCAT = {fmt(d_snap)},\n")
            f.write(f" CRAD = {fmt(Rc_snap)}, RMAX = {fmt(rc_snap)} ,"
                    f"DZ = {DZ}, DX =  {DX}\n")

        print(f"  ✓  Point table saved →  {txt_path}\n")
    except Exception as e:
        print(f"  WARNING: Could not save text file — {e}\n")



def write_synthesis_report(r, dp, meta, path=None):
    """Write every synthesis result to one plain-text file."""
    if path is None:
        path = os.path.join(output_dir(), "pierce_gun_synthesis_results.txt")
    MM = IN_TO_MM
    g  = lambda k: r[k] * MM
    L, A = [], None
    A = L.append
    RULE = "-" * 78
    A("PIERCE ELECTRON GUN \u2014 SYNTHESIS RESULTS")
    A("=" * 78)
    A("")
    A("INPUTS")
    A(RULE)
    for lab, val, unit in [("Beam voltage  V", r['V'] / 1000.0, "kV"),
                           ("Beam current  I", r['I'], "A"),
                           ("Beam waist radius  rw", round(r['rw'] * MM, 6), "mm"),
                           ("Cathode emission density  Jc", r['Jc'], "A/cm2")]:
        A(f"  {lab:<40}{val:>12}  {unit}")
    A("")
    A("SYNTHESIS RESULTS")
    A(RULE)
    for lab, val, unit in [("Microperveance  uP", r['muP'], ""),
                           ("Relativistic diode current factor  kf", r['rel_factor'], ""),
                           ("Cathode disc radius  rc", g('rc'), "mm"),
                           ("Convergence half angle  theta", r['theta'], "deg"),
                           ("Cathode spherical radius  Rc", g('Rc'), "mm"),
                           ("Anode sphere radius  Ra", g('Ra'), "mm"),
                           ("Cathode-anode spacing  zac", g('zac'), "mm"),
                           ("Anode plane position  za", g('za'), "mm"),
                           ("Beam radius at anode  rb(za)", g('rb_za'), "mm"),
                           ("Anode aperture radius  ra", g('ra'), "mm"),
                           ("Normalised compression  R", r['R'], ""),
                           ("Area compression  (rc/rw)^2", (r['rc'] / r['rw']) ** 2, ""),
                           ("Throw, cathode vertex to waist  zw", g('zw'), "mm")]:
        A(f"  {lab:<40}{val:>12.5f}  {unit}")
    A("")
    A("SPHERICAL-ABERRATION CORRECTION  (factor k = 0.905)")
    A(RULE)
    A(f"  {'Corrected half angle  thetaT':<40}{r['thetaT']:>12.5f}  deg")
    A(f"  {'Corrected cathode sphere radius  RcT':<40}{g('RcT'):>12.5f}  mm")
    A("")
    A("SOLVER DIAGNOSTICS")
    A(RULE)
    A(f"  {'Status':<40}{'CONVERGED (bracketed bisection)':>12}")
    A(f"  {'Roots found in the valid window':<40}{r['n_roots']:>12}")
    A(f"  {'Search window':<40}{'0 - %.2f deg' % r['theta_hi']:>12}")
    cc = r.get('theta_crosscheck')
    if cc is None:
        A(f"  {'Cross-check, Yang (2006) iteration':<40}{'did not converge':>12}")
    else:
        A(f"  {'Cross-check, Yang (2006) iteration':<40}{cc:>12.3f}  deg")
        A(f"  {'Agreement between the two solvers':<40}{abs(cc - r['theta']):>12.4f}  deg")
    A(f"  {'Anode aperture equation valid':<40}{('yes' if r['za_valid'] else 'NO'):>12}")
    A(f"  Warnings: {'; '.join(r['warnings']) if r['warnings'] else 'none'}")
    A("")
    A("ELECTRODE GEOMETRY SETTINGS USED")
    A(RULE)
    A(f"  {'Focus electrode angle to beam edge':<42}{PIERCE_ANGLE:.2f}  deg   (Pierce value, fixed)")
    A(f"  {'Cathode built with  thetaT, RcT':<42}{r['thetaT']:.3f} deg, {g('RcT'):.4f} mm  (Vaughan Sec. III-F)")
    A(f"  {'Anode side shifted downstream by':<42}{dp.get('Dz', 0.0):.4f} mm  (extra depth of corrected cathode)")
    A(f"  {'Cathode rim to electrode gap  k1':<42}{dp['k1']:.3f} x RcT = {dp['k1'] * g('RcT'):.4f} mm")
    A(f"  {'Anode sphere rim angle  beta':<42}{dp['beta_deg']:.2f}  deg   (set by pt 9 lying below pt 8)")
    A(f"  {'Focus electrode tip height':<42}{ELECTRODE_TOP_FACTOR:g} x H2  = {dp['r6']:.4f} mm  (H2 = height of pt 2)")
    A(f"  {'Gap between pts 7 and 8':<42}{dp['alpha']:.4f} mm  (defaults to the cathode-anode spacing)")
    A(f"  {'Electrode length':<42}{meta.get('elec_len', float('nan')):.3f}  mm")
    A(f"  {'Grid spacing  H_z, H_r':<42}{dp['H_z']}, {dp['H_r']} mm")
    A(f"  {'Cathode vertex offset  d':<42}{dp['d']:.2f} mm;  cathode body depth  l  {dp['l']:.2f} mm")
    A(f"  {'Anode aperture factor  ra / rb(za)':<42}{r.get('aperture_factor',1.2):.3f}"
      f"{'' if abs(r.get('aperture_factor',1.2)-1.2)<1e-9 else '   (Vaughan uses 1.2)'}")
    A(f"  {'Gap, pts 11 to 10 (pt 10 above pt 11)':<42}{dp['nH_gap']} x H_r  = {dp['nH_gap'] * dp['H_r']:.4f} mm")
    A(f"  {'Drift-tube taper, pts 11 to 12':<42}{dp['phi2_deg']:.3f}  deg   (beam exit angle phi2)")
    A(f"  {'Drift-tube radius, pts 12 and 13':<42}{dp['r12']:.4f} mm")
    A(f"  {'Smallest drift-tube / beam radius':<42}{dp['clearance']:.3f}")
    A(f"  {'Rmax = %g x H2' % RMAX_FACTOR:<42}{dp['Rmax']:.4f} mm")
    A(f"  {'Zmax = d + 1.3 x throw':<42}{dp['Zmax']:.4f} mm")
    A("")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(L) + "\n")
        print(f"  \u2713  Synthesis results saved \u2192  {path}\n")
    except Exception as e:
        print(f"  WARNING: Could not write the synthesis results file \u2014 {e}\n")
    return path


def plot_gun_wireframe(pts, meta, save_path=None):
    """
    Draw the Pierce gun half cross-section as a clean wireframe.
    All 15 points (0-14) are labelled; segments are straight lines or the cathode arc.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("  NOTE: matplotlib not installed — plot skipped.\n"); return

    pt       = {row[0]: (row[2], row[3]) for row in pts}
    Rc_mm    = meta['Rc_mm']
    arc_cz   = meta['arc_centre_z']
    theta    = math.radians(meta['theta_deg'])

    def arc(phi0, phi1, rho, n=300):
        phis = np.linspace(phi0, phi1, n)
        return arc_cz - rho*np.cos(phis), rho*np.sin(phis)

    z12, r12 = arc(0, theta, Rc_mm)   # cathode arc 1->2

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    line_kw = dict(color='black', lw=1.4, zorder=3)
    axis_kw = dict(color='#aaaaaa', lw=0.9, ls='--', zorder=1)

    # Symmetry axis
    ax.axhline(0, **axis_kw)

    # Straight segments — all edges drawn including gaps 3->4 and 7->8; profile closes at 14->1
    straight_segs = [
        (2,  3),
        (3,  4),
        (4,  5),
        (5,  6),
        (6,  7),
        (7,  8),
        (8,  9),
        (9,  10),
        (10, 11),
        (11, 12),
        (12, 13),
        (13, 14),
        (14,  1),
    ]
    for a_id, b_id in straight_segs:
        ax.plot([pt[a_id][0], pt[b_id][0]],
                [pt[a_id][1], pt[b_id][1]], **line_kw)

    # Arc segment: cathode arc 1->2 only
    ax.plot(z12,  r12,  **line_kw)

    # Point markers and labels
    nudge = {
        0:  (-1.5, -1.2),  1: (-1.2, -1.2),  2: ( 0.5,  1.0),
        3:  (-1.8,  0.8),  4: (-1.8,  0.8),  5: (-1.8, -1.4),
        6:  ( 0.5, -1.4),  7: ( 0.5,  1.0),  8: ( 1.2,  1.0),
        9:  ( 1.5,  0.5), 10: (-1.8,  1.0), 11: ( 0.5, -1.4),
        12: ( 1.2,  1.0), 13: ( 1.2,  1.0), 14: ( 1.2, -1.2),
    }
    all_z, all_r = zip(*[pt[i] for i in range(15)])
    z_span = max(all_z) - min(all_z)
    sc     = z_span / 100.0

    for row in pts:
        pid        = row[0]
        z_pt, r_pt = pt[pid]
        ax.plot(z_pt, r_pt, 'o', color='black', ms=4, zorder=5)
        dz, dr = nudge.get(pid, (1.0, 1.0))
        ax.text(z_pt + dz*sc, r_pt + dr*sc*0.5, str(pid),
                fontsize=8, fontweight='bold', ha='center', va='center', zorder=6,
                color='black',
                bbox=dict(boxstyle='round,pad=0.15',
                          fc='white', ec='black', lw=0.6, alpha=0.9))

    ax.set_title(
        'Pierce Gun — Half Cross-Section  '
        '(r–z plane; revolve 360° around z-axis)\n'
        f'θT = {meta["theta_deg"]:.3f}°   '
        f'δ = {meta["delta_deg"]:.3f}°   '
        f'ψ = {meta["psi_deg"]:.2f}°   '
        f'β = {meta["beta_deg"]:.1f}°',
        fontsize=10, pad=8)
    ax.set_xlabel('z  (mm)', fontsize=10)
    ax.set_ylabel('r  (mm)', fontsize=10)

    z_lo = min(min(all_z), 0.0) - z_span * 0.05
    z_hi = max(all_z) + z_span * 0.05
    r_hi = max(all_r) * 1.15
    r_lo = -max(all_r) * 0.15
    ax.set_xlim(z_lo, z_hi)
    ax.set_ylim(r_lo, r_hi)
    ax.grid(True, color='#e0e0e0', lw=0.4, ls='--', zorder=0)
    ax.tick_params(labelsize=9)

    if save_path is None:
        save_path = os.path.join(output_dir(), "pierce_gun_design.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓  Diagram saved →  {save_path}\n")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Jc sweep — table, CSV export, and plots
# ─────────────────────────────────────────────────────────────────────────────

CSV_COLS = [
    ("Jc (A/cm2)", 'Jc_in', 1),  ("uP",         'muP',   1),
    ("rc (cm)",    'rc',    CM),  ("theta (deg)", 'theta', 1),
    ("Rc (cm)",    'Rc',    CM),  ("Ra (cm)",     'Ra',    CM),
    ("zac (cm)",   'zac',   CM),  ("za (cm)",     'za',    CM),
    ("rb_za (cm)", 'rb_za', CM),  ("R",           'R',     1),
    ("ra (cm)",    'ra',    CM),  ("zw (cm)",     'zw',    CM),
    ("thetaT (deg)",'thetaT',1),  ("RcT (cm)",    'RcT',   CM),
]

def _synthesis_worker(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle the call."""
    V, I, rw_in, Jc = args
    return pierce_synthesis(V, I, rw_in, Jc)


def run_sweep(V, I, rw_cm, Jc_min, Jc_max, n_steps=10, csv_name=""):
    """Run synthesis across n_steps+1 Jc values, print table, save CSV, plot."""

    rw_in   = rw_cm / CM
    Jc_vals = [Jc_min + (Jc_max - Jc_min) * i / n_steps for i in range(n_steps + 1)]

    # ── Parallel synthesis — each Jc in its own process ──────────────────────
    results_map = {}
    with ProcessPoolExecutor() as pool:
        futures = {
            pool.submit(_synthesis_worker, (V, I, rw_in, Jc)): (idx, Jc)
            for idx, Jc in enumerate(Jc_vals)
        }
        for fut in as_completed(futures):
            idx, Jc = futures[fut]
            try:
                r = fut.result()
                r['Jc_in'] = Jc
                results_map[idx] = r
            except Exception as e:
                print(f"  WARNING: Jc={Jc:.4f} failed — {e}")

    # Restore original Jc ordering
    results = [results_map[i] for i in sorted(results_map)]

    if not results:
        print("  No valid results produced. Check your inputs.\n")
        return

    muP_val = results[0]['muP']

    colsA = [
        ("Jc",     "A/cm²", 8), ("θ",      "deg",   8), ("θT",     "deg",   8),
        ("rc",     "cm",    9), ("rb(za)", "cm",    9), ("ra",     "cm",    9),
        ("R",      "",      8),
    ]
    colsB = [
        ("Jc",  "A/cm²", 8), ("Rc",  "cm",    9), ("RcT", "cm",    9),
        ("Ra",  "cm",    9), ("zac", "cm",    9), ("za",  "cm",    9),
        ("zw",  "cm",    9),
    ]

    def table_header(cols):
        print("┌" + "┬".join("─"*w for _, _, w in cols) + "┐")
        print("|" + "".join(f" {n:^{w-1}}|" for n, _, w in cols))
        print("|" + "".join(f" {u:^{w-1}}|" for _, u, w in cols))
        print("├" + "┼".join("─"*w for _, _, w in cols) + "┤")

    def table_footer(cols):
        print("└" + "┴".join("─"*w for _, _, w in cols) + "┘")

    W_hdr = max(
        sum(w for _, _, w in colsA) + len(colsA) + 1,
        sum(w for _, _, w in colsB) + len(colsB) + 1,
    )

    rw_mm = rw_cm * 10
    print(f"\n{'═'*W_hdr}")
    print(f"  Jc SWEEP")
    print(f"  V = {V/1000:.4g} kV    I = {I:.4g} A    rw = {rw_mm:.5f} mm    μP = {muP_val:.4f}")
    print(f"  Jc range: {Jc_min:.4g} → {Jc_max:.4g} A/cm²   ({len(results)} rows)")
    print(f"{'═'*W_hdr}")

    print("\n  ── Beam Parameters ──")
    table_header(colsA)
    for r in results:
        print(
            f"|{r['Jc_in']:>8.4f}|{r['theta']:>8.3f}|{r['thetaT']:>8.3f}"
            f"|{r['rc']*CM:>9.5f}|{r['rb_za']*CM:>9.5f}"
            f"|{r['ra']*CM:>9.5f}|{r['R']:>8.4f}|"
        )
    table_footer(colsA)

    print("\n  ── Electrode Geometry ──")
    table_header(colsB)
    for r in results:
        print(
            f"|{r['Jc_in']:>8.4f}|{r['Rc']*CM:>9.5f}|{r['RcT']*CM:>9.5f}"
            f"|{r['Ra']*CM:>9.5f}|{r['zac']*CM:>9.5f}"
            f"|{r['za']*CM:>9.5f}|{r['zw']*CM:>9.5f}|"
        )
    table_footer(colsB)
    print()

    if csv_name:
        if not csv_name.lower().endswith(".csv"):
            csv_name += ".csv"
        # Write next to the executable, as the .txt and .png already do. A bare
        # filename lands in the current working directory, which for a
        # double-clicked .exe is not the folder the .exe is in.
        csv_name = os.path.join(output_dir(), os.path.basename(csv_name))
        try:
            with open(csv_name, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([h for h, _, _ in CSV_COLS])
                for r in results:
                    writer.writerow([f"{r[k]*s:.6f}" for _, k, s in CSV_COLS])
            print(f"  CSV saved to: {os.path.abspath(csv_name)}\n")
        except Exception as e:
            print(f"  WARNING: Could not save CSV — {e}\n")

    if not MATPLOTLIB_AVAILABLE:
        print("  NOTE: matplotlib not installed — plots skipped.\n"); return

    Jc_x  = [r['Jc_in'] for r in results]
    scale = {'rc': CM, 'Rc': CM, 'RcT': CM, 'Ra': CM, 'zac': CM, 'zw': CM}
    data  = {k: [r[k] * scale.get(k, 1) for r in results]
             for k in ['theta', 'thetaT', 'rc', 'Rc', 'RcT', 'Ra', 'zac', 'zw', 'R']}

    tag = f"V={V/1000:.3g} kV    I={I:.3g} A    rw={rw_mm:.4g} mm"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Pierce Gun Sweep Results\n{tag}", fontsize=12)

    ax = axes[0, 0]
    ax.plot(Jc_x, data['theta'],  'b-o', linewidth=1.5, label='θ (uncorrected)')
    ax.plot(Jc_x, data['thetaT'], 'r-s', linewidth=1.5, label='θ_T (corrected)')
    ax.set(xlabel='Jc (A/cm²)', ylabel='Convergence Half-Angle (deg)', title='Beam Angles vs Jc')
    ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(Jc_x, data['rc'],  'b-o',  linewidth=1.5, label='rc (cathode disc)')
    ax.plot(Jc_x, data['Rc'],  'r-s',  linewidth=1.5, label='Rc (cathode sphere)')
    ax.plot(Jc_x, data['RcT'], 'm--^', linewidth=1.5, label='RcT (corrected)')
    ax.plot(Jc_x, data['Ra'],  'g-d',  linewidth=1.5, label='Ra (anode sphere)')
    ax.set(xlabel='Jc (A/cm²)', ylabel='Radius (cm)', title='Electrode Radii vs Jc')
    ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(Jc_x, data['zac'], 'b-o', linewidth=1.5, label='zac (anode-cathode spacing)')
    ax.plot(Jc_x, data['zw'],  'r-s', linewidth=1.5, label='zw (throw)')
    ax.set(xlabel='Jc (A/cm²)', ylabel='Axial Distance (cm)', title='Axial Dimensions vs Jc')
    ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(Jc_x, data['R'], 'k-o', linewidth=1.5)
    ax.set(xlabel='Jc (A/cm²)', ylabel='R = rb(za) / rw', title='Normalised Compression vs Jc')
    ax.grid(True)

    plt.tight_layout()
    sweep_png = os.path.join(output_dir(), "pierce_gun_sweep.png")
    try:
        plt.savefig(sweep_png, dpi=150, bbox_inches="tight")
        print(f"  Sweep plot saved to: {sweep_png}")
    except Exception as e:
        print(f"  WARNING: Could not save sweep plot — {e}")
    plt.show()
    print("  4 plots generated.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════════════╗"
          "\n║         PIERCE GUN SYNTHESIS  (Vaughan 1981)        ║"
          "\n╚══════════════════════════════════════════════════════╝\n")

    while True:
        print("  Select mode:\n"
              "    [1]  Single design  — enter one value of Jc\n"
              "    [2]  Jc sweep table — enter a range of Jc values\n"
              "    [q]  Quit\n")
        mode = input("  Mode [1/2/q] : ").strip().lower()

        if mode in ("q", "quit"):
            print("\n  Goodbye.\n"); break

        if mode not in ("1", "2"):
            print("  Please enter 1, 2, or q.\n"); continue

        V     = ask("Beam voltage          V  [kV]    ", lo=0.001) * 1000
        I     = ask("Beam current          I  [Amps]  ", lo=1e-9)
        rw_mm = ask("Waist radius         rw  [mm]    ", lo=1e-6)
        print("\n  Anode aperture radius = factor x beam radius there. Vaughan's value is 1.2;")
        print("  a larger factor widens the pts 10-11 gap and moves the aperture plane only.")
        ap_f  = ask("Aperture factor          [-]      ", default=1.2, lo=1.0, hi=3.5)
        rw_cm = rw_mm / 10          # mm → cm for internal use
        rw_in = rw_cm / CM          # cm → inches for pierce_synthesis

        if mode == "1":
            Jc = ask("Cathode current dens Jc  [A/cm²] ", lo=1e-6)
            try:
                result = pierce_synthesis(V, I, rw_in, Jc, aperture_factor=ap_f)
                result['rw_mm'] = rw_mm
                print_results(result)

                # ── Geometry: coordinate table and diagram ────────────────────
                dp = get_design_params_mode1(result)
                pts, meta = compute_gun_points_mm(result, dp)
                print_gun_points(pts, meta, V=V, H_z=dp["H_z"], H_r=dp["H_r"],
                               Rc_mm_val=result["RcT"]*CM*10,
                               rc_mm_val=result["rc"]*CM*10,
                               d_val=dp["d"])
                write_synthesis_report(result, dp, meta)
                plot_gun_wireframe(pts, meta)

            except Exception as e:
                print(f"\n  ✖  Calculation failed: {e}\n"
                      "     Check that your inputs define a physically realisable gun.\n")
        else:
            print()
            Jc_min   = ask("Jc minimum  [A/cm²] ", lo=1e-6)
            Jc_max   = ask("Jc maximum  [A/cm²] ", lo=Jc_min + 1e-9)
            csv_name = input("  CSV filename (e.g. sweep1.csv, or Enter to skip) : ").strip()
            run_sweep(V, I, rw_cm, Jc_min, Jc_max, n_steps=10, csv_name=csv_name)

        if input("  Run another calculation? [y/n] : ").strip().lower() not in ("y", "yes"):
            print("\n  Goodbye.\n"); break
        print()


if __name__ == "__main__":
    # Required when frozen with PyInstaller: without it every worker process
    # re-runs this interactive script from the top instead of just the worker.
    import multiprocessing
    multiprocessing.freeze_support()
    main()
