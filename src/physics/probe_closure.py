"""
EXPERIMENT (not a deliverable): does a first-principles two-equation model
reproduce Panahi et al. Table 2's theta_iterative column?

Model
-----
Unknowns: theta (half cone angle), g = ln(Rc/Ra).

(I) Spherical space-charge-limited flow (Langmuir-Blodgett):
        P_uperv = 29.33 * (1 - cos theta) / alpha(g)^2
    Constant 29.33 = (16*pi/9)*eps0*sqrt(2e/m) * 1e6.
    alpha solves  3*a*a'' + a'^2 + 3*a*a' = 1,  a(0)=0, a'(0)=1
    integrated in u = ln(r/rc) < 0 for converging flow (derived here from
    Poisson's equation; its series a = u - 0.3u^2 + ... matches the classical
    Langmuir-Blodgett coefficients, which is the check that the ODE is right).

(II) Drift to the waist. For r'' = K/r with r'=0 at the waist:
        r'^2 = 2K ln(r/rw),   K = 1.5156e-2 * P_uperv
     At the anode, r_a = sqrt(C)*rw*exp(-g), so ln(r_a/rw) = 0.5*ln(C) - g.
     Slope leaving the anode = sin(theta) * (1 - c * alpha'/alpha), where the
     bracket is the aperture (Davisson-Calbick) lens weakening the convergence.
     c is the coefficient under test.

Sweeping c tells us whether the closure is right and, if so, what c is.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import csv, os, sys

P_CONST = 29.33          # uperv, spherical diode
K_CONST = 1.5156e-2      # generalised perveance per uperv


# --- Langmuir-Blodgett alpha, by integrating the ODE -------------------------
def _rhs(u, y):
    a, ap = y
    # 3 a a'' + a'^2 + 3 a a' = 1
    if abs(a) < 1e-12:
        app = -0.6                      # limit value at the cathode
    else:
        app = (1.0 - ap * ap - 3.0 * a * ap) / (3.0 * a)
    return [ap, app]


_UMAX = 4.0
_sol = solve_ivp(_rhs, [0.0, -_UMAX], [0.0, 1.0],
                 dense_output=True, rtol=1e-10, atol=1e-12, max_step=0.01)


def alpha_and_deriv(g):
    """|alpha| and d|alpha|/dg at gamma = ln(Rc/Ra) = -u, g >= 0."""
    if g <= 0:
        return 0.0, 1.0
    if g > _UMAX:
        return np.nan, np.nan
    a, ap = _sol.sol(-g)
    return abs(a), abs(ap)


def series_check():
    """The ODE must reproduce the classical converging series near 0."""
    for g in (0.05, 0.1, 0.2):
        a, _ = alpha_and_deriv(g)
        ser = g + 0.3 * g**2 + 0.075 * g**3 + 0.0143182 * g**4
        print(f"   g={g:<5} ODE={a:.8f}  series={ser:.8f}  diff={a-ser:+.2e}")


# --- solve the two equations for a given (P, rw, C) and lens coefficient c ---
def solve_case(P, rw, C, c):
    lnC2 = 0.5 * np.log(C)
    K = K_CONST * P

    def residual(g):
        a, ap = alpha_and_deriv(g)
        if not np.isfinite(a) or a <= 0:
            return np.nan
        # (I) -> theta
        val = P * a * a / P_CONST
        if val >= 2.0:
            return np.nan
        th = np.arccos(1.0 - val)
        # (II) mismatch
        arg = lnC2 - g
        if arg <= 0:
            return np.nan
        lhs = np.sin(th) * (1.0 - c * ap / a)
        rhs = np.sqrt(2.0 * K * arg)
        return lhs - rhs

    lo, hi = 1e-4, min(lnC2 - 1e-6, _UMAX - 1e-6)
    if hi <= lo:
        return np.nan
    gs = np.linspace(lo, hi, 400)
    vals = [residual(g) for g in gs]
    root = None
    for i in range(len(gs) - 1):
        v0, v1 = vals[i], vals[i + 1]
        if np.isfinite(v0) and np.isfinite(v1) and v0 * v1 < 0:
            root = brentq(residual, gs[i], gs[i + 1], xtol=1e-12)
            break
    if root is None:
        return np.nan
    a, _ = alpha_and_deriv(root)
    return np.degrees(np.arccos(1.0 - P * a * a / P_CONST))


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    rows = list(csv.DictReader(
        open(os.path.join(root, "data", "processed", "dataset_literature.csv"))))

    print("ODE vs classical Langmuir-Blodgett series:")
    series_check()

    print("\nSweeping the aperture-lens coefficient c against theta_iterative:")
    print(f"  {'c':>8} {'solved':>7} {'MAE(deg)':>9} {'MRE(%)':>8}")
    best = None
    for c in np.arange(0.0, 1.21, 0.05):
        errs, rels = [], []
        for r in rows:
            th = solve_case(float(r["perveance_uperv"]), float(r["rw_mm"]),
                            float(r["C"]), c)
            if np.isfinite(th):
                ref = float(r["theta_iterative_deg"])
                errs.append(abs(th - ref))
                rels.append(abs(th - ref) / ref * 100)
        if len(errs) >= 20:
            mae, mre = np.mean(errs), np.mean(rels)
            print(f"  {c:8.2f} {len(errs):7d} {mae:9.3f} {mre:8.2f}")
            if best is None or mae < best[1]:
                best = (c, mae, mre, len(errs))
    if best:
        print(f"\nBest: c={best[0]:.2f}  MAE={best[1]:.3f} deg  "
              f"MRE={best[2]:.2f}%  on {best[3]}/30 cases")
        print("Acceptance gate: MAE <= 0.5 deg or MRE <= 1.5%  ->",
              "PASS" if (best[1] <= 0.5 or best[2] <= 1.5) else "FAIL")


if __name__ == "__main__":
    main()
