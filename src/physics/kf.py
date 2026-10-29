"""
Relativistic diode factor kf(V, Rc/Ra), tabulated from Vedant's solver.

kf is the space-charge-limited current of the converging spherical diode with
relativistic charge density, divided by the classical Langmuir-Blodgett value
at the same V and Rc/Ra (Greenwood, Zhang, Lau et al., Phys. Plasmas 23,
072101 (2016)). Vedant's `relativistic_factor` computes it by RK4 + secant in
about 15 ms per call, which is too slow inside a 50 000-evaluation search, so
it is tabulated here once and interpolated.

Grid: uniform in u = ln V and in y = ln(Rc/Ra). Interpolation: 4-point
(cubic) Lagrange in each axis on the uniform grid -- chosen over a spline
because it is a closed formula that the browser can reproduce bit-for-bit.

    python -m src.physics.kf          # rebuild models/kf_table.json
"""

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

V_MIN, V_MAX = 500.0, 600e3       # volts
Y_MIN, Y_MAX = 0.0, 4.0           # ln(Rc/Ra); Eq.(5) limit -alpha=9.6 is y ~ 2.9
NU, NY = 49, 41

_TABLE = None


def table_path():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "..", "models", "kf_table.json")


def _cell(args):
    from . import vaughan_adapter as va
    V, y = args
    return va.module().relativistic_factor(V, math.exp(y))


def build_table(workers=None):
    us = np.linspace(math.log(V_MIN), math.log(V_MAX), NU)
    ys = np.linspace(Y_MIN, Y_MAX, NY)
    jobs = [(math.exp(u), y) for u in us for y in ys]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        vals = list(pool.map(_cell, jobs, chunksize=16))
    return {
        "source": "relativistic_factor() of src/physics/vendor/pierce_gun_design_fixed.py "
                  "(Greenwood et al. 2016), evaluated on this grid",
        "u_min": math.log(V_MIN), "u_max": math.log(V_MAX), "nu": NU,
        "y_min": Y_MIN, "y_max": Y_MAX, "ny": NY,
        "kf": np.asarray(vals).reshape(NU, NY).tolist(),
    }


def load_table():
    global _TABLE
    if _TABLE is None:
        with open(table_path(), encoding="utf-8") as f:
            t = json.load(f)
        t["kf"] = np.asarray(t["kf"], dtype=float)
        _TABLE = t
    return _TABLE


def _lagrange4(t):
    """Weights of the 4-point Lagrange stencil at nodes -1, 0, 1, 2 for t in [0, 1]."""
    return (-t * (t - 1) * (t - 2) / 6.0,
            (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0,
            (t + 1) * t * (t - 1) / 6.0)


def _stencil(x, x0, x1, n):
    """Index of stencil node 0 and the fractional offset, clamped to the grid."""
    h = (x1 - x0) / (n - 1)
    s = (x - x0) / h
    i = np.clip(np.floor(s).astype(int), 1, n - 3)
    return i, s - i


def kf(V, y, table=None):
    """kf at voltage V (volts) and y = ln(Rc/Ra). Vectorised.

    y below 0 is clamped to 0, exactly as the vendored code clamps Rc/Ra to 1.
    V outside [V_MIN, V_MAX] raises: extrapolating a relativistic correction is
    not something to do silently.
    """
    t = table or load_table()
    V = np.asarray(V, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), t["y_min"], t["y_max"])
    if np.any(V < V_MIN * (1 - 1e-12)) or np.any(V > V_MAX * (1 + 1e-12)):
        raise ValueError(f"kf table covers V in [{V_MIN:g}, {V_MAX:g}] V")
    V, y = np.broadcast_arrays(V, y)
    iu, fu = _stencil(np.log(V), t["u_min"], t["u_max"], t["nu"])
    iy, fy = _stencil(y, t["y_min"], t["y_max"], t["ny"])
    wu, wy = _lagrange4(fu), _lagrange4(fy)
    K = t["kf"]
    out = np.zeros(V.shape)
    for a in range(4):
        for b in range(4):
            out = out + wu[a] * wy[b] * K[iu - 1 + a, iy - 1 + b]
    return out


def main():
    t = build_table()
    with open(table_path(), "w", encoding="utf-8") as f:
        json.dump(t, f)
    print(f"wrote {os.path.abspath(table_path())}  ({NU} x {NY})")


if __name__ == "__main__":
    main()
