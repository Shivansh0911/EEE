"""
Adapter around Vedant's exact Pierce-gun synthesis solver.

The solver lives, unmodified, in `src/physics/vendor/pierce_gun_design_fixed.py`
(Vaughan 1981 + Yang-Jia-Zhu 2006 amendments + Liao 2003 start + Greenwood 2016
relativistic diode factor). He is a dependency: this module imports it and
never edits it. Its interactive parts (`ask()` -> `input()`) are bypassed by
temporarily replacing the module's `ask` attribute at call time, which changes
nothing in his file.

Units at this boundary are the project's: V in volts, I in amps, rw in MILLIMETRES,
Jc in A/cm^2, lengths out in millimetres. His solver works in inches internally;
the conversion happens here and only here.
"""

import contextlib
import importlib
import io
import math
import os
from concurrent.futures import ProcessPoolExecutor

IN_TO_MM = 25.4
V_REF = 1000.0   # volts; see to_vaughan_inputs

LENGTH_KEYS = ("rw", "rc", "Rc", "Ra", "zac", "za", "rb_za", "ra", "zw", "RcT")

_MOD = None


def module():
    """The vendored solver, imported once."""
    global _MOD
    if _MOD is None:
        _MOD = importlib.import_module("src.physics.vendor.pierce_gun_design_fixed")
    return _MOD


def to_vaughan_inputs(P_uperv, rw_mm, C, V=V_REF):
    """
    (P, rw, C) -> his four inputs (V, I, rw, Jc).

    His solver sees V and I only through muP = I / V^1.5 and rc = sqrt(I/(pi Jc)),
    plus V through the relativistic factor kf. The literature rows carry no
    voltage, so a reference V is needed. V_REF = 1 kV is the classical limit
    (kf - 1 is about -1e-4 there), which keeps the labels a function of
    (P, rw, C) alone -- the only inputs the ANN sees.
    """
    I = P_uperv * 1e-6 * V ** 1.5
    rc_cm = math.sqrt(C) * rw_mm / 10.0
    Jc = I / (math.pi * rc_cm ** 2)
    return dict(V=float(V), I=float(I), rw_mm=float(rw_mm), Jc=float(Jc))


def synthesize(V, I, rw_mm, Jc, aperture_factor=1.2):
    """
    One exact synthesis. Returns a dict with lengths in mm and `feasible`.

    `GunInfeasible` (his explicit refusal) becomes feasible=False with his
    message as `reason`; any other exception is NOT swallowed, because it would
    mean the adapter is misusing his code.
    """
    m = module()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r = m.pierce_synthesis(V, I, rw_mm / IN_TO_MM, Jc,
                                   aperture_factor=aperture_factor)
    except m.GunInfeasible as e:
        return dict(V=V, I=I, rw_mm=rw_mm, Jc=Jc, feasible=False, reason=str(e))
    out = dict(r)
    out["raw_inches"] = {k: r[k] for k in LENGTH_KEYS}
    for k in LENGTH_KEYS:
        out[k + "_mm"] = r[k] * IN_TO_MM
    out["rw_mm"] = rw_mm
    out["C"] = (r["rc"] / r["rw"]) ** 2
    out["theta_raw"] = r["theta"]
    out["feasible"] = True
    out["reason"] = ""
    return out


def _worker(args):
    return synthesize(*args)


def synthesize_many(inputs, workers=None):
    """Parallel exact synthesis over [(V, I, rw_mm, Jc[, aperture_factor]), ...].

    Order is preserved. Each design is ~1-2 s of mpmath, so this is what makes a
    100-design verification take seconds per core rather than minutes.
    """
    inputs = [tuple(x) for x in inputs]
    if workers == 1 or len(inputs) < 2:
        return [synthesize(*x) for x in inputs]
    workers = workers or min(len(inputs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, inputs, chunksize=1))


@contextlib.contextmanager
def _patched_ask(overrides):
    """Make his interactive `ask()` return its own default, or our override.

    Overrides are keyed by the first token of his prompt text (e.g. 'k1', 'd',
    'H_z', 'g', 'nH_gap'). Restored on exit even if his code raises.
    """
    m = module()
    original = m.ask

    def ask(prompt, default=None, lo=None, hi=None):
        key = prompt.split()[0]
        v = overrides.get(key, default)
        if v is None:
            raise ValueError(f"no default for {key!r} and no override given")
        v = float(v)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise ValueError(f"{key} = {v} is outside his allowed range [{lo}, {hi}]")
        return v

    m.ask = ask
    try:
        yield
    finally:
        m.ask = original


class DeckRefused(ValueError):
    """The design is not one his EGUN deck construction can honestly represent."""


def write_deck(result, path, overrides=None, report_path=None):
    """
    Write his EGUN point deck for a synthesised design, using his own code:
    get_design_params_mode1 -> compute_gun_points_mm -> print_gun_points.

    Refuses (DeckRefused) when the design is infeasible or when theta_raw is above
    56.44 deg, where Vaughan's Eq.(20) has no real root, the anode plane za is a
    placeholder, and pt 11 of the deck would be placed at a made-up position.
    Returns (points, meta, design_params).
    """
    m = module()
    if not result.get("feasible", False):
        raise DeckRefused(f"design is infeasible: {result.get('reason', '')}")
    if not result.get("za_valid", False):
        raise DeckRefused(
            f"theta_raw = {result['theta']:.3f} deg > 56.44 deg: Vaughan Eq.(20) has "
            "no real root for the anode plane za, so the deck's aperture point would "
            "be fictitious. Sharma (2001) would fix this and is not implemented.")
    r = {k: result[k] for k in result if k not in ("raw_inches",)}
    for k, v in result["raw_inches"].items():
        r[k] = v                                   # his geometry wants inches
    r["rw_mm"] = result["rw_mm"]
    buf = io.StringIO()
    with _patched_ask(overrides or {}), contextlib.redirect_stdout(buf):
        dp = m.get_design_params_mode1(r)
        pts, meta = m.compute_gun_points_mm(r, dp)
        m.print_gun_points(pts, meta, V=r["V"], H_z=dp["H_z"], H_r=dp["H_r"],
                           Rc_mm_val=r["RcT"] * IN_TO_MM, rc_mm_val=r["rc"] * IN_TO_MM,
                           d_val=dp["d"], txt_path=path)
        if report_path:
            m.write_synthesis_report(r, dp, meta, path=report_path)
    if not os.path.exists(path):
        raise RuntimeError(f"his deck writer did not produce {path}:\n{buf.getvalue()}")
    return pts, meta, dp


def relative_error(a, b):
    return abs(a - b) / abs(b) if b else float("inf")


def verify(designs, tol=0.02, keys=("thetaT", "theta", "Rc_mm", "zac_mm", "zw_mm"),
           workers=None):
    """
    Re-run designs through the exact solver and compare with predicted values.

    `designs` is a list of dicts with V, I, rw_mm, Jc and predicted values under
    the same keys as `keys`. Returns a list of per-design dicts with the exact
    values, each relative error, `max_err`, `feasible`, and `ok`.
    """
    exact = synthesize_many([(d["V"], d["I"], d["rw_mm"], d["Jc"]) for d in designs],
                            workers=workers)
    out = []
    for d, ex in zip(designs, exact):
        row = {"V": d["V"], "I": d["I"], "rw_mm": d["rw_mm"], "Jc": d["Jc"],
               "feasible": ex["feasible"], "reason": ex.get("reason", "")}
        errs = []
        if ex["feasible"]:
            for k in keys:
                row[k + "_pred"] = d[k]
                row[k + "_exact"] = ex[k]
                e = relative_error(d[k], ex[k])
                row[k + "_relerr"] = e
                errs.append(e)
        row["max_err"] = max(errs) if errs else float("nan")
        row["ok"] = bool(ex["feasible"] and max(errs) <= tol)
        out.append(row)
    return out
