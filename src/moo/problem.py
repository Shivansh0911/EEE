"""
The Pierce-gun design problem for NSGA-II.

Decision variables are Vedant's own four inputs with the current fixed by the
study: x = (V, r_w, Jc) at I = const. Each candidate is synthesised by a
backend; the default, `lb-closure`, is the G1-verified emulator of his exact
solver. (His solver itself takes 3-5 s per design: 50 000 evaluations would be
two to three days. The emulator does a population of 100 in ~0.1 s.)

Objectives (all minimised; pick any subset in the study config):
  Jc         cathode loading, A/cm^2      (cathode life)
  thetaT     built half-angle, deg        (lower = gentler optics)
  V          beam voltage, V              (supply, X-ray, breakdown)
  zac        cathode-anode spacing, mm    (compactness)
  throw      cathode vertex to waist, mm
  rw_dev     |r_w - target| / target
  theta_dev  |theta_T - target| / target

Constraints (g <= 0 is satisfied):
  solver-feasible; C and P inside the study's ranges; theta_T <= thetaT_max;
  mean gap field V / z_ac <= E_max; the Eq.(5) Newton inversion converged
  (-alpha < 7.9); optionally the anode aperture defined (theta_raw <= 56.44,
  Eq.(20)), a throw target within a tolerance, and the ANN-E training envelope.
"""

import json
import os

import numpy as np
from pymoo.core.problem import Problem

from ..physics import closure

VARS = ("V", "rw_mm", "Jc")
OBJECTIVES = ("Jc", "thetaT", "V", "zac", "throw", "rw_dev", "theta_dev")
BIG = 1e9
GEOM = ("muP", "C", "theta", "thetaT", "rc_mm", "Rc_mm", "RcT_mm", "Ra_mm", "zac_mm",
        "za_mm", "ra_mm", "rb_za_mm", "zw_mm", "R", "rel_factor", "E_kV_per_mm",
        "za_valid")


def load_study(name_or_path):
    here = os.path.dirname(os.path.abspath(__file__))
    path = name_or_path if name_or_path.endswith(".json") else \
        os.path.join(here, "studies", name_or_path + ".json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def anne_envelope():
    from ..ann import dataset as ds
    with open(os.path.join(ds.repo_root(), "models", "model_anne.json"), encoding="utf-8") as f:
        env = json.load(f)["training_envelope"]
    return env


def evaluate(X, study, backend="lb-closure"):
    """Synthesise designs X = (n, 3) [V, rw_mm, Jc]; return (F, G, results)."""
    if backend != "lb-closure":
        raise ValueError(f"unknown backend {backend!r}; only 'lb-closure' is fast enough "
                         "for 50 000 evaluations (the exact solver verifies, it does not search)")
    X = np.atleast_2d(X)
    V, rw, Jc = X[:, 0], X[:, 1], X[:, 2]
    r = closure.synthesize(V, study["I"], rw, Jc)
    r["throw_mm"] = r["zw_mm"]
    r["E_kV_per_mm"] = V / 1000.0 / r["zac_mm"]
    feas = r["feasible"]

    tgt = study.get("targets", {}) or {}
    cols = {
        "Jc": Jc, "thetaT": r["thetaT"], "V": V, "zac": r["zac_mm"], "throw": r["zw_mm"],
        "rw_dev": np.abs(rw - tgt["rw_mm"]) / tgt["rw_mm"] if tgt.get("rw_mm") else None,
        "theta_dev": (np.abs(r["thetaT"] - tgt["thetaT"]) / tgt["thetaT"]
                      if tgt.get("thetaT") else None),
    }
    F = []
    for name in study["objectives"]:
        if name not in OBJECTIVES:
            raise ValueError(f"unknown objective {name!r}; choose from {OBJECTIVES}")
        if cols[name] is None:
            raise ValueError(f"objective {name!r} needs a target in the study config")
        F.append(np.where(feas, cols[name], BIG))
    F = np.column_stack(F)

    c = study["constraints"]
    nanbig = lambda v: np.where(feas, v, BIG)
    P = r["muP"]
    G = [
        np.where(feas, 0.0, 1.0),                              # solver-feasible
        c["C"][0] - r["C"], r["C"] - c["C"][1],
        c["P"][0] - P, P - c["P"][1],
        nanbig(r["thetaT"] - c["thetaT_max"]),
        nanbig(r["E_kV_per_mm"] - c["E_max_kV_per_mm"]),
        np.where(r["alpha_ok"] | ~feas, 0.0, 1.0),
    ]
    if c.get("aperture_valid"):
        G.append(np.where(r["za_valid"] | ~feas, 0.0, 1.0))
    if c.get("deck_constructible"):
        from ..physics.vaughan_adapter import deck_constructible
        G.append(np.where(deck_constructible(r) | ~feas, 0.0, 1.0))
    if c.get("throw_mm"):
        G.append(nanbig(np.abs(r["zw_mm"] - c["throw_mm"]) / c["throw_mm"] - c["throw_tol"]))
    if c.get("ann_envelope"):
        env = anne_envelope()
        for name, val in (("perveance_uperv", P), ("rw_mm", rw), ("C", r["C"])):
            lo, hi = env[name]
            G += [lo - val, val - hi]
    return F, np.column_stack(G), r


class GunProblem(Problem):
    def __init__(self, study, backend="lb-closure"):
        self.study, self.backend = study, backend
        b = study["bounds"]
        n_constr = len(evaluate(np.array([[b[v][0] for v in VARS]]), study, backend)[1][0])
        super().__init__(n_var=3, n_obj=len(study["objectives"]), n_ieq_constr=n_constr,
                         xl=np.array([b[v][0] for v in VARS], float),
                         xu=np.array([b[v][1] for v in VARS], float))

    def _evaluate(self, X, out, *args, **kwargs):
        F, G, _ = evaluate(X, self.study, self.backend)
        out["F"], out["G"] = F, G


def knee_index(F):
    """Min Euclidean distance to the ideal point after normalising each objective
    to [0, 1] over the front. The rule is stated, not tuned."""
    lo, hi = F.min(0), F.max(0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return int(np.argmin(np.linalg.norm((F - lo) / span, axis=1)))
