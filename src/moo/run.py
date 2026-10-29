"""
Run one NSGA-II study and write its outputs.

    python -m src.moo.run shin_eik
    python -m src.moo.run ceeri_klystron
    python -m src.moo.run path/to/my_study.json

Writes reports/moo/<name>/:
  pareto.csv          every non-dominated design: variables, objectives, full geometry
  vaughan_inputs.csv  the same designs as Vedant's four inputs (V, I, rw, Jc)
  knee.json           the knee design, its +/-5 % sensitivity, and the theta to
                      expect if the gun is built and MEASURED (calibrated, with band)
  pareto.png          pairwise objective plots, knee and reference marked
  run.json            settings, evaluation count, wall time
verify.csv is written by `python -m src.moo.verify` (Gate G3).
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

from ..ann import dataset as ds
from ..physics import closure
from .problem import GEOM, VARS, GunProblem, evaluate, knee_index, load_study

def out_dir(name):
    d = os.path.join(ds.repo_root(), "reports", "moo", name)
    os.makedirs(d, exist_ok=True)
    return d


def calibration():
    """Solver -> measured calibration for designs (reports/synthetic_roles.json).

    The scale-free scalar k, NOT the LOO-selected ridge: the ridge's advantage
    comes entirely from ln rw, which cannot physically change theta (rw is a
    length scale; theta depends on P and C), and it extrapolates by -22 % / +33 %
    at the two studies' knees, whose rw lie outside the measured guns' range.
    See D11 and reports/INTEGRATION_SUMMARY.md.
    """
    path = os.path.join(ds.repo_root(), "reports", "synthetic_roles.json")
    with open(path, encoding="utf-8") as f:
        s = json.load(f)
    return {"k": s["closure_calibrator_scalar"]["k"], "band_pct": s["band_scalar_pct_p90_loo"],
            "envelope": s["calibration_envelope"]}


def expected_measured(thetaT, P, rw, C, cal):
    t = cal["k"] * float(thetaT)
    env = cal["envelope"]
    outside = [n for n, v in (("perveance_uperv", P), ("C", C))
               if not env[n][0] <= v <= env[n][1]]
    return {"thetaT_solver_deg": float(thetaT), "thetaT_expected_measured_deg": t,
            "band_pct": cal["band_pct"],
            "range_deg": [t * (1 - cal["band_pct"] / 100), t * (1 + cal["band_pct"] / 100)],
            "extrapolated": bool(outside), "outside_measured_range": outside,
            "calibration": f"theta_meas ~ k * theta_T(solver), k = {cal['k']:.4f}, fitted on "
                           "the 23 real training guns; band = 90th percentile of its "
                           "leave-one-out |relative error| on those guns"}


def sensitivity(x, study, pct=5.0):
    """+/-pct % on each variable around x; relative change of key outputs."""
    base = evaluate(np.array([x]), study)[2]
    out = {}
    for j, v in enumerate(VARS):
        for sgn in (-1, 1):
            xp = np.array(x, float)
            xp[j] *= 1 + sgn * pct / 100
            r = evaluate(np.array([xp]), study)[2]
            key = f"{v}{'+' if sgn > 0 else '-'}{pct:g}%"
            out[key] = {"feasible": bool(r["feasible"][0])}
            if r["feasible"][0]:
                for k in ("thetaT", "Rc_mm", "zac_mm", "zw_mm", "E_kV_per_mm"):
                    out[key][k + "_rel_change_pct"] = float(
                        (r[k][0] - base[k][0]) / base[k][0] * 100)
    return out


def plot(front, F, names, knee, ref_F, title, path):
    k = len(names)
    pairs = [(i, j) for i in range(k) for j in range(i + 1, k)] or [(0, 0)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 4.2), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        ax.scatter(F[:, i], F[:, j], s=14, c="#4C78A8", label="Pareto (emulator)")
        ax.scatter(F[knee, i], F[knee, j], s=90, marker="*", c="#E45756", label="knee", zorder=5)
        if ref_F is not None:
            ax.scatter(ref_F[i], ref_F[j], s=60, marker="s", facecolors="none",
                       edgecolors="black", label="published design", zorder=5)
        ax.set_xlabel(names[i]); ax.set_ylabel(names[j]); ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run(name_or_path, n_gen=None, pop_size=None, seed=None, verbose=True):
    study = load_study(name_or_path)
    n_gen = n_gen or study["n_gen"]
    pop_size = pop_size or study["pop_size"]
    seed = study["seed"] if seed is None else seed
    t0 = time.time()
    res = minimize(GunProblem(study), NSGA2(pop_size=pop_size), ("n_gen", n_gen),
                   seed=seed, verbose=False)
    wall = time.time() - t0
    if res.X is None:
        raise RuntimeError("NSGA-II found no feasible design; loosen the constraints")
    X = np.atleast_2d(res.X)
    F, G, r = evaluate(X, study)
    assert (G <= 1e-9).all(), "a reported Pareto design violates a constraint"
    order = np.argsort(F[:, 0])
    X, F = X[order], F[order]
    F, G, r = evaluate(X, study)

    d = out_dir(study["name"])
    df = pd.DataFrame(X, columns=VARS)
    df.insert(2, "I", study["I"])
    for i, nme in enumerate(study["objectives"]):
        df["obj_" + nme] = F[:, i]
    for k in GEOM:
        df[k] = r[k]
    df.to_csv(os.path.join(d, "pareto.csv"), index=False, float_format="%.8g")
    df[["V", "I", "rw_mm", "Jc"]].to_csv(os.path.join(d, "vaughan_inputs.csv"),
                                          index=False, float_format="%.8g")

    ki = knee_index(F)
    cal = calibration()
    ref = study.get("reference_design")
    ref_F = None
    ref_info = None
    if ref:
        xr = np.array([[ref["V"], ref["rw_mm"], ref["Jc"]]])
        Fr, Gr, rr = evaluate(xr, study)
        ref_F = Fr[0]
        ref_info = {"design": ref, "objectives": dict(zip(study["objectives"], map(float, Fr[0]))),
                    "satisfies_constraints": bool((Gr <= 1e-9).all()),
                    "thetaT": float(rr["thetaT"][0]), "throw_mm": float(rr["zw_mm"][0])}
    knee = {
        "rule": "min distance to the ideal point, objectives normalised to [0,1] over the front",
        "index_in_pareto_csv": ki,
        "design": {"V": float(X[ki, 0]), "I": study["I"], "rw_mm": float(X[ki, 1]),
                   "Jc": float(X[ki, 2])},
        "objectives": dict(zip(study["objectives"], map(float, F[ki]))),
        "geometry_mm_deg": {k: (float(r[k][ki]) if k != "za_valid" else bool(r[k][ki]))
                            for k in GEOM},
        "sensitivity_pm5pct": sensitivity(X[ki], study),
        "expected_if_measured": expected_measured(r["thetaT"][ki], r["muP"][ki], X[ki, 1],
                                                  r["C"][ki], cal),
        "reference_design": ref_info,
    }
    with open(os.path.join(d, "knee.json"), "w", encoding="utf-8") as f:
        json.dump(knee, f, indent=2)
    plot(X, F, study["objectives"], ki, ref_F, study["title"], os.path.join(d, "pareto.png"))
    meta = {"study": study, "evaluations": int(res.algorithm.evaluator.n_eval),
            "pareto_size": int(len(X)), "wall_seconds": wall, "backend": "lb-closure",
            "seed": seed}
    with open(os.path.join(d, "run.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if verbose:
        print(f"{study['name']}: {meta['evaluations']} evaluations, {len(X)} Pareto designs, "
              f"{wall:.1f} s; knee {knee['design']}")
    return meta, knee


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    for name in argv or ["shin_eik", "ceeri_klystron"]:
        run(name)


if __name__ == "__main__":
    main()
