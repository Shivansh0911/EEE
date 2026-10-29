"""
Gate G1: does the LBClosure emulator reproduce Vedant's exact solver?

    python -m src.physics.gate_g1 [--n 100] [--seed 7]

Draws n random designs over a deliberately wide box (wider than any study),
runs every one through BOTH the emulator and his exact mpmath solver, and
writes reports/gates/G1.json. Pass criteria (from the brief):

    theta (raw and T): mean relative error <= 0.05 %
    lengths (Rc, Ra, z_ac, z_a, throw, RcT): mean relative error <= 0.1 %
    feasibility: the two agree on every design

Max errors are reported alongside the means so a rare outlier is visible.
Exit status is non-zero on failure.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

from . import closure, vaughan_adapter as va

ANGLES = ("theta", "thetaT")
LENGTHS = ("Rc_mm", "Ra_mm", "zac_mm", "za_mm", "zw_mm", "RcT_mm")
THETA_TOL = 0.05e-2
LENGTH_TOL = 0.1e-2


def sample(n, seed):
    """(V, I, rw_mm, Jc) via (P, rw, C, V): P 0.2-3.7 uP, rw 0.1-10 mm,
    C 5-300 log-uniform, V 1-300 kV log-uniform."""
    rng = np.random.default_rng(seed)
    P = rng.uniform(0.2, 3.7, n)
    rw = np.exp(rng.uniform(math.log(0.1), math.log(10.0), n))
    C = np.exp(rng.uniform(math.log(5.0), math.log(300.0), n))
    V = np.exp(rng.uniform(math.log(1e3), math.log(300e3), n))
    I = P * 1e-6 * V ** 1.5
    Jc = I / (math.pi * (np.sqrt(C) * rw / 10.0) ** 2)
    return V, I, rw, Jc


def run(n=100, seed=7, workers=None):
    V, I, rw, Jc = sample(n, seed)
    t0 = time.time()
    em = closure.synthesize(V, I, rw, Jc)
    t_em = time.time() - t0
    t0 = time.time()
    ex = va.synthesize_many(list(zip(V, I, rw, Jc)), workers=workers)
    t_ex = time.time() - t0

    feas_ex = np.array([e["feasible"] for e in ex])
    agree = feas_ex == em["feasible"]
    both = feas_ex & em["feasible"]
    errs = {}
    for k in ANGLES + LENGTHS:
        e = np.array([abs(em[k][i] - ex[i][k]) / abs(ex[i][k]) for i in np.nonzero(both)[0]])
        errs[k] = {"mean": float(e.mean()), "max": float(e.max())}
    theta_mean = max(errs[k]["mean"] for k in ANGLES)
    length_mean = max(errs[k]["mean"] for k in LENGTHS)
    passed = bool(both.sum() >= 80 and agree.all()
                  and theta_mean <= THETA_TOL and length_mean <= LENGTH_TOL)
    return {
        "gate": "G1", "passed": passed,
        "n_drawn": int(n), "seed": int(seed),
        "n_feasible_both": int(both.sum()),
        "n_feasibility_disagree": int((~agree).sum()),
        "n_alpha_beyond_newton_limit": int((both & ~em["alpha_ok"]).sum()),
        "criteria": {"theta_mean_rel_max": THETA_TOL, "length_mean_rel_max": LENGTH_TOL,
                     "min_designs": 80},
        "worst_theta_mean_rel": theta_mean, "worst_length_mean_rel": length_mean,
        "errors": errs,
        "seconds_emulator": t_em, "seconds_exact_parallel": t_ex,
        "box": "P 0.2-3.7 uP, rw 0.1-10 mm, C 5-300 (log), V 1-300 kV (log)",
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args(argv)
    res = run(a.n, a.seed)
    out = os.path.join(os.path.dirname(__file__), "..", "..", "reports", "gates", "G1.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "errors"}, indent=2))
    for k, v in res["errors"].items():
        print(f"  {k:8s} mean {v['mean']:.2e}  max {v['max']:.2e}")
    print("G1", "PASS" if res["passed"] else "FAIL")
    sys.exit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
