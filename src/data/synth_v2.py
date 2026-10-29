"""
Synthetic data v2 (tier B2): Latin-hypercube designs labelled by Vedant's EXACT solver.

    python -m src.data.synth_v2 [--n-draw 1088]

THE RULE (D11). Synthetic rows may only teach what real rows cannot, and are
never test data. What real rows cannot teach is the *shape* of the classical
synthesis map (P, rw, C) -> theta_T between and around 23 training guns. What
synthetic rows cannot teach is reality: the exact solver itself sits ~10 % from
measured guns. So:

  Stage 1 (this file): rows whose label is the exact solver's theta_T. ANN-E is
           trained on them and judged against the SOLVER, never against
           measurement.
  Stage 2 (src/ann/stage2.py): the solver-to-measurement gap is fitted on the
           23 real training guns only.

Why v1 is kept but not used: v1 (dataset_tier_b.csv) was labelled by our probe
closure with the full-sphere Langmuir-Blodgett constant 29.33 where the cone
needs 14.665, plus a fitted lens coefficient 0.62; it sits 3.22 % MRE from the
exact solver. See D10.

The envelope is the solver's validated region, not a convenience box:
  C >= 8 (D9), C <= 306.3 and P, rw within Panahi Table 2's ranges,
  theta_raw <= 56 deg (Eq.(20) and the aperture are defined; 56.44 is the limit),
  -alpha below the vendored Newton inversion's convergence limit (7.9).
Every row carries its flags so a reader can check rather than trust.
"""

import argparse
import hashlib
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import qmc

from ..ann import dataset as ds
from ..physics import closure
from ..physics import vaughan_adapter as va

P_RANGE = (0.30, 3.67)        # Panahi Table 2
RW_RANGE = (0.51, 4.42)       # mm, Panahi Table 2
C_RANGE = (8.0, 306.3)        # C >= 8 envelope (D9); top = Table 2 max
THETA_RAW_MAX = 56.0          # deg; Eq.(20) fails at 56.44
VAL_FRACTION = 0.15           # ANN-E early-stopping split; synthetic, so never "test"
TIER = "B2_synthetic"
OUT = os.path.join("data", "processed", "dataset_synthetic_v2.csv")


def vendor_sha1():
    path = os.path.join(ds.repo_root(), "src", "physics", "vendor", "pierce_gun_design_fixed.py")
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()[:12]


def draw(n, seed):
    u = qmc.LatinHypercube(d=3, seed=seed).random(n)
    P = P_RANGE[0] + u[:, 0] * (P_RANGE[1] - P_RANGE[0])
    rw = RW_RANGE[0] + u[:, 1] * (RW_RANGE[1] - RW_RANGE[0])
    C = np.exp(math.log(C_RANGE[0]) + u[:, 2] * (math.log(C_RANGE[1]) - math.log(C_RANGE[0])))
    return P, rw, C


def test_triples():
    m = pd.read_csv(ds.default_path())
    t = m[(m.tier == ds.LITERATURE_TIER) & (m.paper_split == "test")]
    return set(map(tuple, t[["perveance_uperv", "rw_mm", "C"]].round(4).to_numpy()))


def generate(n_draw=1088, seed=20261029, workers=None):
    P, rw, C = draw(n_draw, seed)
    inputs = [va.to_vaughan_inputs(p, w, c) for p, w, c in zip(P, rw, C)]
    exact = va.synthesize_many([(d["V"], d["I"], d["rw_mm"], d["Jc"]) for d in inputs],
                               workers=workers)
    em = closure.from_pcr(P, rw, C, V=va.V_REF)
    sha = vendor_sha1()
    banned = test_triples()

    rows, stats = [], {"drawn": n_draw, "infeasible": 0, "theta_raw_gt_56": 0,
                       "alpha_beyond_newton": 0}
    for i, (d, ex) in enumerate(zip(inputs, exact)):
        if not ex["feasible"]:
            stats["infeasible"] += 1
            continue
        alpha_ok = bool(em["alpha_ok"][i])
        if ex["theta"] > THETA_RAW_MAX:
            stats["theta_raw_gt_56"] += 1
            continue
        if not alpha_ok:
            stats["alpha_beyond_newton"] += 1
            continue
        key = (round(P[i], 4), round(rw[i], 4), round(C[i], 4))
        assert key not in banned, f"draw {key} coincides with a held-out test gun"
        rows.append({
            "tier": TIER,
            "generator": f"vendor pierce_synthesis (exact, mpmath) sha1:{sha}",
            "theta_source": "vaughan_exact_solver",
            "beam_type": "pencil",
            "perveance_uperv": P[i], "rw_mm": rw[i], "C": C[i],
            "V_ref_V": d["V"], "I_A": d["I"], "Jc_Acm2": d["Jc"],
            "kf": ex["rel_factor"],
            "theta_raw_deg": ex["theta"], "thetaT_deg": ex["thetaT"],
            "theta_cone_deg": ex["thetaT"],        # the ANN target name (= theta_T)
            "Rc_mm": ex["Rc_mm"], "zac_mm": ex["zac_mm"], "zw_mm": ex["zw_mm"],
            "za_valid": ex["za_valid"],
            "env_C_ge_8": C[i] >= 8.0,
            "env_theta_raw_le_56": ex["theta"] <= THETA_RAW_MAX,
            "env_P_rw_in_table2": (P_RANGE[0] <= P[i] <= P_RANGE[1]
                                   and RW_RANGE[0] <= rw[i] <= RW_RANGE[1]),
            "env_alpha_newton_ok": alpha_ok,
            "emulator_thetaT_relerr": abs(em["thetaT"][i] - ex["thetaT"]) / ex["thetaT"],
            "notes": "SYNTHETIC - exact classical-solver output at V_ref = 1 kV, "
                     "not a measurement; never a test row",
        })
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    val = rng.random(len(df)) < VAL_FRACTION
    df.insert(0, "split", np.where(val, "val", "train"))
    df.insert(0, "row_id", [f"SYN2-{i:05d}" for i in range(1, len(df) + 1)])
    stats["kept"] = len(df)
    return df, stats


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--n-draw", type=int, default=1088)
    p.add_argument("--seed", type=int, default=20261029)
    a = p.parse_args(argv)
    df, stats = generate(a.n_draw, a.seed)
    path = os.path.join(ds.repo_root(), OUT)
    df.to_csv(path, index=False, float_format="%.10g")
    print(stats)
    print(f"theta_T {df.thetaT_deg.min():.2f}..{df.thetaT_deg.max():.2f} deg; "
          f"max emulator rel err {df.emulator_thetaT_relerr.max():.2e}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
