"""
The supporting measurements quoted in reports/BENCHMARK.md.

    python -m src.ann.experiments

Each of these answers a question the write-up would otherwise have to hand-wave:

E1  Do the per-output residual weights matter?
    They are exposed as config because a future target on a different scale
    would need them. Both current targets are min-max scaled to [-1, 1] first,
    so the claim is that 1.0/1.0 is not merely a default but a no-op. That is a
    measurable claim, so it is measured rather than asserted.

E2  What does the second output cost the first?
    Rc_mm adds no independent information (D3) but does share the 26 parameters.
    Running n_out = 1 against n_out = 2 on the same split prices that.

E3  How much of the headline number is the restart lottery?
    The spread of test MRE across seeds, and whether the cross-validation score
    can identify a good seed at all.

E4  Is the skew in C the problem?
    C spans 5.12 to 306.3 but 21 of 23 training rows sit below 51, so after
    linear min-max scaling almost every row lands near -0.9 and the feature
    loses resolution exactly where test case 12 (C = 111) falls. Replacing C
    with ln C is the one physically motivated alternative -- the physics enters
    through ln(Rc/Ra), and VALIDATION_table2.md already found the closure error
    correlates with ln C at R = +0.794.

    E4 IS A DIAGNOSTIC, NOT A CHANGE. Per D5, sweeping one physically meaningful
    choice and reporting where it lands is legitimate; adopting whichever
    preprocessing makes the test number smallest would be fitting to the test
    set, which is the same 7 rows the result is reported on. M1 keeps the
    paper's preprocessing.
"""

import dataclasses
import json
import os

import numpy as np

from . import dataset as ds
from .config import TrainConfig
from .lm import train_lm
from .metrics import mre, pearson_r, rmse
from .network import predict
from .scaling import MinMaxScaler
from .train import cv_score, fit_scalers

N_SEEDS = 30


def _fit_and_score(Xtr, Ytr, Xte, Yte, cfg, seed, epochs=None):
    c = cfg if epochs is None else dataclasses.replace(cfg, max_epochs=epochs)
    xs, ys = fit_scalers(Xtr, Ytr)
    r = train_lm(xs.transform(Xtr), ys.transform(Ytr), c, seed=seed)
    P_tr = ys.inverse_transform(predict(r.theta, xs.transform(Xtr), c))
    P_te = ys.inverse_transform(predict(r.theta, xs.transform(Xte), c))
    return {
        "train_mre": mre(Ytr[:, 0], P_tr[:, 0]),
        "test_mre": mre(Yte[:, 0], P_te[:, 0]),
        "train_rmse": rmse(Ytr[:, 0], P_tr[:, 0]),
        "test_rmse": rmse(Yte[:, 0], P_te[:, 0]),
        "test_r": pearson_r(Yte[:, 0], P_te[:, 0]),
        "epochs": r.epochs_run,
    }


def _spread(Xtr, Ytr, Xte, Yte, cfg, epochs=None, n=N_SEEDS):
    runs = [_fit_and_score(Xtr, Ytr, Xte, Yte, cfg, 100 + i, epochs) for i in range(n)]
    t = np.array([r["test_mre"] for r in runs])
    tr = np.array([r["train_mre"] for r in runs])
    return {
        "n_seeds": n,
        "test_mre_min": float(t.min()),
        "test_mre_median": float(np.median(t)),
        "test_mre_max": float(t.max()),
        "train_mre_median": float(np.median(tr)),
        "frac_below_paper_3p74": float(np.mean(t < 3.74)),
    }


def main():
    root = ds.repo_root()
    base = TrainConfig()
    df = ds.load(cfg=base)
    tr_df, te_df = ds.paper_split(df)
    out = {}

    # ---------------- E1: do the residual weights matter? ----------------
    Xtr, Ytr = ds.matrices(tr_df, base)
    Xte, Yte = ds.matrices(te_df, base)
    e1 = {}
    for label, w in [("1.0 / 1.0 (default)", [1.0, 1.0]),
                     ("4.0 / 1.0 (theta favoured)", [4.0, 1.0]),
                     ("1.0 / 4.0 (Rc favoured)", [1.0, 4.0]),
                     ("1.0 / 0.0 (Rc switched off)", [1.0, 0.0])]:
        cfg = dataclasses.replace(base, residual_weights=w)
        e1[label] = _spread(Xtr, Ytr, Xte, Yte, cfg, epochs=200, n=N_SEEDS)
        print(f"E1 {label:30} test MRE median {e1[label]['test_mre_median']:6.2f} % "
              f"(min {e1[label]['test_mre_min']:5.2f}, max {e1[label]['test_mre_max']:6.2f})")
    out["E1_residual_weights"] = e1

    # ---------------- E2: what does the second output cost? ----------------
    e2 = {}
    for label, targets in [("n_out = 1 (theta only)", ["theta_cone_deg"]),
                           ("n_out = 2 (theta + Rc_mm)", ["theta_cone_deg", "Rc_mm"])]:
        cfg = dataclasses.replace(base, targets=targets)
        A, B = ds.matrices(tr_df, cfg)
        C, D = ds.matrices(te_df, cfg)
        e2[label] = _spread(A, B, C, D, cfg, epochs=200, n=N_SEEDS)
        e2[label]["n_params"] = 12 + (3 * 2 + 2) + (2 * len(targets) + len(targets))
        print(f"E2 {label:30} test MRE median {e2[label]['test_mre_median']:6.2f} % "
              f"train MRE median {e2[label]['train_mre_median']:5.2f} %")
    out["E2_second_output_cost"] = e2

    # ---------------- E3: can CV identify a good restart? ----------------
    e3 = {}
    for label, targets in [("n_out = 1", ["theta_cone_deg"]),
                           ("n_out = 2", ["theta_cone_deg", "Rc_mm"])]:
        cfg = dataclasses.replace(base, targets=targets)
        A, B = ds.matrices(tr_df, cfg)
        C, D = ds.matrices(te_df, cfg)
        cvs, tests = [], []
        for i in range(20):
            seed = 100 + i
            score, epochs, _, _ = cv_score(A, B, cfg, seed, base.cv_folds)
            budget = max(int(np.median(epochs)), 1)
            s = _fit_and_score(A, B, C, D, cfg, seed, budget)
            cvs.append(score)
            tests.append(s["test_mre"])
        e3[label] = {
            "n_seeds": len(cvs),
            "corr_cv_vs_test_mre": pearson_r(np.array(cvs), np.array(tests)),
            "test_mre_min": float(np.min(tests)),
            "test_mre_median": float(np.median(tests)),
            "test_mre_max": float(np.max(tests)),
        }
        print(f"E3 {label:30} corr(CV, test MRE) = "
              f"{e3[label]['corr_cv_vs_test_mre']:+.3f}")
    out["E3_restart_lottery"] = e3

    # ---------------- E4: linear C against ln C (DIAGNOSTIC ONLY) ----------------
    e4 = {}
    for label, targets in [("n_out = 1", ["theta_cone_deg"]),
                           ("n_out = 2", ["theta_cone_deg", "Rc_mm"])]:
        cfg = dataclasses.replace(base, targets=targets)
        A, B = ds.matrices(tr_df, cfg)
        C, D = ds.matrices(te_df, cfg)
        for scaling in ("linear C (as shipped)", "ln C (diagnostic only)"):
            A2, C2 = A.copy(), C.copy()
            if scaling.startswith("ln"):
                j = cfg.inputs.index("C")
                A2[:, j] = np.log(A2[:, j])
                C2[:, j] = np.log(C2[:, j])
            e4[f"{label} / {scaling}"] = _spread(A2, B, C2, D, cfg, epochs=200, n=N_SEEDS)
            s = e4[f"{label} / {scaling}"]
            print(f"E4 {label} / {scaling:24} test MRE median "
                  f"{s['test_mre_median']:6.2f} % (min {s['test_mre_min']:5.2f})")
    out["E4_C_scaling_diagnostic"] = e4

    # training-row skew in C, quoted in the write-up
    Ctr = tr_df["C"].to_numpy(dtype=float)
    out["C_distribution"] = {
        "train_min": float(Ctr.min()), "train_max": float(Ctr.max()),
        "train_median": float(np.median(Ctr)),
        "train_rows_above_51": int((Ctr > 51).sum()),
        "test_case_12_C": 111.0,
    }

    path = os.path.join(root, "reports", "experiments.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
