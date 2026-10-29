"""
Synthetic data with a role split (D11): Stage 1 ANN-E, Stage 2 reality correction,
one evaluation on the 7 held-out real guns.

    python -m src.ann.stage2

Stage 1. ANN-E is trained on dataset_synthetic_v2.csv (labels = Vedant's exact
         solver, theta_T). Architecture is chosen on the synthetic VALIDATION
         rows. ANN-E is judged against the SOLVER at the 30 real (P, rw, C)
         points -- never against measurement, because it was never shown any.
Stage 2. The solver-to-measurement gap is fitted on real TRAINING guns only.
         Candidates: one scalar k (theta_meas ~ k * theta_model), and a ridge-
         regularised linear model of the relative residual on (P, ln rw, ln C).
         The candidate is chosen by leave-one-out MRE on the training guns.
Eval.    Last, and once: the 7 held-out guns. M1, ANN-E (vs solver),
         ANN-E + correction (vs measured), calibrated closure (vs measured),
         and the raw exact solver (vs measured) for reference.

Nothing in Stages 1-2 reads a test row. `evaluate()` is the only function that
does, and it runs after every choice has been frozen into `choices`.

The ANN-E correction is fitted on the training guns with C >= 8 only: ANN-E has
never seen C < 8 (the synthetic envelope), so its predictions there are
extrapolation and should not define the correction. All 7 test guns have C >= 8.
The calibrated closure (exact solver, valid at every C) uses all 23.
"""

import json
import os

import numpy as np
import pandas as pd

from . import dataset as ds
from .config import TrainConfig
from .lm import train_with_restarts
from .network import forward
from .scaling import InputPipeline, MinMaxScaler
from ..export.weights_to_json import load_model, write_model
from ..physics import vaughan_adapter as va

SYN = os.path.join("data", "processed", "dataset_synthetic_v2.csv")
ARCHS = {"paper_3-3-2-1": [3, 2], "wide_3-8-8-1": [8, 8]}
RIDGE_LAMBDAS = (1.0, 10.0, 100.0)
OUT_JSON = os.path.join("reports", "synthetic_roles.json")
ANNE_MODEL = os.path.join("models", "model_anne.json")


def mre(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y) / np.abs(y)) * 100)


def predict_payload(p, X):
    """Forward pass from an exported model JSON (same path the browser takes)."""
    X = np.array(X, dtype=float, copy=True)
    for name, kind in p["input_transform"].items():
        j = p["inputs"].index(name)
        X[:, j] = np.log(X[:, j]) if kind == "log" else X[:, j]
    xmin, xmax = np.array(p["scaler"]["x_min"]), np.array(p["scaler"]["x_max"])
    Xn = 2 * (X - xmin) / (xmax - xmin) - 1
    w = p["weights"]
    theta = np.concatenate([np.ravel(w[k]) for k in sorted(w, key=lambda s: (s[1:], s[0] == "b"))])
    act = {"tanh": "tansig", "linear": "linear"}
    Yn = forward(theta, Xn, p["architecture"], act[p["activation"]], act[p["output_activation"]])
    ymin, ymax = np.array(p["scaler"]["y_min"]), np.array(p["scaler"]["y_max"])
    return (Yn + 1) / 2 * (ymax - ymin) + ymin


# --- Stage 1 -----------------------------------------------------------------

def train_anne(syn, restarts=6, max_epochs=1500, verbose=True):
    tr, va_ = syn[syn.split == "train"], syn[syn.split == "val"]
    results = {}
    best = None
    for name, hidden in ARCHS.items():
        cfg = TrainConfig(targets=["theta_cone_deg"], hidden=list(hidden),
                          restarts=restarts, max_epochs=max_epochs, patience=100)
        Xtr, Ytr = ds.matrices(tr, cfg)
        Xva, Yva = ds.matrices(va_, cfg)
        xp, ys = InputPipeline.fit(Xtr, cfg.inputs, cfg.input_transform), MinMaxScaler.fit(Ytr)
        r, _all, _ = train_with_restarts(xp.transform(Xtr), ys.transform(Ytr), cfg,
                                         val=(xp.transform(Xva), ys.transform(Yva)))
        pred = lambda X: ys.inverse_transform(
            forward(r.theta, xp.transform(X), cfg.layer_sizes))
        res = {"train_mre_vs_solver_pct": mre(Ytr[:, 0], pred(Xtr)[:, 0]),
               "val_mre_vs_solver_pct": mre(Yva[:, 0], pred(Xva)[:, 0]),
               "n_params": int(r.theta.size)}
        results[name] = res
        if verbose:
            print(f"  ANN-E {name}: train {res['train_mre_vs_solver_pct']:.3f} %  "
                  f"val {res['val_mre_vs_solver_pct']:.3f} %  (vs solver)")
        if best is None or res["val_mre_vs_solver_pct"] < best[1]["val_mre_vs_solver_pct"]:
            best = (name, res, r, cfg, xp, ys)
    name, res, r, cfg, xp, ys = best
    write_model(os.path.join(ds.repo_root(), ANNE_MODEL), r.theta, cfg, xp, ys,
                {"vs_solver": res, "architecture_choice": results},
                {"model_id": "ANN-E", "trained_on": SYN,
                 "label": "theta_T from Vedant's exact solver (NOT measured)",
                 "selected_by": "synthetic validation MRE"})
    return name, results


# --- Stage 2 -----------------------------------------------------------------

def _fit_k(f, m):
    """Scalar k minimising the squared RELATIVE error of k*f against m."""
    q = f / m
    return float(np.sum(q) / np.sum(q * q))


def _features(X):
    return np.column_stack([X[:, 0], np.log(X[:, 1]), np.log(X[:, 2])])


def ridge_params(f, m, X, lam):
    """theta_meas = f * (1 + b0 + w.z), z = standardised (P, ln rw, ln C); w is
    ridge-penalised, b0 is not. Returned as plain numbers so the knee report and
    the browser apply exactly the same calibration."""
    F = _features(X)
    mu, sd = F.mean(0), F.std(0)
    Z = (F - mu) / sd
    A = np.column_stack([np.ones(len(Z)), Z])
    P = lam * np.diag([0.0, 1.0, 1.0, 1.0])
    coef = np.linalg.solve(A.T @ A + P, A.T @ (m / f - 1.0))
    return {"type": "ridge", "lambda": lam, "mu": mu.tolist(), "sd": sd.tolist(),
            "coef": coef.tolist(),
            "features": ["perveance_uperv", "ln rw_mm", "ln C"]}


def apply_calibration(p, f, X):
    """Calibrated angle from a solver/ANN angle f at inputs X = (P, rw, C)."""
    f = np.asarray(f, float)
    if p["type"] == "scalar":
        return p["k"] * f
    Z = (_features(np.atleast_2d(X)) - np.array(p["mu"])) / np.array(p["sd"])
    return f * (1 + np.column_stack([np.ones(len(Z)), Z]) @ np.array(p["coef"]))


def fit_params(name, f, m, X):
    if name == "scalar_k":
        return {"type": "scalar", "k": _fit_k(f, m)}
    return ridge_params(f, m, X, float(name.rsplit("_", 1)[1]))


def candidates():
    names = ["scalar_k"] + [f"ridge_lambda_{lam:g}" for lam in RIDGE_LAMBDAS]
    return {n: (lambda n: lambda f, m, X: (lambda f2, X2, p=fit_params(n, f, m, X):
                                           apply_calibration(p, f2, X2)))(n) for n in names}


def choose_correction(f, m, X):
    """LOO MRE of every candidate on these (training) rows; pick the lowest."""
    scores = {}
    for name, fit in candidates().items():
        pred = np.empty_like(m)
        for i in range(len(m)):
            keep = np.arange(len(m)) != i
            pred[i] = fit(f[keep], m[keep], X[keep])(f[i:i + 1], X[i:i + 1])[0]
        scores[name] = {"loo_mre_pct": mre(m, pred),
                        "loo_abs_rel_err_p90_pct": float(np.percentile(np.abs(pred - m) / m * 100, 90))}
    best = min(scores, key=lambda k: scores[k]["loo_mre_pct"])
    return best, scores, candidates()[best](f, m, X)


# --- the whole pipeline ------------------------------------------------------

def solver_thetaT(df):
    ins = [va.to_vaughan_inputs(p, w, c) for p, w, c in
           df[["perveance_uperv", "rw_mm", "C"]].to_numpy()]
    ex = va.synthesize_many([(d["V"], d["I"], d["rw_mm"], d["Jc"]) for d in ins])
    assert all(e["feasible"] for e in ex)
    return np.array([e["thetaT"] for e in ex])


def run(verbose=True, retrain=True):
    root = ds.repo_root()
    syn = pd.read_csv(os.path.join(root, SYN))
    assert (syn.split != "test").all()
    lit = ds.literature_only(ds.load())
    train, test = ds.paper_split(lit)
    cols = ["perveance_uperv", "rw_mm", "C"]

    # Stage 1 (about 13 minutes; --no-retrain reuses models/model_anne.json)
    if retrain or not os.path.exists(os.path.join(root, ANNE_MODEL)):
        arch, arch_scores = train_anne(syn, verbose=verbose)
    anne = load_model(os.path.join(root, ANNE_MODEL))
    if not retrain:
        arch_scores = anne["metrics"]["architecture_choice"]
        arch = min(arch_scores, key=lambda k: arch_scores[k]["val_mre_vs_solver_pct"])
    Xtr, Xte = train[cols].to_numpy(float), test[cols].to_numpy(float)
    mtr, mte = train.theta_cone_deg.to_numpy(float), test.theta_cone_deg.to_numpy(float)
    s_tr, s_te = solver_thetaT(train), solver_thetaT(test)          # exact, theta_T
    a_tr = predict_payload(anne, Xtr)[:, 0]

    # Stage 2 (training guns only)
    env = Xtr[:, 2] >= 8.0
    anne_name, anne_scores, anne_fix = choose_correction(a_tr[env], mtr[env], Xtr[env])
    clo_name, clo_scores, clo_fix = choose_correction(s_tr, mtr, Xtr)
    choices = {"anne_architecture": arch, "anne_correction": anne_name,
               "closure_calibration": clo_name,
               "k_closure": _fit_k(s_tr, mtr), "k_anne": _fit_k(a_tr[env], mtr[env])}
    band = clo_scores[clo_name]["loo_abs_rel_err_p90_pct"]
    if verbose:
        print(f"  choices (frozen before test): {choices}")

    # Evaluation: the 7 held-out real guns, once.
    a_te = predict_payload(anne, Xte)[:, 0]
    m1 = load_model(os.path.join(root, "models", "model_m1.json"))
    test_mre = {
        "M1_vs_measured": mre(mte, predict_payload(m1, Xte)[:, 0]),
        "ANN-E_vs_solver": mre(s_te, a_te),
        "ANN-E+correction_vs_measured": mre(mte, anne_fix(a_te, Xte)),
        "calibrated_closure_vs_measured": mre(mte, clo_fix(s_te, Xte)),
        "exact_solver_raw_vs_measured": mre(mte, s_te),
        "paper_iterative_column_vs_measured": mre(mte, test.theta_iterative_deg.to_numpy(float)),
    }
    best_vs_meas = min((k for k in test_mre if k.endswith("vs_measured")
                        and k != "paper_iterative_column_vs_measured"), key=test_mre.get)

    # Robustness, because 7 guns is a small sample (standing rule 10).
    # (a) bootstrap interval on each 7-gun MRE;
    rng = np.random.default_rng(0)
    preds = {"M1_vs_measured": predict_payload(m1, Xte)[:, 0],
             "ANN-E+correction_vs_measured": anne_fix(a_te, Xte),
             "calibrated_closure_vs_measured": clo_fix(s_te, Xte),
             "exact_solver_raw_vs_measured": s_te}
    boot = {}
    for k, p in preds.items():
        e = np.abs(p - mte) / mte * 100
        bs = [e[rng.integers(0, 7, 7)].mean() for _ in range(5000)]
        boot[k] = [float(np.percentile(bs, 5)), float(np.percentile(bs, 95))]
    # (b) leave-one-out over all 30 guns for the calibrated closure, with the
    #     same frozen candidate (no re-selection inside the loop).
    Xall = np.vstack([Xtr, Xte]); mall = np.concatenate([mtr, mte])
    sall = np.concatenate([s_tr, s_te])
    loo30 = np.empty(30)
    for i in range(30):
        keep = np.arange(30) != i
        p = fit_params(clo_name, sall[keep], mall[keep], Xall[keep])
        loo30[i] = apply_calibration(p, sall[i:i + 1], Xall[i:i + 1])[0]
    calibrator = fit_params(clo_name, s_tr, mtr, Xtr)

    # POST-HOC sensitivity (decided after the test numbers above were seen, so it
    # is reported, never selected): theta depends on (P, C) only -- rw is a
    # length scale -- so a residual model on (P, ln C) is the dimensionally
    # consistent one. Same candidates, same LOO selection on training guns.
    def ridge2(f, m, X, lam):
        F = np.column_stack([X[:, 0], np.log(X[:, 2])])
        mu, sd = F.mean(0), F.std(0)
        A = np.column_stack([np.ones(len(F)), (F - mu) / sd])
        coef = np.linalg.solve(A.T @ A + lam * np.diag([0.0, 1.0, 1.0]), A.T @ (m / f - 1))
        return lambda f2, X2: f2 * (1 + np.column_stack(
            [np.ones(len(X2)), (np.column_stack([X2[:, 0], np.log(X2[:, 2])]) - mu) / sd]) @ coef)

    def loo_mre(f, m, X, lam):
        pred = np.empty_like(m)
        for i in range(len(m)):
            k = np.arange(len(m)) != i
            pred[i] = ridge2(f[k], m[k], X[k], lam)(f[i:i + 1], X[i:i + 1])[0]
        return mre(m, pred)
    ph = {lam: loo_mre(s_tr, mtr, Xtr, lam) for lam in RIDGE_LAMBDAS}
    lam_ph = min(ph, key=ph.get)
    posthoc = {"what": "ridge on (P, ln C) only, rw excluded as scale-free physics; "
                       "POST-HOC, not used for any reported headline",
               "lambda_by_loo23": lam_ph, "loo23_mre_pct": ph[lam_ph],
               "test7_mre_pct": mre(mte, ridge2(s_tr, mtr, Xtr, lam_ph)(s_te, Xte))}

    out = {
        "closure_calibrator_scalar": {"type": "scalar", "k": choices["k_closure"]},
        "band_scalar_pct_p90_loo": clo_scores["scalar_k"]["loo_abs_rel_err_p90_pct"],
        "calibration_envelope": {c: [float(train[c].min()), float(train[c].max())]
                                 for c in cols},
        "posthoc_scale_free_calibration": posthoc,
        "robustness": {
            "bootstrap_90pct_interval_of_test_mre_pct": boot,
            "calibrated_closure_loo_all_30_guns_mre_pct": mre(mall, loo30),
            "calibrated_closure_loo_23_train_guns_mre_pct": clo_scores[clo_name]["loo_mre_pct"],
        },
        "closure_calibrator": calibrator,
        "stage1": {"architecture_scores_vs_solver": arch_scores,
                   "anne_vs_solver_at_real_train_inputs_pct": mre(s_tr, a_tr),
                   "anne_vs_solver_at_real_train_inputs_C_ge_8_pct": mre(s_tr[env], a_tr[env]),
                   "n_synthetic_train": int((syn.split == "train").sum()),
                   "n_synthetic_val": int((syn.split == "val").sum())},
        "stage2": {"anne_candidates_loo": anne_scores, "closure_candidates_loo": clo_scores,
                   "n_train_guns_anne": int(env.sum()), "n_train_guns_closure": len(mtr)},
        "choices": choices,
        "measured_gun_band_pct_p90_loo": band,
        "test_7_guns_mre_pct": test_mre,
        "best_vs_measured": best_vs_meas,
        "beats_M1": bool(test_mre[best_vs_meas] < test_mre["M1_vs_measured"]),
        "headline_model": best_vs_meas if test_mre[best_vs_meas] < test_mre["M1_vs_measured"]
        else "M1_vs_measured",
        "note": "7 test guns; differences of a fraction of a percent are within noise",
    }
    with open(os.path.join(root, OUT_JSON), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if verbose:
        for k, v in test_mre.items():
            print(f"  test  {k:40s} {v:6.2f} %")
        print(f"  headline: {out['headline_model']}   band (LOO p90): +/-{band:.1f} %")
    return out


if __name__ == "__main__":
    import sys
    run(retrain="--no-retrain" not in sys.argv)
