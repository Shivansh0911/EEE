"""
Train a model. CLI entry point for P5.

    python -m src.ann.train --model m1

M1 is the replication: trained on the paper's 23 training cases, tested on the
paper's 7 asterisked cases. It is the correctness check on the whole pipeline
rather than an attempt at a new result -- if M1 cannot roughly reproduce the
paper's numbers on the paper's own split, the fault is in this code, not in the
data.

M2 (all Tier A, held-out Tier A) and M3 (Tier A + Tier B, tested on real guns
only) are NOT trained here. M3 requires Tier B, which does not exist because the
physics gate failed (D4), and M2 without M3 has nothing to be compared against.
Both are refused explicitly rather than silently skipped.

Model selection without a validation set
----------------------------------------
The paper's split leaves no room for one: 30 rows minus 7 test leaves 23, and
carving a validation set out of those gives four rows, which is noise. So
restarts are scored by k-fold cross-validation INSIDE the 23 training rows:

  * for each restart seed, run k folds; within each fold the scaler is refit on
    that fold's training rows only, so no fold's validation data touches its own
    normalisation;
  * the seed's score is its mean out-of-fold MSE;
  * the winning seed is refit on all 23 rows, with an epoch budget taken from
    the median stopping epoch of its own folds. That is how early stopping is
    obtained without a held-out set: the budget is chosen by data the final fit
    is then trained on, but never by the test set.

The 7 test rows take no part in any of this.
"""

import argparse
import dataclasses
import json
import os

import numpy as np

from . import dataset as ds
from .config import TrainConfig
from .lm import train_lm, train_with_restarts
from .metrics import format_table, pearson_r, per_output
from .network import n_params, predict
from .scaling import MinMaxScaler, save_scalers, saturation_report


def fit_scalers(X, Y):
    """Fit on the given rows only. Callers must pass TRAINING rows."""
    return MinMaxScaler.fit(X), MinMaxScaler.fit(Y)


def cv_score(X, Y, cfg, seed, folds, rng_seed=0):
    """
    Mean out-of-fold MSE for one restart seed, in normalised units.

    Returns (mean_oof_mse, stop_epochs, oof_predictions_physical, fold_curves),
    where fold_curves is a list of (train_history, val_history) per fold. The
    curves are what figure 4 plots: with no held-out validation set to speak of,
    the honest "validation" curve is the mean of the folds' own.
    """
    rng = np.random.default_rng(rng_seed)
    parts = ds.kfold_indices(len(X), folds, rng)
    oof = np.full_like(Y, np.nan, dtype=float)
    scores, epochs, curves = [], [], []

    for hold in parts:
        mask = np.ones(len(X), dtype=bool)
        mask[hold] = False
        Xtr, Ytr = X[mask], Y[mask]
        Xva, Yva = X[hold], Y[hold]

        xs, ys = fit_scalers(Xtr, Ytr)          # fold-train only: no leak
        r = train_lm(xs.transform(Xtr), ys.transform(Ytr), cfg,
                     val=(xs.transform(Xva), ys.transform(Yva)), seed=seed)

        oof[hold] = ys.inverse_transform(predict(r.theta, xs.transform(Xva), cfg))
        scores.append(r.val_mse if r.val_mse is not None else r.train_mse)
        epochs.append(r.best_epoch + 1)
        curves.append((list(r.history), list(r.val_history)))

    return float(np.mean(scores)), epochs, oof, curves


def train_m1(cfg, args):
    root = ds.repo_root()
    df = ds.load(cfg=cfg)
    train_df, test_df = ds.paper_split(df)
    ds.assert_no_test_leakage(train_df, test_df)

    X_tr, Y_tr = ds.matrices(train_df, cfg)
    X_te, Y_te = ds.matrices(test_df, cfg)

    print(f"architecture {cfg.layer_sizes}  ->  {n_params(cfg.layer_sizes)} parameters")
    print(f"targets      {cfg.targets}")
    print(f"residual wts {cfg.weights}")
    print(f"split        {len(train_df)} train / {len(test_df)} test (paper's own)")
    print(f"             test cases {sorted(test_df['source_case_id'])}")

    # --- restart selection by cross-validation inside the training rows ---
    folds = cfg.cv_folds
    seed_scores = []
    if folds:
        label = "leave-one-out" if folds == -1 else f"{folds}-fold"
        print(f"\nselecting among {cfg.restarts} Nguyen-Widrow restarts "
              f"by {label} CV within the {len(train_df)} training rows")
        for i in range(cfg.restarts):
            seed = cfg.seed + i
            score, epochs, oof, curves = cv_score(X_tr, Y_tr, cfg, seed, folds)
            seed_scores.append({"seed": seed, "cv_mse": score, "epochs": epochs,
                                "oof": oof, "curves": curves})
            print(f"  seed {seed:4d} | mean out-of-fold MSE {score:.6e} "
                  f"| median stop epoch {int(np.median(epochs))}")
        seed_scores.sort(key=lambda d: d["cv_mse"])
        best = seed_scores[0]
        budget = max(int(np.median(best["epochs"])), 1)
        print(f"  -> seed {best['seed']}, epoch budget {budget} "
              f"(median of its own folds)")
    else:
        best, budget = None, cfg.max_epochs

    # --- final fit on all 23 training rows ---
    x_scaler, y_scaler = fit_scalers(X_tr, Y_tr)
    Xn_tr, Yn_tr = x_scaler.transform(X_tr), y_scaler.transform(Y_tr)
    Xn_te = x_scaler.transform(X_te)

    if best is not None:
        final_cfg = dataclasses.replace(cfg, max_epochs=budget)
        result = train_lm(Xn_tr, Yn_tr, final_cfg, seed=best["seed"], verbose=args.verbose)
        cv_mse = best["cv_mse"]
        oof = best["oof"]
    else:
        result, all_runs, score = train_with_restarts(Xn_tr, Yn_tr, cfg,
                                                      verbose=args.verbose)
        print("  WARNING: CV disabled, so restarts were selected on TRAINING MSE. "
              "That is selection on the training set and gives no estimate of "
              "generalisation.")
        cv_mse, oof = None, None

    print(f"\ntrained: {result.epochs_run} epochs, "
          f"{result.accepted_steps} accepted / {result.rejected_steps} rejected steps, "
          f"stopped on {result.stop_reason}")

    restart_spread = restart_variance(seed_scores, cfg, X_tr, Y_tr, X_te, Y_te)

    # --- evaluation, in physical units ---
    Pn_tr = predict(result.theta, Xn_tr, cfg)
    Pn_te = predict(result.theta, Xn_te, cfg)
    P_tr = y_scaler.inverse_transform(Pn_tr)
    P_te = y_scaler.inverse_transform(Pn_te)

    blocks = {
        "train": per_output(Y_tr, P_tr, cfg.targets),
        "test": per_output(Y_te, P_te, cfg.targets),
    }
    print("\n" + format_table(blocks, cfg.targets))

    if cv_mse is not None:
        cv_blocks = {"cv (out-of-fold, train rows)": per_output(Y_tr, oof, cfg.targets)}
        print("\n" + format_table(cv_blocks, cfg.targets))

    # --- saturation warnings ---
    warn_saturation("train", Pn_tr, train_df, cfg)
    warn_saturation("test", Pn_te, test_df, cfg)

    # --- leakage sanity check (standing rule 10) ---
    for name in cfg.targets:
        m = blocks["test"][name]
        if m["mre_pct"] < 1.0 or m["pearson_r"] > 0.999:
            print(f"\n  SUSPECT LEAKAGE: test {name} MRE {m['mre_pct']:.2f} % / "
                  f"R {m['pearson_r']:.4f} is better than 30 real rows should allow. "
                  f"Check for duplicate guns across the split and for a scaler "
                  f"that has seen test data before believing this.")

    # --- persist ---
    os.makedirs(os.path.join(root, "models"), exist_ok=True)
    scaler_path = os.path.join(root, "models", "scaler.json")
    save_scalers(scaler_path, x_scaler, y_scaler, cfg.inputs, cfg.targets)
    print(f"\nwrote {scaler_path}")

    run = {
        "model_id": "M1",
        "config": dataclasses.asdict(cfg),
        "n_params": n_params(cfg.layer_sizes),
        "seed": int(result.seed) if result.seed is not None else None,
        "epoch_budget": budget,
        "epochs_run": result.epochs_run,
        "accepted_steps": result.accepted_steps,
        "rejected_steps": result.rejected_steps,
        "stop_reason": result.stop_reason,
        "history": result.history,
        "cv": None if cv_mse is None else {
            "folds": folds,
            "mean_oof_mse_normalised": cv_mse,
            "seed_scores": [{"seed": s["seed"], "cv_mse": s["cv_mse"]}
                            for s in seed_scores],
            "oof_predictions": np.asarray(oof).tolist(),
            "fold_curves": best["curves"],
        },
        "metrics": blocks,
        "restart_spread": restart_spread,
        "theta": result.theta.tolist(),
        "train_case_ids": [int(c) for c in train_df["source_case_id"]],
        "test_case_ids": [int(c) for c in test_df["source_case_id"]],
        "predictions": {
            "train": P_tr.tolist(),
            "test": P_te.tolist(),
        },
        "truth": {"train": Y_tr.tolist(), "test": Y_te.tolist()},
    }
    run_path = os.path.join(root, "reports", "m1_run.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)
    print(f"wrote {run_path}")

    return result, cfg, x_scaler, y_scaler, blocks, run


def restart_variance(seed_scores, cfg, X_tr, Y_tr, X_te, Y_te):
    """
    POST-HOC DIAGNOSTIC. Not used for selection, and it must not be.

    Refits every restart seed on the full training set and records what each
    would have scored on the 7 test cases. The point is to show how much of the
    headline number is the model and how much is which local minimum LM happened
    to land in -- with 26 parameters and 23 rows that spread is the single most
    important thing to know, and reporting one number hides it.

    The test rows are touched only here, after the model has already been chosen
    by cross-validation on the training rows. Choosing a seed from this table
    would be selecting on the test set.
    """
    if not seed_scores:
        return None

    x_scaler, y_scaler = fit_scalers(X_tr, Y_tr)
    Xn_tr, Yn_tr = x_scaler.transform(X_tr), y_scaler.transform(Y_tr)
    Xn_te = x_scaler.transform(X_te)

    rows = []
    for s in seed_scores:
        budget = max(int(np.median(s["epochs"])), 1)
        r = train_lm(Xn_tr, Yn_tr, dataclasses.replace(cfg, max_epochs=budget),
                     seed=s["seed"])
        P_tr = y_scaler.inverse_transform(predict(r.theta, Xn_tr, cfg))
        P_te = y_scaler.inverse_transform(predict(r.theta, Xn_te, cfg))
        rows.append({
            "seed": s["seed"],
            "cv_mse": s["cv_mse"],
            "epoch_budget": budget,
            "train_mre_pct": per_output(Y_tr, P_tr, cfg.targets)[cfg.targets[0]]["mre_pct"],
            "test_mre_pct": per_output(Y_te, P_te, cfg.targets)[cfg.targets[0]]["mre_pct"],
        })

    t = np.array([r["test_mre_pct"] for r in rows])
    c = np.array([r["cv_mse"] for r in rows])
    corr = pearson_r(c, t)

    print(f"\nrestart spread on {cfg.targets[0]} (post-hoc, never used to choose):")
    print(f"  test MRE over {len(rows)} restarts: min {t.min():.2f} % | "
          f"median {np.median(t):.2f} % | max {t.max():.2f} %")
    print(f"  correlation between CV score and test MRE: R = {corr:+.3f}")
    if abs(corr) < 0.5:
        print("  -> the CV score barely ranks the restarts. With 23 rows the CV "
              "estimate is itself too noisy to identify a good minimum, so which "
              "restart wins is close to arbitrary. Read the spread, not the point.")
    return {"rows": rows, "corr_cv_vs_test": corr,
            "test_mre_min": float(t.min()), "test_mre_median": float(np.median(t)),
            "test_mre_max": float(t.max())}


def warn_saturation(split, Pn, df, cfg):
    hits = saturation_report(Pn, cfg.saturation_limit)
    if not hits:
        return
    print(f"\n  SATURATION WARNING ({split}): {len(hits)} normalised prediction(s) "
          f"beyond +/-{cfg.saturation_limit}.")
    print("  A tansig output over min-max scaled targets asymptotes at +/-1, which "
          "maps back to the training range's edge, so these are the network "
          "against its ceiling rather than confident extrapolation.")
    for i, k, v in hits:
        case = df["source_case_id"].iloc[i]
        print(f"    case {case:>3}  {cfg.targets[k]:<16} normalised {v:+.4f}")


def build_config(args):
    cfg = TrainConfig()
    if args.targets:
        cfg = dataclasses.replace(cfg, targets=args.targets)
    if args.residual_weights:
        cfg = dataclasses.replace(cfg, residual_weights=args.residual_weights)
    return dataclasses.replace(
        cfg,
        restarts=args.restarts,
        max_epochs=args.max_epochs,
        cv_folds=args.cv_folds,
        output_activation=args.output_activation,
        seed=args.seed,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the Pierce-gun surrogate.")
    p.add_argument("--model", default="m1", choices=["m1", "m2", "m3"])
    p.add_argument("--restarts", type=int, default=10)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--cv-folds", type=int, default=5,
                   help="k-fold CV within the training rows; 0 disables, -1 is LOO")
    p.add_argument("--output-activation", default="tansig", choices=["tansig", "linear"])
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--targets", nargs="+", default=None)
    p.add_argument("--residual-weights", nargs="+", type=float, default=None)
    p.add_argument("--export", default=None, help="path for the B.3 model JSON")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    if args.model in ("m2", "m3"):
        raise SystemExit(
            f"{args.model.upper()} is not trainable. M3 needs Tier B, which has zero "
            f"rows because the physics gate failed at MAE 1.66 deg / MRE 5.75 % "
            f"against a required 0.5 deg / 1.5 % (see reports/VALIDATION_table2.md "
            f"and D4). M2 is only meaningful as M3's control. Neither is faked here."
        )

    cfg = build_config(args)
    result, cfg, x_scaler, y_scaler, blocks, run = train_m1(cfg, args)

    if args.export:
        from ..export.weights_to_json import write_model
        import datetime as dt
        metrics = {
            "train_mre_pct": {k: blocks["train"][k]["mre_pct"] for k in cfg.targets},
            "test_mre_pct": {k: blocks["test"][k]["mre_pct"] for k in cfg.targets},
            "test_rmse": {k: blocks["test"][k]["rmse"] for k in cfg.targets},
            "test_pearson_r": {k: blocks["test"][k]["pearson_r"] for k in cfg.targets},
        }
        provenance = {
            "model_id": "M1",
            "trained_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "n_train": len(run["train_case_ids"]),
            "n_test": len(run["test_case_ids"]),
            "tier_a_rows": 30,
            "tier_b_rows": 0,
            "tier_c_rows": 0,
            "split": "paper's own 23/7 (Table 2 asterisked cases as test)",
            "seed": run["seed"],
            "restarts": cfg.restarts,
            "dataset": "data/processed/dataset_master.csv",
            "notes": ("second target Rc_mm is a PLACEHOLDER under reading C of D3 "
                      "and is a deterministic function of the inputs and theta"),
        }
        path = args.export if os.path.isabs(args.export) else os.path.join(
            ds.repo_root(), args.export)
        write_model(path, result.theta, cfg, x_scaler, y_scaler, metrics, provenance)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
