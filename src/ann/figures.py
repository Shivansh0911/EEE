"""
Reproduce the paper's figures 3-6 from a training run, into reports/figures/.

    python -m src.ann.figures

Reads reports/m1_run.json (written by train.py) and the master dataset, so the
figures can be regenerated without retraining and always correspond to the run
that is actually reported.

Colour is assigned by entity and never cycled or repainted: each synthesis
method keeps its hue across figures 5 and 6, so a reader who learns "orange is
the iterative method" in one figure is not re-taught in the next. The four hues
were validated all-pairs for colour-vision deficiency before use, which is why
the fourth is violet rather than the obvious yellow -- yellow beside orange
fails the normal-vision separation floor. Aqua sits below 3:1 against the
surface, so every series is also named in a legend and tabulated in BENCHMARK.md
rather than being identified by colour alone.

The experimental column is drawn in neutral ink rather than a series colour: it
is the reference the methods are judged against, not one of the competitors.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import dataset as ds

# --- design tokens (light surface; see the dataviz reference palette) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8a85"
GRID = "#e3e2de"

SERIES = {
    "ann": "#2a78d6",          # slot 1, blue   -- this work (M1)
    "iterative": "#eb6834",    # slot 2, orange
    "noniterative": "#1baf7a", # slot 3, aqua
    "modified": "#4a3aa7",     # slot 4, violet
}

METHODS = [
    ("theta_iterative_deg", "Iterative", SERIES["iterative"], "s"),
    ("theta_noniterative_deg", "Non-iterative", SERIES["noniterative"], "^"),
    ("theta_modified_noniter_deg", "Modified non-iterative", SERIES["modified"], "D"),
]

DPI = 200


def _style(ax, title=None, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    if title:
        ax.set_title(title, color=INK, fontsize=12, pad=12, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def _save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {path}")


def _legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, **kw)
    return leg


def fig3_regression(run, out_dir, target_index=0):
    """
    Predicted vs experimental, with the y = x line the points would sit on if
    the model were exact. One file per split, as in the paper.

    A regression plot is the right form here because the question is agreement
    between two measurements of the same quantity, not a trend: the diagonal is
    the hypothesis, and distance from it is the error.
    """
    name = run["config"]["targets"][target_index]
    unit = "deg" if name.startswith("theta") else "mm"

    for split in ("train", "test"):
        y = np.array(run["truth"][split])[:, target_index]
        p = np.array(run["predictions"][split])[:, target_index]
        r = run["metrics"][split][name]

        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        lo = min(y.min(), p.min()) * 0.92
        hi = max(y.max(), p.max()) * 1.06
        ax.plot([lo, hi], [lo, hi], color=INK_MUTED, linewidth=1.5,
                linestyle="--", zorder=1, label="perfect agreement (y = x)")
        ax.scatter(y, p, s=64, color=SERIES["ann"], edgecolor=SURFACE,
                   linewidth=2.0, zorder=3, label=f"M1 prediction ({split})")

        _style(ax,
               title=f"Fig. 3 — {split} regression, {name}",
               xlabel=f"Experimental {name} [{unit}]",
               ylabel=f"Predicted {name} [{unit}]")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.text(0.04, 0.95,
                f"n = {r['n']}\nR = {r['pearson_r']:.4f}\n"
                f"RMSE = {r['rmse']:.2f} {unit}\nMRE = {r['mre_pct']:.2f} %",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=9, color=INK_2, linespacing=1.6)
        _legend(ax, loc="lower right")
        _save(fig, os.path.join(out_dir, f"fig3_regression_{split}.png"))


def fig4_mse_vs_epochs(run, out_dir):
    """
    Training and validation MSE against epoch, log y.

    The validation curve is NOT a held-out split -- there is no room for one in
    23 rows. It is the mean of the winning restart's own cross-validation folds,
    each fold's validation MSE, which is the honest substitute. The gap between
    the two curves is the overfitting, and it opens early.
    """
    hist = np.array(run["history"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(np.arange(1, len(hist) + 1), hist, color=SERIES["ann"],
            linewidth=2.0, zorder=3, label="training MSE (final fit, 23 rows)")

    cv = run.get("cv") or {}
    curves = cv.get("fold_curves")
    if curves:
        vals = [np.array(v, dtype=float) for _, v in curves if len(v)]
        if vals:
            n = min(len(v) for v in vals)
            mean_val = np.mean([v[:n] for v in vals], axis=0)
            ax.plot(np.arange(1, n + 1), mean_val, color=SERIES["iterative"],
                    linewidth=2.0, zorder=3,
                    label=f"validation MSE (mean of {len(vals)} CV folds)")
            best = int(np.argmin(mean_val))
            ax.axvline(best + 1, color=INK_MUTED, linewidth=1.2, linestyle=":",
                       zorder=2)
            ax.annotate(f"validation minimum, epoch {best + 1}",
                        xy=(best + 1, mean_val[best]),
                        xytext=(8, 14), textcoords="offset points",
                        fontsize=9, color=INK_2)

    ax.set_yscale("log")
    _style(ax, title="Fig. 4 — MSE against epoch (normalised units)",
           xlabel="Epoch", ylabel="MSE (normalised)")
    _legend(ax, loc="upper right")
    _save(fig, os.path.join(out_dir, "fig4_mse_vs_epochs.png"))


def _all_case_predictions(run, df):
    """ANN prediction per case id, from the run's train and test blocks."""
    pred = {}
    for split in ("train", "test"):
        ids = run[f"{split}_case_ids"]
        P = np.array(run["predictions"][split])
        for case, row in zip(ids, P):
            pred[int(case)] = (float(row[0]), split)
    return pred


def fig5_method_comparison(run, df, out_dir):
    """
    Every method against the experimental value, across all 30 real cases,
    ordered by experimental theta so the eye reads a curve rather than noise.

    Test cases are marked. Twenty-three of the thirty blue points are training
    points, and a figure that did not say so would be showing the model its own
    homework and calling it agreement.
    """
    d = df.sort_values("theta_cone_deg").reset_index(drop=True)
    x = np.arange(len(d))
    pred = _all_case_predictions(run, df)
    ann = np.array([pred[int(c)][0] for c in d["source_case_id"]])
    is_test = np.array([pred[int(c)][1] == "test" for c in d["source_case_id"]])

    fig, ax = plt.subplots(figsize=(11.0, 5.4))

    for i, t in enumerate(is_test):
        if t:
            ax.axvspan(i - 0.5, i + 0.5, color=GRID, zorder=0)

    ax.plot(x, d["theta_cone_deg"], color=INK_2, linewidth=2.0, zorder=4,
            marker="o", markersize=5, markerfacecolor=SURFACE,
            markeredgecolor=INK_2, markeredgewidth=1.6,
            label="Experimental (reference)")

    for col, label, color, marker in METHODS:
        ax.plot(x, d[col], linestyle="none", marker=marker, markersize=6,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                zorder=3, label=label)

    ax.plot(x, ann, linestyle="none", marker="o", markersize=7,
            color=SERIES["ann"], markeredgecolor=SURFACE, markeredgewidth=1.6,
            zorder=5, label="ANN, this work (M1)")

    _style(ax, title="Fig. 5 — synthesis methods against experiment, 30 real guns",
           xlabel="Case, ordered by experimental theta", ylabel="Half-beam cone angle [deg]")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in d["source_case_id"]], fontsize=7)
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.text(0.005, -0.18, "shaded columns are the paper's 7 test cases; "
            "the other 23 are training points for M1",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)
    _legend(ax, loc="upper left", ncol=2)
    _save(fig, os.path.join(out_dir, "fig5_method_comparison.png"))


def fig6_percentage_error(run, df, out_dir):
    """
    Signed percentage error per case, same ordering and the same colour per
    method as figure 5. Zero is the reference line, so sign and size read
    together -- a method that is consistently low is a different problem from
    one that scatters.
    """
    d = df.sort_values("theta_cone_deg").reset_index(drop=True)
    x = np.arange(len(d))
    truth = d["theta_cone_deg"].to_numpy(dtype=float)
    pred = _all_case_predictions(run, df)
    ann = np.array([pred[int(c)][0] for c in d["source_case_id"]])
    is_test = np.array([pred[int(c)][1] == "test" for c in d["source_case_id"]])

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    for i, t in enumerate(is_test):
        if t:
            ax.axvspan(i - 0.5, i + 0.5, color=GRID, zorder=0)
    ax.axhline(0.0, color=INK_2, linewidth=1.4, zorder=2)

    for col, label, color, marker in METHODS:
        err = 100.0 * (d[col].to_numpy(dtype=float) - truth) / truth
        ax.plot(x, err, linestyle="none", marker=marker, markersize=6,
                color=color, markeredgecolor=SURFACE, markeredgewidth=1.4,
                zorder=3, label=label)

    ax.plot(x, 100.0 * (ann - truth) / truth, linestyle="none", marker="o",
            markersize=7, color=SERIES["ann"], markeredgecolor=SURFACE,
            markeredgewidth=1.6, zorder=5, label="ANN, this work (M1)")

    _style(ax, title="Fig. 6 — signed percentage error in theta, 30 real guns",
           xlabel="Case, ordered by experimental theta", ylabel="Error [%]")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(c)) for c in d["source_case_id"]], fontsize=7)
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.text(0.005, -0.18, "shaded columns are the paper's 7 test cases; "
            "the other 23 are training points for M1",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)
    # Upper right is the only quadrant with no marks in it: the high-theta cases
    # sit at the right and are predicted well, so their errors cluster on zero.
    # The headroom keeps the legend clear of case 12's +85 % outlier.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.30 * (hi - lo))
    _legend(ax, loc="upper right", ncol=2)
    _save(fig, os.path.join(out_dir, "fig6_percentage_error.png"))


def fig7_restart_histogram(out_dir, experiments):
    """
    The distribution of test MRE over 30 restarts, with the paper's 3.74 % marked.

    This figure exists because a single test-MRE number on seven guns is one draw
    from a wide distribution, and quoting the draw instead of the distribution is
    how a reproducibility problem gets hidden. Two models, two panels, a shared
    x-axis so the spreads are directly comparable.

    The paper's value is drawn as a dashed neutral rule rather than a coloured
    series: it is a reference line, not a competitor in the same distribution.
    """
    e5 = (experiments or {}).get("E5_shipped_distributions")
    if not e5:
        print("  skipping fig7: no E5 block (run python -m src.ann.experiments)")
        return

    labels = list(e5)
    fig, axes = plt.subplots(len(labels), 1, figsize=(7.6, 2.5 * len(labels) + 1.2),
                             sharex=True)
    axes = np.atleast_1d(axes)
    allv = np.concatenate([np.array(e5[k]["test_mre_all"]) for k in labels])

    # A single divergent restart would otherwise stretch the axis to 120 % and
    # squash every bar that matters into one column. The axis is clipped and the
    # runs beyond it are named in text instead of being silently dropped.
    x_max = float(np.percentile(allv, 90)) * 1.6
    bins = np.linspace(0, x_max, 22)

    for ax, key, colour in zip(axes, labels, (SERIES["ann"], SERIES["modified"])):
        v = np.array(e5[key]["test_mre_all"], dtype=float)
        ax.hist(np.clip(v, None, bins[-1]), bins=bins, color=colour,
                edgecolor=SURFACE, linewidth=1.2, zorder=3)
        off = v[v > x_max]
        if off.size:
            ax.text(0.985, 0.62,
                    f"+{off.size} restart{'s' if off.size > 1 else ''} beyond this "
                    f"axis, at {', '.join(f'{o:.0f} %' for o in sorted(off))}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=colour)
        med = float(np.median(v))
        ax.axvline(3.74, color=INK_2, linewidth=1.8, linestyle="--", zorder=4)
        ax.axvline(med, color=colour, linewidth=1.8, zorder=4)
        _style(ax, title=f"{key} — {len(v)} restarts", ylabel="restarts")
        top = ax.get_ylim()[1]
        ax.annotate("paper, 3.74 %", xy=(3.74, top * 0.92),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=INK_2, va="top")
        ax.annotate(f"our median {med:.2f} %", xy=(med, top * 0.45),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=9, color=INK_2, va="top")
        ax.text(0.985, 0.92,
                f"median {med:.2f} %   IQR {np.percentile(v, 25):.2f}–"
                f"{np.percentile(v, 75):.2f} %   full range {v.min():.2f}–{v.max():.2f} %",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color=INK_MUTED)
        ax.set_xlim(0, x_max)

    axes[-1].set_xlabel("Test MRE on θ across the paper's 7 test guns [%]",
                        color=INK_2, fontsize=10)
    fig.suptitle("Fig. 7 — the result is a distribution, not a number",
                 color=INK, fontsize=12, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, os.path.join(out_dir, "fig7_restart_histogram.png"))


def main():
    root = ds.repo_root()
    out_dir = os.path.join(root, "reports", "figures")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(root, "reports", "m1_run.json"), encoding="utf-8") as f:
        run = json.load(f)
    exp_path = os.path.join(root, "reports", "experiments.json")
    experiments = None
    if os.path.exists(exp_path):
        with open(exp_path, encoding="utf-8") as f:
            experiments = json.load(f)
    df = pd.read_csv(ds.default_path())
    df = df[df["beam_type"] == "pencil"].reset_index(drop=True)

    print("figures ->", out_dir)
    fig3_regression(run, out_dir)
    fig4_mse_vs_epochs(run, out_dir)
    fig5_method_comparison(run, df, out_dir)
    fig6_percentage_error(run, df, out_dir)
    fig7_restart_histogram(out_dir, experiments)


if __name__ == "__main__":
    main()
