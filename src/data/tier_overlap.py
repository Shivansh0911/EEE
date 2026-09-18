"""
The reality gap: does the synthetic cloud actually cover the real guns?

    python -m src.data.tier_overlap

Generating a thousand physics rows is worthless if they sit somewhere the real
guns are not. Two things have to be checked, and both are reported as numbers
rather than as a picture to squint at:

1.  **Containment.** Every real gun inside the validated envelope should lie
    within the synthetic cloud, not merely near it. Checked twice — against the
    axis-aligned bounding box, and against the convex hull in normalised
    (P, r_w, ln C) space, which is the stricter test. A point can sit inside the
    box and outside the hull.

2.  **The excluded corner.** The seven real guns with C < 8 are outside the
    envelope by construction, and they are plotted and counted as EXCLUDED
    rather than quietly dropped. A coverage figure that showed only the covered
    region would be advertising, not evidence.

Distances are computed in normalised coordinates — each axis scaled to its own
Tier B range, with C in log space — because the raw units are not comparable:
a millimetre of r_w and a unit of C are not the same size of disagreement.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)
from scipy.spatial import Delaunay

from ..ann import dataset as ds
from ..ann.figures import GRID, INK, INK_2, INK_MUTED, SURFACE, _save, _style

C_MIN = 8.0

# Validated all-pairs for colour-vision deficiency (see src/ann/figures.py).
COL_SYNTH = "#2a78d6"    # blue
COL_REAL = "#eb6834"     # orange
COL_EXCL = "#4a3aa7"     # violet

AXES = [
    ("perveance_uperv", "P  [µperv]", False),
    ("rw_mm", "r_w  [mm]", False),
    ("C", "C   (log axis)", True),
]


def normalise(X, lo, hi):
    """Each axis onto [0, 1] against the Tier B range; C already logged."""
    return (X - lo) / (hi - lo)


def to_space(df):
    """(P, rw, ln C) as an array."""
    return np.column_stack([
        df["perveance_uperv"].to_numpy(float),
        df["rw_mm"].to_numpy(float),
        np.log(df["C"].to_numpy(float)),
    ])


def coverage(real, synth):
    """
    Containment statistics for the real points against the synthetic cloud.

    Returns per-point flags plus summary numbers. The convex hull is computed
    with Delaunay rather than ConvexHull because membership testing is what is
    wanted, and `Delaunay.find_simplex` answers exactly that.
    """
    lo, hi = synth.min(axis=0), synth.max(axis=0)
    rn = normalise(real, lo, hi)
    sn = normalise(synth, lo, hi)

    in_box = np.all((rn >= 0.0) & (rn <= 1.0), axis=1)

    tri = Delaunay(sn)
    in_hull = tri.find_simplex(rn) >= 0

    # Distance from each real point to its nearest synthetic neighbour.
    d_real = np.sqrt(((rn[:, None, :] - sn[None, :, :]) ** 2).sum(-1)).min(axis=1)

    # Sixteen of the real points ARE anchors, reproduced exactly in Tier B, so
    # their nearest-neighbour distance is identically zero by construction.
    # Including them would make the median meaningless -- the informative number
    # is how far the NON-anchored real guns sit from the cloud, and those are
    # precisely the held-out test guns the comparison turns on.
    anchored = d_real < 1e-12
    free = ~anchored

    # The synthetic cloud's own spacing, for scale: a real point further from the
    # cloud than the cloud is from itself is sitting in a hole. Indices are
    # tracked explicitly so a point is not compared against itself.
    idx = np.arange(0, len(sn), max(1, len(sn) // 300))
    dd = np.sqrt(((sn[idx][:, None, :] - sn[None, :, :]) ** 2).sum(-1))
    dd[np.arange(len(idx)), idx] = np.inf      # mask each sample against itself
    d_synth = dd.min(axis=1)

    return {
        "in_box": in_box,
        "in_hull": in_hull,
        "nn_distance": d_real,
        "anchored": anchored,
        "summary": {
            "n_real_in_envelope": int(len(real)),
            "n_anchored_exactly": int(anchored.sum()),
            "inside_bounding_box": int(in_box.sum()),
            "inside_convex_hull": int(in_hull.sum()),
            "box_coverage_pct": float(100 * in_box.mean()),
            "hull_coverage_pct": float(100 * in_hull.mean()),
            "max_nn_distance_normalised": float(d_real.max()),
            "median_nn_distance_unanchored": (
                float(np.median(d_real[free])) if free.any() else None),
            "max_nn_distance_unanchored": (
                float(d_real[free].max()) if free.any() else None),
            "median_synthetic_spacing_normalised": float(np.median(d_synth)),
        },
    }


def main():
    root = ds.repo_root()
    master = pd.read_csv(os.path.join(root, "data", "processed", "dataset_master.csv"))
    tb_path = os.path.join(root, "data", "processed", "dataset_tier_b.csv")
    if not os.path.exists(tb_path):
        raise SystemExit("Tier B not generated; run python -m src.data.synth_generator")
    synth_df = pd.read_csv(tb_path)

    real_in = master[master["C"] >= C_MIN]
    real_out = master[master["C"] < C_MIN]

    synth = to_space(synth_df)
    cov = coverage(to_space(real_in), synth)
    s = cov["summary"]

    outside = real_in.loc[~cov["in_hull"]]
    outside_box = real_in.loc[~cov["in_box"]]

    print("Tier overlap")
    print(f"  Tier B rows                 {len(synth_df)}")
    print(f"  Tier A rows, C >= {C_MIN:g}        {len(real_in)}")
    print(f"  Tier A rows, C <  {C_MIN:g}  EXCLUDED  {len(real_out)}  "
          f"(cases {sorted(real_out['source_case_id'])})")
    print(f"  inside synthetic bounding box  {s['inside_bounding_box']}/"
          f"{s['n_real_in_envelope']}  ({s['box_coverage_pct']:.1f} %)")
    print(f"  inside synthetic convex hull   {s['inside_convex_hull']}/"
          f"{s['n_real_in_envelope']}  ({s['hull_coverage_pct']:.1f} %)")
    print(f"  of those, reproduced exactly as anchors  {s['n_anchored_exactly']}")
    print(f"  nearest-neighbour distance, un-anchored reals: median "
          f"{s['median_nn_distance_unanchored']:.4f}  max "
          f"{s['max_nn_distance_unanchored']:.4f}")
    print(f"  synthetic cloud's own spacing: median "
          f"{s['median_synthetic_spacing_normalised']:.4f}")

    # The converse of containment, which the figure makes obvious and a
    # containment number hides: the real guns are not spread through the box,
    # they sit on a band. Reported because it bounds what Tier B can do.
    def corr(d):
        return float(np.corrcoef(np.log(d["C"].to_numpy(float)),
                                 d["rw_mm"].to_numpy(float))[0, 1])
    r_real, r_synth = corr(real_in), corr(synth_df)
    print(f"  structure  R(ln C, r_w): real {r_real:+.3f}   synthetic {r_synth:+.3f}")

    if len(outside):
        print("\n  REAL POINTS OUTSIDE THE SYNTHETIC HULL:")
        for _, r in outside.iterrows():
            print(f"    case {int(r['source_case_id']):>3}  P={r['perveance_uperv']:.2f} "
                  f"rw={r['rw_mm']:.2f} C={r['C']:.2f}  split={r['paper_split']}")
    else:
        print("\n  every real gun in the envelope lies inside the synthetic hull")

    # ---------------------------------------------------------------- figure
    fig = plt.figure(figsize=(12.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], hspace=0.38, wspace=0.30)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for k, (i, j) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, k])
        xi, xl, xlog = AXES[i]
        yi, yl, ylog = AXES[j]

        ax.scatter(synth_df[xi], synth_df[yi], s=7, c=COL_SYNTH, alpha=0.28,
                   linewidths=0, zorder=2, label=f"Tier B, synthetic ({len(synth_df)})")
        ax.scatter(real_in[xi], real_in[yi], s=72, marker="o", facecolor="none",
                   edgecolor=COL_REAL, linewidths=2.0, zorder=4,
                   label=f"Tier A, real, in envelope ({len(real_in)})")
        ax.scatter(real_out[xi], real_out[yi], s=90, marker="X", c=COL_EXCL,
                   edgecolor=SURFACE, linewidths=1.0, zorder=5,
                   label=f"Tier A, real, C < 8 — EXCLUDED ({len(real_out)})")

        if xlog:
            ax.set_xscale("log")
        if ylog:
            ax.set_yscale("log")
        _style(ax, xlabel=xl, ylabel=yl)
        if k == 0:
            ax.set_title("Coverage in each projection", color=INK, fontsize=11,
                         loc="left", pad=10)

    # 3D view
    ax3 = fig.add_subplot(gs[1, 0:2], projection="3d")
    ax3.scatter(synth_df["perveance_uperv"], synth_df["rw_mm"],
                np.log10(synth_df["C"]), s=5, c=COL_SYNTH, alpha=0.18, linewidths=0)
    ax3.scatter(real_in["perveance_uperv"], real_in["rw_mm"], np.log10(real_in["C"]),
                s=60, marker="o", facecolor="none", edgecolor=COL_REAL, linewidths=1.8)
    ax3.scatter(real_out["perveance_uperv"], real_out["rw_mm"], np.log10(real_out["C"]),
                s=80, marker="X", c=COL_EXCL, edgecolor=SURFACE, linewidths=0.8)
    ax3.set_xlabel("P [µperv]", color=INK_2, fontsize=9)
    ax3.set_ylabel("r_w [mm]", color=INK_2, fontsize=9)
    ax3.set_zlabel("log₁₀ C", color=INK_2, fontsize=9)
    ax3.tick_params(colors=INK_2, labelsize=8)
    ax3.set_facecolor(SURFACE)
    ax3.view_init(elev=20, azim=-58)
    ax3.set_title("The same cloud in three dimensions", color=INK, fontsize=11,
                  loc="left", pad=0)

    # the numbers, as a panel rather than a caption
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    lines = [
        ("Tier B synthetic rows", f"{len(synth_df)}"),
        ("Tier A real, C ≥ 8", f"{len(real_in)}"),
        ("Tier A real, C < 8 (excluded)", f"{len(real_out)}"),
        ("", ""),
        ("Real inside synthetic box", f"{s['inside_bounding_box']}/{s['n_real_in_envelope']}"
                                      f"  ({s['box_coverage_pct']:.0f} %)"),
        ("Real inside convex hull", f"{s['inside_convex_hull']}/{s['n_real_in_envelope']}"
                                    f"  ({s['hull_coverage_pct']:.0f} %)"),
        ("", ""),
        ("Of those, exact anchors", f"{s['n_anchored_exactly']}"),
        ("", ""),
        ("Nearest-neighbour distance", ""),
        ("   median, un-anchored reals", f"{s['median_nn_distance_unanchored']:.3f}"),
        ("   worst, un-anchored reals", f"{s['max_nn_distance_unanchored']:.3f}"),
        ("   synthetic cloud spacing", f"{s['median_synthetic_spacing_normalised']:.3f}"),
    ]
    y = 0.98
    ax4.text(0, y, "Coverage", color=INK, fontsize=11, va="top", weight="bold")
    y -= 0.085
    for label, value in lines:
        if label:
            ax4.text(0, y, label, color=INK_2, fontsize=9, va="top")
            ax4.text(1.0, y, value, color=INK, fontsize=9, va="top", ha="right",
                     family="monospace")
        y -= 0.072
    ax4.text(0, y - 0.02,
             "Distances are normalised: each axis\nscaled to its Tier B range, C in log\n"
             "space. A real point further from the\ncloud than the cloud's own spacing\n"
             "would be sitting in a hole.",
             color=INK_MUTED, fontsize=8, va="top", linespacing=1.6)

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=9,
               labelcolor=INK_2, bbox_to_anchor=(0.995, 0.995))
    fig.suptitle("Tier A against Tier B — does the synthetic cloud cover the real guns?",
                 color=INK, fontsize=13, x=0.012, ha="left", y=0.995)

    out_dir = os.path.join(root, "reports", "figures")
    os.makedirs(out_dir, exist_ok=True)
    _save(fig, os.path.join(out_dir, "tier_overlap.png"))

    payload = dict(s)
    payload["corr_lnC_rw_real"] = r_real
    payload["corr_lnC_rw_synthetic"] = r_synth
    payload["excluded_cases"] = sorted(int(c) for c in real_out["source_case_id"])
    payload["outside_hull_cases"] = sorted(int(c) for c in outside["source_case_id"])
    payload["outside_box_cases"] = sorted(int(c) for c in outside_box["source_case_id"])
    payload["n_synthetic"] = int(len(synth_df))
    payload["C_min"] = C_MIN
    with open(os.path.join(root, "reports", "tier_overlap.json"), "w",
              encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  wrote {os.path.join(root, 'reports', 'tier_overlap.json')}")


if __name__ == "__main__":
    main()
