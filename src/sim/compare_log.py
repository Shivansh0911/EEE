"""
Predicted current and perveance against target.exe's own result, read from
pierce.log at the LAST MAJOR ITERATION (sir's instruction).

    python -m src.sim.compare_log              # sir's deck, our 8 guns, the 5 tuned guns
    python -m src.sim.compare_log --show sir_gun   # also print that run's iteration table

pierce.log has an iteration table:

     MAJOR  MINOR    POTENTIAL  OMEGA  CONV.    CURRENT (A)     MICROPERV.   DEL  BETA
     ITER.  ITER.    ERROR
        ...
       20  1091     0.609E-06 1.375    1  1     0.1236E+03    0.2128E+01  0.20  0.60

The last row is target.exe's converged answer: the cathode current (A) and the
microperveance (I / V^1.5 x 1e6). It is compared with what our code predicted
for the gun it built:

  predicted perveance  = the design input P (our gun is synthesised for it)
  predicted current    = P x 1e-6 x V^1.5
  V                    = the anode potential, read from the same log

Two checks on the log itself: the last row's CONV. flag (1 = converged), and
that its perveance agrees with its current (P = I / V^1.5), and the "Cathode
Current" in the log's current balance, which is the same number unrounded.

Reads the logs saved in the repository (reports/simulation/...), so it runs
without target.exe. Writes reports/simulation/compare_log.{md,json}.
"""

import argparse
import json
import os
import re

from ..ann import dataset as ds
from . import lktron as lk

ROOT = ds.repo_root()
OURGUN = os.path.join(ROOT, "reports", "simulation", "ourgun")
REF_LOG = os.path.join(ROOT, "simulation", "lktron", "reference_ceeri", "pierce.log")

HEAD = re.compile(r"MAJOR\s+MINOR\s+POTENTIAL")
ROW = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.E+-]+)\s+([\d.]+)\s+(-?1)\s+(-?1)\s+"
                 r"([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def last_major_iteration(path):
    """The last row of pierce.log's iteration table, as printed and parsed."""
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()
    rows = [(i, ROW.match(l)) for i, l in enumerate(lines)]
    rows = [(i, m) for i, m in rows if m]
    if not rows:
        return None
    i, m = rows[-1]
    g = m.groups()
    return {"line_no": i + 1, "line": lines[i].rstrip(), "iteration": int(g[0]),
            "converged": g[5] == "1", "current_A": float(g[6]), "microperveance": float(g[7]),
            "table": [lines[j].rstrip() for j in range(len(lines))
                      if HEAD.search(lines[j]) or ROW.match(lines[j]) or "ITERATION NO." in lines[j]]}


def compare(name, log_path, P_pred, label, note=""):
    it = last_major_iteration(log_path)
    full = lk.parse_log(log_path)
    V = max(e["potential_V"] for e in full["electrodes"])
    I_pred = P_pred * 1e-6 * V ** 1.5
    return {"name": name, "label": label, "log": os.path.relpath(log_path, ROOT), "V": V,
            "iteration": it["iteration"], "converged": it["converged"], "line": it["line"],
            "line_no": it["line_no"],
            "pred_microperveance": P_pred, "log_microperveance": it["microperveance"],
            "perveance_err_pct": 100 * (it["microperveance"] / P_pred - 1),
            "pred_current_A": I_pred, "log_current_A": it["current_A"],
            "current_err_pct": 100 * (it["current_A"] / I_pred - 1),
            "check_P_from_I": it["current_A"] / V ** 1.5 * 1e6,
            "cathode_current_balance_A": full["cathode_current_A"],
            "note": note}


def runs():
    """Sir's reference deck, our gun on the 8 guns, and the 5 anode-tuned guns."""
    out = [("sir_deck", REF_LOG, 2.0656, "Sir's reference deck (CEERI, 120 A design)",
            "sir's own electrodes")]
    with open(os.path.join(OURGUN, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    for g in s["guns"]:
        if g.get("ok"):
            lost = g["lost_in_gun"]
            out.append((g["case"], os.path.join(OURGUN, g["case"], "pierce.log"), g["P"],
                        f"Our gun, {g['case'].replace('_', ' ')}",
                        f"{lost} electrons hit the gun" if lost else ""))
    for t in s.get("tuned", []):
        P = next(g["P"] for g in s["guns"] if g["case"] == t["case"])
        out.append((t["variant"], os.path.join(OURGUN, t["variant"], "pierce.log"), P,
                    f"Our gun, {t['case'].replace('_', ' ')}, anode tuned {t['anode_shift_mm']:+.3f} mm",
                    "tuned with target.exe in the loop"))
    return out


def fmt_I(a):
    return f"{a:9.3f} A " if a >= 1 else f"{a * 1e3:9.3f} mA"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--show", help="print this run's iteration table from pierce.log")
    a = p.parse_args(argv)
    res = [compare(*r) for r in runs()]
    if a.show:
        r = next(x for x in res if x["name"] == a.show)
        it = last_major_iteration(os.path.join(ROOT, r["log"]))
        print(f"{r['log']}: iteration table")
        for ln in it["table"]:
            print(ln)
        print(f"-> last major iteration: line {r['line_no']} (iteration {r['iteration']})\n")
    print("Predicted vs target.exe (pierce.log, last major iteration)")
    print(f"{'run':24s} {'it':>3s} conv  {'I predicted':>12s} {'I in log':>12s} {'err':>7s}   "
          f"{'P pred':>7s} {'P in log':>8s} {'err':>7s}  P=I/V^1.5")
    for r in res:
        print(f"{r['name'][:24]:24s} {r['iteration']:3d}  {'yes' if r['converged'] else 'NO '} "
              f"{fmt_I(r['pred_current_A'])} {fmt_I(r['log_current_A'])} {r['current_err_pct']:+6.1f}%   "
              f"{r['pred_microperveance']:7.3f} {r['log_microperveance']:8.3f} {r['perveance_err_pct']:+6.1f}%  "
              f"{r['check_P_from_I']:.3f}  {r['note']}")
    tuned = [r for r in res if "_tune_" in r["name"]]
    if tuned:
        print(f"\nanode-tuned guns: perveance within "
              f"{max(abs(r['perveance_err_pct']) for r in tuned):.1f} % of the prediction")
    L = ["# Predicted current and perveance vs target.exe (pierce.log, last major iteration)", "",
         "Generated by `python -m src.sim.compare_log`. Predicted perveance = the design input P; "
         "predicted current = P x 1e-6 x V^1.5. Log values: the last row of pierce.log's MAJOR "
         "ITER. table (CURRENT (A), MICROPERV.).", "",
         "| Run | V (kV) | Last major iter. | Converged | I predicted | I in pierce.log | Error | "
         "P predicted | P in pierce.log | Error | Note |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res:
        L.append(f"| {r['label']} | {r['V'] / 1e3:g} | {r['iteration']} | "
                 f"{'yes' if r['converged'] else 'no'} | {fmt_I(r['pred_current_A']).strip()} | "
                 f"{fmt_I(r['log_current_A']).strip()} | {r['current_err_pct']:+.1f} % | "
                 f"{r['pred_microperveance']:.3f} | {r['log_microperveance']:.3f} | "
                 f"{r['perveance_err_pct']:+.1f} % | {r['note']} |")
    out = os.path.join(ROOT, "reports", "simulation", "compare_log")
    with open(out + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    figure(res)
    print("\nwrote reports/simulation/compare_log.md and .json, reports/figures/compare_log.png")
    return res


def figure(res):
    """Perveance error (pierce.log vs predicted) per gun: as built, and with the anode tuned."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    guns = [r for r in res if r["name"] != "sir_deck" and "_tune_" not in r["name"]]
    tuned = {r["name"].split("_tune_")[0]: r for r in res if "_tune_" in r["name"]}
    names = [g["name"] for g in guns]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = 0.38
    b1 = ax.bar([i - w / 2 for i in x], [g["perveance_err_pct"] for g in guns], w,
                color="#2a6fdb", label="as built")
    tv = [tuned[n]["perveance_err_pct"] if n in tuned else None for n in names]
    b2 = ax.bar([i + w / 2 for i, v in zip(x, tv) if v is not None], [v for v in tv if v is not None],
                w, color="#d9822b", label="anode tuned with target.exe in the loop")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax.annotate(f"{v:+.1f}%", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=8, xytext=(0, 2 if v >= 0 else -2),
                        textcoords="offset points")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(list(x), [n.replace("sir_gun", "sir's gun") + (" *" if not g["converged"] else "")
                            for n, g in zip(names, guns)])
    ax.set_ylabel("perveance error (%)\npierce.log vs predicted")
    ax.set_title("Predicted perveance vs target.exe (pierce.log, last major iteration)   "
                 "* = not converged, beam hits the gun", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = os.path.join(ROOT, "reports", "figures", "compare_log.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
