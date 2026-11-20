"""
Figures and the results table for the PIERCE simulations.

    python -m src.sim.report

Reads reports/simulation/*/summary.json and plot.json (written by
src.sim.experiments), writes reports/figures/sim_*.png, the Tiwary-Basu Fig. 2
reproduction, and reports/SIMULATION_RESULTS.md. Every number in the markdown is
read from those JSON files, none is typed in.
"""

import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..ann import dataset as ds
from ..physics import bfe

ROOT = ds.repo_root()
SIM = os.path.join(ROOT, "reports", "simulation")
FIG = os.path.join(ROOT, "reports", "figures")
COL = {"boundary": "#000000", "trajectories": "#d62728", "equipotentials": "#2ca02c"}

SHOW = [("reference", "Sir's reference deck"),
        ("vedant", "Vedant's deck (67.5 deg, corrected cathode)"),
        ("tiwary_basu", "Tiwary-Basu electrodes (Paper 1)"),
        ("TUNED_PSI", "Electrode angle tuned to the design current"),
        ("TUNED_2D", "Angle + anode position tuned (current and waist)")]


def load(name):
    with open(os.path.join(SIM, name, "summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    with open(os.path.join(SIM, name, "plot.json"), encoding="utf-8") as f:
        p = json.load(f)
    return s, p


def tuned_names():
    with open(os.path.join(SIM, "tuned_psi.json"), encoding="utf-8") as f:
        a = json.load(f)["variant"]
    with open(os.path.join(SIM, "tuned_2d.json"), encoding="utf-8") as f:
        b = json.load(f)["best"]["variant"]
    return {"TUNED_PSI": a, "TUNED_2D": b}


def plot_variant(ax, plot, title, c):
    for key in ("equipotentials", "trajectories", "boundary"):
        for seg in plot[key]:
            a = np.array(seg)
            if len(a) > 1:
                ax.plot(a[:, 0], a[:, 1], color=COL[key],
                        lw=1.6 if key == "boundary" else 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nI = {c['sim_current_A']:.1f} A ({c['current_err_pct']:+.1f} %), "
                 f"lost in gun {c['lost_in_gun']}/{c['n_trajectories']} traj., "
                 f"waist {c['sim_waist_mm_refined']:.2f} mm",
                 fontsize=9)
    ax.set_xlabel("z, mm")
    ax.set_ylabel("r, mm")


def figures():
    os.makedirs(FIG, exist_ok=True)
    names = tuned_names()
    fig, axes = plt.subplots(len(SHOW), 1, figsize=(9, 3.4 * len(SHOW)))
    for ax, (key, title) in zip(axes, SHOW):
        s, p = load(names.get(key, key))
        plot_variant(ax, p, title, s["compare"])
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sim_trajectories.png"), dpi=120)
    plt.close(fig)

    # focus-electrode angle sweep (Paper 2's study, on our gun, in sir's code)
    pts = []
    for d in glob.glob(os.path.join(SIM, "*", "summary.json")):
        s = json.load(open(d, encoding="utf-8"))
        n = s["name"]
        if s.get("ok") and (n.startswith("psi_") or n == "vedant" or n.startswith("tuned_psi_")):
            c = s["compare"]
            pts.append((s["notes"]["psi_deg"], c["sim_current_A"], c["sim_waist_mm_refined"],
                        c["lost_in_gun"]))
    pts = np.array(sorted(set(pts)))
    fig, a1 = plt.subplots(figsize=(7, 4))
    a1.plot(pts[:, 0], pts[:, 1], "o-", color="#1f77b4", label="cathode current (PIERCE)")
    a1.axhline(120, color="#1f77b4", ls="--", lw=0.8, label="design current 120 A")
    a1.set_xlabel("focus electrode angle to beam edge, deg")
    a1.set_ylabel("current, A")
    a2 = a1.twinx()
    a2.plot(pts[:, 0], pts[:, 2], "s-", color="#d62728", label="beam waist (PIERCE)")
    a2.axhline(9.8, color="#d62728", ls="--", lw=0.8, label="design waist 9.8 mm")
    a2.set_ylabel("waist radius, mm")
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    a1.set_title("Focus-electrode angle sweep, CEERI gun, Vedant's construction", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "sim_psi_sweep.png"), dpi=120)
    plt.close(fig)

    # Tiwary & Basu Fig. 2, reproduced
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, ratio, rc, th, ls in [("Litton L-2087", 1.985, 13.665, 13.44, "-x"),
                                    ("Varian VTC 6364", 2.389, 10.158, 15.59, "-."),
                                    ("Frost and Purl", 2.393, 17.567, 27.17, "--")]:
        t0, Ra = math.radians(th), rc / ratio
        for s in (bfe.bfe_s(t_max=40, n=4000), bfe.anode_s(math.log(ratio), tau_max=80, n=2000)):
            rho, t = rc * np.exp(s.real), t0 + s.imag
            k = t < math.radians(100)
            ax.plot((rho * np.cos(t))[k], (rho * np.sin(t))[k], ls, color="k", ms=3,
                    markevery=60, lw=1)
        for R in (rc, Ra):
            tt = np.linspace(0, t0, 30)
            ax.plot(R * np.cos(tt), R * np.sin(tt), "k", lw=0.8)
        ax.plot([0, rc * math.cos(t0)], [0, rc * math.sin(t0)], "k", lw=0.6, label=None)
        ax.plot([], [], ls, color="k", label=f"{name} (rc/ra {ratio}, rc {rc} mm, theta0 {th} deg)")
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlabel("radial distance from the centre of curvature, mm")
    ax.set_title("Tiwary & Basu (1988) Fig. 2, reproduced by src/physics/bfe.py", fontsize=10)
    fig.savefig(os.path.join(FIG, "tiwary_basu_fig2.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


def table():
    with open(os.path.join(SIM, "summary.json"), encoding="utf-8") as f:
        summ = json.load(f)
    names = tuned_names()
    order = ["reference", "vedant", "vedant_raw", "tiwary_basu", "psi_57.5", "psi_62.5",
             "psi_72.5", names["TUNED_PSI"], names["TUNED_2D"]]
    d = summ["design"]
    L = ["# PIERCE simulation results (generated by `python -m src.sim.report`)", "",
         f"Gun: {d['V']/1e3:g} kV, {d['I']:g} A, waist {d['rw_mm']} mm, {d['Jc']} A/cm^2 "
         f"(CEERI 8 MW klystron; sir's reference deck is this gun). Design (Vaughan/Vedant): "
         f"microperveance {d['muP']:.4f}, throw {d['zw_mm']:.2f} mm. Grid H = "
         f"{summ['grid_mm']} mm.", "",
         "Trajectories that stop before the waist hit the gun's own electrodes ('lost in gun'); "
         "those that stop after it are the expanding beam scraping the drift tube, which a "
         "magnet prevents in a real device, so they are not counted against the gun. A waist "
         "on the end wall is not reported (the beam is still converging there).", "",
         "| Variant | Current A | vs design | Lost in gun | Throw mm | vs design | "
         "Waist mm | vs design | Run s |", "|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        p = os.path.join(SIM, n, "summary.json")
        if not os.path.exists(p):
            continue
        s = json.load(open(p, encoding="utf-8"))
        if not s.get("ok"):
            L.append(f"| {n} | failed: {s['error']} |||||||")
            continue
        c = s["compare"]
        f = lambda v, fmt: "no waist" if v is None or (isinstance(v, float) and math.isnan(v)) \
            else format(v, fmt)
        L.append(f"| {n} | {c['sim_current_A']:.2f} | {c['current_err_pct']:+.1f} % | "
                 f"{c['lost_in_gun']}/{c['n_trajectories']} | {f(c['sim_throw_mm_refined'], '.1f')} | "
                 f"{f(c['throw_err_pct'], '+.1f')} % | {f(c['sim_waist_mm_refined'], '.2f')} | "
                 f"{f(c['waist_err_pct'], '+.1f')} % | {c['wall_seconds']:.0f} |")
    with open(os.path.join(ROOT, "reports", "SIMULATION_RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


WEB_VARIANTS = [("reference", "Sir's reference deck"),
                ("vedant", "Vedant's deck (ψ 67.5°, corrected cathode)"),
                ("vedant_raw", "Vedant's construction, uncorrected cathode"),
                ("tiwary_basu", "Tiwary & Basu (1988) electrodes"),
                ("psi_57.5", "ψ 57.5°"), ("psi_62.5", "ψ 62.5°"), ("psi_72.5", "ψ 72.5°"),
                ("TUNED_PSI", "ψ tuned to the design current"),
                ("TUNED_2D", "ψ + anode position tuned (current and waist)")]


def export_web():
    """web/data/simulation.js: results and thinned plots for the Simulation tab."""
    names = tuned_names()
    with open(os.path.join(SIM, "summary.json"), encoding="utf-8") as f:
        summ = json.load(f)
    rnd = lambda seg, k: [[round(z, 2), round(r, 2)] for z, r in seg[::k]] + [
        [round(seg[-1][0], 2), round(seg[-1][1], 2)]]
    out = []
    for key, label in WEB_VARIANTS:
        n = names.get(key, key)
        s, p = load(n)
        c = s["compare"]
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in c.items()}
        out.append({"key": n, "label": label, "notes": {k: v for k, v in s["notes"].items()
                                                        if k not in ("bfe", "anode")},
                    "compare": clean,
                    "plot": {"boundary": [rnd(x, 1) for x in p["boundary"]],
                             "trajectories": [rnd(x, 2) for x in p["trajectories"]],
                             "equipotentials": [rnd(x, 2) for x in p["equipotentials"]]}})
    sweep = sorted([(o["notes"]["psi_deg"], o["compare"]["sim_current_A"],
                     o["compare"]["sim_waist_mm_refined"]) for o in out
                    if "psi_deg" in o["notes"] and o["notes"].get("anode_shift_mm", 0) == 0
                    and o["notes"].get("cathode", "").startswith("RcT")])
    with open(os.path.join(SIM, "tuned_2d.json"), encoding="utf-8") as f:
        t2 = json.load(f)
    data = {"design": summ["design"], "grid_mm": summ["grid_mm"], "variants": out,
            "sweep": sweep, "tuned_2d": {"psi_deg": t2["best"]["psi_deg"],
                                         "anode_shift_mm": t2["best"]["anode_shift_mm"],
                                         "converged": t2["converged"]}}
    path = os.path.join(ROOT, "web", "data", "simulation.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("/* GENERATED by src/sim/report.py -- do not edit. */\n"
                "window.EGUN_DATA = window.EGUN_DATA || {};\n"
                "window.EGUN_DATA.simulation = " + json.dumps(data, separators=(",", ":")) + ";\n")
    return path


def main():
    figures()
    table()
    print(export_web())
    print("wrote reports/figures/sim_trajectories.png, sim_psi_sweep.png, "
          "tiwary_basu_fig2.png, reports/SIMULATION_RESULTS.md")


if __name__ == "__main__":
    main()
