"""
What happens to our gun in target.exe when one design quantity is changed.

    python -m src.sim.sweeps                 # both sweeps, sir's reference gun
    python -m src.sim.sweeps --gun case15    # one of the paper's held-out guns

1. Anode shift.  The anode (face, aperture, drift tube) is moved along the axis,
   everything else kept. The cathode-anode spacing d changes, so the current the
   gun draws changes. Compared with two predictions:
     * the planar Child-law scaling P ~ 1/d^2 from the unshifted gun;
     * Langmuir-Blodgett for the spherical diode: P ~ 1/alpha^2(Rc/Ra'), with the
       anode sphere's radius Ra' = Ra - shift about the same centre (Vaughan's
       Eq. 5 series, src/physics/lb.py).
   Also recorded: where the beam focuses (throw), its waist, and whether any
   electron hits the gun (lost before the waist, and where).

2. Cathode radius of curvature.  The cathode disc radius r_c and the perveance are
   kept; the beam angle theta_T is changed, so R_cT = r_c / sin(theta) changes.
   The anode is re-placed by the Langmuir-Blodgett equation for the same
   perveance (Vaughan's synthesis), and the Tiwary-Basu focus electrode follows.
   Recorded: the cathode's aim point (its centre of curvature, where electrons
   would meet with no space charge), the actual focus point in target.exe, the
   waist, the current, and electrons hitting the gun.

Results: reports/simulation/sweeps/<gun>.json (+ each run's deck, log and plot
data), figures in reports/figures/, and web/data/sweeps.js for the website.
"""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

from ..ann import dataset as ds
from ..physics import lb
from . import ourgun as og

OUT = os.path.join(ds.repo_root(), "reports", "simulation", "sweeps")
RUNS = os.path.join(ds.repo_root(), "simulation", "runs", "sweeps")
ANODE_SHIFTS = (-4, -3, -2, -1, 0, 1, 2, 3, 4, 6)          # mm
THETA_STEPS = (-8, -6, -4, -2, 0, 2, 4, 6, 8)               # deg added to our network's angle


def gun(name):
    for g in og.guns():
        if g[0] == name:
            return g
    raise KeyError(name)


def _row(res):
    if not res["ok"]:
        return {"ok": False, "error": res["error"][:200]}
    i, c = res["info"], res["compare"]
    clean = lambda v: None if isinstance(v, float) and math.isnan(v) else v
    return {"ok": True, "run": res["name"], "anode_shift_mm": i["anode_shift_mm"],
            "cathode_anode_mm": i["cathode_anode_mm"], "thetaT_deg": i["thetaT_built_deg"],
            "RcT_mm": i["RcT_mm"], "Rc_mm": i["Rc_mm"], "Ra_mm": i["Ra_mm"], "rc_mm": i["rc_mm"],
            "aim_point_mm": i["RcT_mm"],
            "sim_microperveance": c["sim_microperveance"], "sim_current_A": c["sim_current_A"],
            "perveance_err_pct": c["perveance_err_pct"],
            "throw_mm": clean(c["sim_throw_mm"]), "waist_mm": clean(c["sim_waist_mm"]),
            "lost_in_gun": c["lost_in_gun"], "n_trajectories": c["n_trajectories"],
            "lost_in_gun_at_mm": c["lost_in_gun_at_mm"]}


def run_all(name, workers=4):
    g = gun(name)
    _, P, rw, C, _, V = g
    theta0 = og.predict_theta(P, rw, C)
    runs, out = os.path.join(RUNS, name), os.path.join(OUT, name)
    jobs = [("anode", s, f"anode_{s:+g}", dict(anode_shift=float(s))) for s in ANODE_SHIFTS]
    jobs += [("rc", t, f"theta_{t:+g}", dict(theta=theta0 + t)) for t in THETA_STEPS]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(lambda j: og.simulate(j[2], P, rw, C, V, runs=runs, out=out, **j[3]),
                            jobs))
    anode = [dict(_row(r), step=j[1]) for j, r in zip(jobs, res) if j[0] == "anode"]
    rc = [dict(_row(r), step=j[1]) for j, r in zip(jobs, res) if j[0] == "rc"]
    # predictions for the anode sweep, from the unshifted gun
    base = next(r for r in anode if r["ok"] and r["anode_shift_mm"] == 0)
    d0, P0 = base["cathode_anode_mm"], base["sim_microperveance"]
    a0 = lb.a_from_y(math.log(base["Rc_mm"] / base["Ra_mm"]))
    for r in anode:
        if not r["ok"]:
            continue
        r["P_child_1_over_d2"] = P0 * (d0 / r["cathode_anode_mm"]) ** 2
        a = lb.a_from_y(math.log(base["Rc_mm"] / (base["Ra_mm"] - r["anode_shift_mm"])))
        r["P_langmuir_blodgett"] = P0 * (a0 / a) ** 2
    data = {"gun": name, "design": {"P": P, "rw_mm": rw, "C": C, "V": V},
            "theta_network_deg": theta0, "anode": anode, "rc": rc}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=float)
    return data


def report(data):
    d = data["design"]
    print(f"gun {data['gun']}: P {d['P']:g}, waist {d['rw_mm']:g} mm, C {d['C']:g}, "
          f"{d['V'] / 1e3:g} kV; our network's angle {data['theta_network_deg']:.2f} deg")
    print("\n1) anode shift (target.exe), P = microperveance")
    print(" shift   d mm   P sim  P 1/d^2  P L-B   throw mm  waist mm  lost in gun")
    for r in data["anode"]:
        if not r["ok"]:
            print(f" {r['step']:+4g}   FAILED: {r['error'][:70]}")
            continue
        t = f"{r['throw_mm']:7.1f}" if r["throw_mm"] is not None else "  none "
        w = f"{r['waist_mm']:7.2f}" if r["waist_mm"] is not None else "  none "
        print(f" {r['anode_shift_mm']:+4g}  {r['cathode_anode_mm']:6.2f}  {r['sim_microperveance']:.3f}"
              f"  {r['P_child_1_over_d2']:.3f}   {r['P_langmuir_blodgett']:.3f}   {t}   {w}"
              f"    {r['lost_in_gun']}/{r['n_trajectories']}")
    print("\n2) cathode radius of curvature (anode re-placed by Langmuir-Blodgett, same P)")
    print(" angle   RcT mm  aim mm  focus mm  waist mm  P sim  lost in gun  (hit at mm)")
    for r in data["rc"]:
        if not r["ok"]:
            print(f" {r['step']:+3g}deg  FAILED: {r['error'][:70]}")
            continue
        t = f"{r['throw_mm']:7.1f}" if r["throw_mm"] is not None else "  none "
        w = f"{r['waist_mm']:7.2f}" if r["waist_mm"] is not None else "  none "
        hit = r["lost_in_gun_at_mm"][:3]
        print(f" {r['thetaT_deg']:5.1f}  {r['RcT_mm']:6.1f}  {r['aim_point_mm']:6.1f}  {t}   {w}"
              f"   {r['sim_microperveance']:.3f}   {r['lost_in_gun']}/{r['n_trajectories']}"
              f"  {hit if hit else ''}")


def figures(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figs = os.path.join(ds.repo_root(), "reports", "figures")
    os.makedirs(figs, exist_ok=True)
    blue, grey, orange = "#2a6fdb", "#8a8f98", "#d9822b"
    A = [r for r in data["anode"] if r["ok"]]
    R = [r for r in data["rc"] if r["ok"]]
    style = dict(lw=2, ms=7)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = [r["anode_shift_mm"] for r in A]
    ax[0].plot(x, [r["sim_microperveance"] for r in A], "o-", color=blue, label="target.exe", **style)
    ax[0].plot(x, [r["P_child_1_over_d2"] for r in A], "--", color=grey, label="1/d² scaling")
    ax[0].plot(x, [r["P_langmuir_blodgett"] for r in A], ":", color=orange,
               label="Langmuir–Blodgett")
    ax[0].axhline(data["design"]["P"], color="k", lw=0.8)
    ax[0].set(xlabel="anode shift (mm, + = away from cathode)", ylabel="microperveance",
              title="Current drawn")
    ax[0].legend(frameon=False)
    xt = [r["anode_shift_mm"] for r in A if r["throw_mm"] is not None]
    ax[1].plot(xt, [r["throw_mm"] for r in A if r["throw_mm"] is not None], "o-", color=blue, **style)
    ax[1].set(xlabel="anode shift (mm)", ylabel="focus point from cathode (mm)", title="Where the beam focuses")
    ax[2].plot(xt, [r["waist_mm"] for r in A if r["throw_mm"] is not None], "o-", color=blue, **style)
    ax[2].axhline(data["design"]["rw_mm"], color="k", lw=0.8)
    ax[2].set(xlabel="anode shift (mm)", ylabel="waist radius (mm)", title="Beam waist (line: design)")
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Our gun in target.exe: moving the anode ({data['gun']})", fontweight="bold")
    fig.tight_layout()
    p1 = os.path.join(figs, f"sweep_anode_{data['gun']}.png")
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    x = [r["RcT_mm"] for r in R]
    ok = [r for r in R if r["throw_mm"] is not None]
    ax[0].plot(x, [r["aim_point_mm"] for r in R], "--", color=grey, label="aim point (= R_c)")
    ax[0].plot([r["RcT_mm"] for r in ok], [r["throw_mm"] for r in ok], "o-", color=blue,
               label="focus in target.exe", **style)
    ax[0].set(xlabel="cathode radius of curvature R_c (mm)", ylabel="distance from cathode (mm)",
              title="Where the beam focuses")
    ax[0].legend(frameon=False)
    ax[1].plot([r["RcT_mm"] for r in ok], [r["waist_mm"] for r in ok], "o-", color=blue, **style)
    ax[1].axhline(data["design"]["rw_mm"], color="k", lw=0.8)
    ax[1].set(xlabel="R_c (mm)", ylabel="waist radius (mm)", title="Beam waist (line: design)")
    ax[2].plot(x, [r["sim_microperveance"] for r in R], "o-", color=blue, **style)
    ax[2].axhline(data["design"]["P"], color="k", lw=0.8)
    ax[2].set(xlabel="R_c (mm)", ylabel="microperveance", title="Current drawn (line: design)")
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Our gun in target.exe: changing the cathode's radius of curvature ({data['gun']})",
                 fontweight="bold")
    fig.tight_layout()
    p2 = os.path.join(figs, f"sweep_rc_{data['gun']}.png")
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    return p1, p2


def export_web(names):
    """web/data/sweeps.js: both sweeps with thinned trajectory plots, per gun."""
    rnd = lambda seg, k: [[round(z, 2), round(r, 2)] for z, r in seg[::k]] + [
        [round(seg[-1][0], 2), round(seg[-1][1], 2)]]
    out = {}
    for n in names:
        with open(os.path.join(OUT, f"{n}.json"), encoding="utf-8") as f:
            data = json.load(f)
        for key in ("anode", "rc"):
            for r in data[key]:
                if not r["ok"]:
                    continue
                with open(os.path.join(OUT, n, r["run"], "plot.json"), encoding="utf-8") as f:
                    p = json.load(f)
                r["plot"] = {"boundary": [rnd(x, 1) for x in p["boundary"]],
                             "trajectories": [rnd(x, 2) for x in p["trajectories"]],
                             "equipotentials": [rnd(x, 3) for x in p["equipotentials"]]}
        out[n] = data
    path = os.path.join(ds.repo_root(), "web", "data", "sweeps.js")
    with open(path, "w", encoding="utf-8") as f:
        f.write("/* GENERATED by src/sim/sweeps.py -- do not edit. */\n"
                "window.EGUN_DATA = window.EGUN_DATA || {};\n"
                "window.EGUN_DATA.sweeps = " + json.dumps(out, separators=(",", ":"), default=float)
                + ";\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gun", default="sir_gun")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--report-only", action="store_true", help="reuse the saved sweep")
    a = p.parse_args(argv)
    if a.report_only:
        with open(os.path.join(OUT, f"{a.gun}.json"), encoding="utf-8") as f:
            data = json.load(f)
    else:
        print(f"running {len(ANODE_SHIFTS) + len(THETA_STEPS)} target.exe simulations of our gun ...")
        data = run_all(a.gun, a.workers)
    report(data)
    for fp in figures(data):
        print("figure:", os.path.relpath(fp, ds.repo_root()))
    done = [f[:-5] for f in os.listdir(OUT) if f.endswith(".json")]
    print("web data:", os.path.relpath(export_web(sorted(done)), ds.repo_root()))


if __name__ == "__main__":
    main()
