"""
Our gun, from our website, simulated in sir's PIERCE (target.exe).

    python -m src.sim.ourgun                 # sir's reference gun + the paper's 7 held-out guns
    python -m src.sim.ourgun --tune          # + anode position tuned with PIERCE in the loop

"Our gun" is the combination the website computes:

  inputs   P (microperv), r_w (mm), C          the Predict tab's three inputs
  angle    theta = our network M1's prediction (models/model_m1.json)
  gun      synthesised around that angle by Vaughan's (1981) equations: cathode
           built with the aberration correction (R_cT = 0.905 R_c), anode face and
           aperture r_a = 1.2 r_b, drift tube, throw (Eqs. 3, 5, 15-21)
  focus electrode  Tiwary & Basu (1988): the V = 0 equipotential of the ideal
           flow, leaving the cathode at 67.5 deg to the beam edge
           (src/physics/bfe.py), followed up to 1.5 x the cathode radius or its
           highest point, then joined to the outer wall

The deck is written with sir's solver settings and run in target.exe. Compared:
PIERCE's perveance with the input P (does the gun draw the current it was
designed for?), whether any electron hits the gun before the waist, and the waist.

Found when building it: a gun made of Tiwary-Basu curves alone (both electrodes
cut at their highest point, plain walls) failed in PIERCE on 2 of the 7 guns
(electrons stopped at the cathode rim; beam hit the anode lip). The combined
construction above runs on all of them.
"""

import argparse
import json
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..ann import dataset as ds
from ..physics import bfe
from ..physics import vaughan_adapter as va
from . import experiments as ex
from . import lktron as lk

OUT = os.path.join(ds.repo_root(), "reports", "simulation", "ourgun")
RUNS = os.path.join(ds.repo_root(), "simulation", "runs", "ourgun")
V_DEFAULT = 10e3
REFERENCE_GUN = ("sir_gun", 2.0656, 9.8, 7.9544, None, 150e3)   # CEERI, sir's reference deck
# Building a deck changes module-wide settings (grid) and drives the synthesis
# through process-wide state; only one build may run at a time. Simulations
# (target.exe) run in parallel.
_BUILD = threading.Lock()


def predict_theta(P, rw, C):
    """Our website's angle: model M1, exactly as the Predict tab computes it."""
    from ..ann.stage2 import predict_payload
    from ..export.weights_to_json import load_model
    m = load_model(os.path.join(ds.repo_root(), "models", "model_m1.json"))
    return float(predict_payload(m, np.array([[P, rw, C]]))[0, 0])


def build(P, rw, C, V=V_DEFAULT, theta=None, anode_shift=0.0):
    """Deck rows of our gun. Returns (deck text, grid, info)."""
    theta = predict_theta(P, rw, C) if theta is None else theta
    ins = va.to_vaughan_inputs(P, rw, C, V)
    exact = va.synthesize(ins["V"], ins["I"], rw, ins["Jc"])
    if not exact["feasible"]:
        raise ValueError(exact["reason"])
    r = ex.geometry_from_theta(exact, theta)
    h = lk.choose_grid(r["rc_mm"], ex.TUBE_FACTOR * r["zw_mm"] + 1)
    ex.H, ex.D = h, h
    rows, meta, dp = ex._vedant_rows(r)
    # rows: 0 vertex(arc) 1 rim 2 body(-1) 3 body 4 electrode base 5 electrode tip 6 wall(-1) 7.. anode
    thT = math.radians(r["thetaT"])
    curve = bfe.to_gun(bfe.bfe_s(n=4000), r["RcT_mm"], thT, d=h)
    curve = curve + [rows[4]["z"] - curve[0, 0], rows[4]["r"] - curve[0, 1]]
    top = int(np.argmax(curve[:, 1]))
    keep = [p for p in curve[1:top + 1] if p[1] <= dp["r6"] and p[0] < rows[7]["z"] - 2 * h]
    if not keep:
        raise ValueError("the Tiwary-Basu electrode has no room before the anode")
    tip = keep[-1]
    wall = dict(rows[6])
    wall["z"] = float(tip[0])
    rows = rows[:5] + [{"code": 1, "z": float(z), "r": float(q), "volt": 0.0} for z, q in keep] \
        + [wall] + rows[7:]
    for row in rows:
        if row["volt"] > 0:
            row["z"] += anode_shift
    info = {"P": P, "rw_mm": rw, "C": C, "V": V, "I": ins["I"], "theta_net_deg": theta,
            "thetaT_built_deg": r["thetaT"], "theta_raw_deg": r["theta"],
            "electrode_angle_to_edge_deg": 67.5, "electrode_angle_to_axis_deg": 67.5 - r["thetaT"],
            "electrode_tip_mm": [float(tip[0]), float(tip[1])], "grid_mm": h,
            "design_throw_mm": r["zw_mm"], "anode_shift_mm": anode_shift}
    title = f"OUR GUN P{P:g} RW{rw:g} C{C:g}"
    return lk.deck_text(rows, title=title, h=h, d=h, crad=meta["Rc_mm"], rc=r["rc_mm"]), h, info


def simulate(name, P, rw, C, V=V_DEFAULT, anode_shift=0.0):
    try:
        with _BUILD:
            text, h, info = build(P, rw, C, V, anode_shift=anode_shift)
    except (ValueError, va.DeckRefused) as e:
        return {"name": name, "ok": False, "error": f"no deck: {e}"}
    work = os.path.join(RUNS, name)
    try:
        sim = lk.run(text, work)
    except lk.SimulationFailed as e:
        return {"name": name, "ok": False, "error": str(e), "info": info}
    zw, rw_sim = ex.refined_waist(sim["envelope"]) if sim["envelope"] else (math.nan, math.nan)
    c = {"design_microperveance": P, "sim_microperveance": sim["microperveance"],
         "perveance_err_pct": 100 * (sim["microperveance"] / P - 1),
         "design_current_A": info["I"], "sim_current_A": sim["cathode_current_A"],
         "design_waist_mm": rw, "sim_waist_mm": rw_sim,
         "waist_err_pct": 100 * (rw_sim / rw - 1),
         "design_throw_mm": info["design_throw_mm"], "sim_throw_mm": zw - h,
         "waist_inside_domain": not math.isnan(rw_sim), "converged": sim.get("converged"),
         "wall_seconds": sim["wall_seconds"]}
    c.update(ex.trajectory_losses(work, c["sim_throw_mm"]))
    res = {"name": name, "ok": True, "info": info, "compare": c,
           "current_history_A": sim.get("current_history_A")}
    dest = os.path.join(OUT, name)
    lk.copy_outputs(work, dest, names=("pierce.in", "pierce.log", "stdout.txt"))
    with open(os.path.join(dest, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)
    with open(os.path.join(dest, "plot.json"), "w", encoding="utf-8") as f:
        json.dump(ex.plot_payload(work), f, separators=(",", ":"))
    return res


def guns():
    """Sir's reference gun (CEERI, 150 kV) and the paper's 7 held-out real guns."""
    lit = ds.literature_only(ds.load())
    _, test = ds.paper_split(lit)
    out = [REFERENCE_GUN]
    out += [(f"case{int(r.source_case_id)}", r.perveance_uperv, r.rw_mm, r.C, r.theta_cone_deg,
             V_DEFAULT) for r in test.itertuples()]
    return out


def tune_anode(name, P, rw, C, V, first, tol_pct=1.0, max_iter=6):
    """Secant on the anode position until PIERCE's perveance equals the input P."""
    step = 0.1 * first["info"]["design_throw_mm"] / 2
    a, fa = 0.0, first["compare"]["perveance_err_pct"]
    b = step if fa > 0 else -step                     # too much current -> anode further away
    rb = simulate(f"{name}_dz{b:+.3f}", P, rw, C, V, b)
    hist = [(a, fa)]
    if not rb["ok"]:
        return {"converged": False, "history": hist, "result": rb, "anode_shift_mm": b}
    fb = rb["compare"]["perveance_err_pct"]
    hist.append((b, fb))
    for _ in range(max_iter):
        if abs(fb) <= tol_pct or fb == fa:
            break
        c = round(b - fb * (b - a) / (fb - fa), 3)
        a, fa = b, fb
        b = c
        rb = simulate(f"{name}_dz{b:+.3f}", P, rw, C, V, b)
        if not rb["ok"]:
            break
        fb = rb["compare"]["perveance_err_pct"]
        hist.append((b, fb))
    return {"converged": abs(fb) <= tol_pct, "history": hist, "result": rb, "anode_shift_mm": b}


def line(name, info, c, measured=None):
    m = f" (measured {measured:5.1f})" if measured else " " * 17
    w = (f"waist {c['sim_waist_mm']:.2f} vs {info['rw_mm']:g} mm ({c['waist_err_pct']:+.0f} %)"
         if c["waist_inside_domain"] else "no waist in the domain")
    return (f"{name:8s} angle {info['theta_net_deg']:5.1f}{m}  perveance "
            f"{c['sim_microperveance']:.3f} vs {info['P']:.3f} ({c['perveance_err_pct']:+6.1f} %)  "
            f"lost in gun {c['lost_in_gun']}/{c['n_trajectories']}  {w}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tune", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args(argv)
    gs = guns()
    print("our gun = network angle + Vaughan synthesis + Tiwary-Basu focus electrode, in PIERCE")
    print(f"{len(gs)} guns: sir's reference gun and the paper's 7 held-out real guns")
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        res = list(pool.map(lambda g: simulate(g[0], *g[1:4], V=g[5]), gs))
    rows = []
    for g, r in zip(gs, res):
        if not r["ok"]:
            print(f"{g[0]:8s} FAILED: {r['error'][:90]}")
            rows.append({"case": g[0], "ok": False, "error": r["error"]})
            continue
        print(line(g[0], r["info"], r["compare"], g[4]))
        rows.append({"case": g[0], "ok": True, "theta_measured": g[4], **r["info"], **r["compare"]})
    tuned = []
    if a.tune:
        print("\ntuning the anode position, PIERCE in the loop, target: the input perveance")
        todo = [(g, r) for g, r in zip(gs, res) if r["ok"] and r["compare"]["lost_in_gun"] == 0]
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            outs = list(pool.map(lambda gr: tune_anode(gr[0][0] + "_tune", *gr[0][1:4], gr[0][5], gr[1]),
                                 todo))
        for (g, r0), t in zip(todo, outs):
            if not t["result"]["ok"]:
                print(f"{g[0]:8s} tuning failed: {t['result']['error'][:70]}")
                continue
            c = t["result"]["compare"]
            print(f"{g[0]:8s} anode {t['anode_shift_mm']:+6.2f} mm -> perveance "
                  f"{c['sim_microperveance']:.3f} vs {g[1]:.3f} ({c['perveance_err_pct']:+5.1f} %), "
                  f"lost {c['lost_in_gun']}, {len(t['history'])} PIERCE runs")
            tuned.append({"case": g[0], "anode_shift_mm": t["anode_shift_mm"],
                          "converged": t["converged"], "history": t["history"],
                          "variant": t["result"]["name"], **c})
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    old["guns"] = rows
    if tuned:
        old["tuned"] = tuned
    with open(path, "w", encoding="utf-8") as f:
        json.dump(old, f, indent=2, default=float)


if __name__ == "__main__":
    main()
