"""
Simulate synthesised guns in Dr. Lalit Kumar's PIERCE (target.exe) and compare
with what the synthesis promised.

    python -m src.sim.experiments                 # all variants, CEERI gun
    python -m src.sim.experiments --only reference vedant

The gun is the CEERI 8 MW klystron (150 kV, 120 A, rw 9.8 mm, 5 A/cm^2)
because it is the gun sir's own reference deck simulates
(simulation/lktron/reference_ceeri/pierce.in), so it is the one case where our
pipeline can be checked against a run sir already trusts.

Variants
--------
reference      sir's deck, unchanged (must reproduce his 121.74 A).
vedant         Vedant's deck as his code builds it: cathode with the
               aberration-corrected thetaT, RcT (Vaughan Sec. III-F), straight
               focus electrode at 67.5 deg to the beam edge.
vedant_raw     the same construction with the uncorrected theta, Rc -- what
               sir's reference deck uses (CRAD = 57.8 = Rc).
tiwary_basu    cathode Rc, theta (the ideal flow), focus electrode and anode
               face replaced by the V = 0 and V = Va equipotentials of
               Tiwary & Basu (1988), src/physics/bfe.py.
psi_<a>        Vedant's construction with the focus electrode at a deg to the
               beam edge instead of 67.5 -- the parameter Gao et al. (2025,
               AIP Adv. 15, 015030) sweep in COMSOL.

Outputs: reports/simulation/<variant>/{pierce.in, pierce.log, summary.json,
plot.json} and reports/simulation/summary.json (+ .md).
"""

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..ann import dataset as ds
from ..physics import bfe, vaughan_adapter as va
from . import lktron as lk

GUN = {"V": 150e3, "I": 120.0, "rw_mm": 9.8, "Jc": 5.0}
H, D = 0.5, 0.5                      # sir's grid step and cathode offset (OCAT)
# Drift tube length for simulation, as a multiple of the designed throw. Vedant's
# deck ends it at 1.3 x throw; PIERCE's waists are often beyond that, and a waist
# that sits on the end wall is not a waist (sir's own deck runs to z = 130 mm,
# 1.67 x). The electrodes are untouched; only the tube is longer.
TUBE_FACTOR = 2.0
OUT = os.path.join(ds.repo_root(), "reports", "simulation")
RUNS = os.path.join(ds.repo_root(), "simulation", "runs")
PSI_SWEEP = (57.5, 62.5, 72.5)


# --- deck builders -------------------------------------------------------------

def extend_tube(rows, zmax):
    """Move the drift tube's last two points (end wall, on-axis back) to zmax."""
    for r in rows[-2:]:
        r["z"] = zmax
    return rows


def _vedant_rows(result, psi=None):
    """Vedant's own construction (his get_design_params_mode1 defaults, with
    sir's grid and offset), optionally at another focus-electrode angle psi.
    His module-level PIERCE_ANGLE is swapped for the call and restored."""
    m = va.module()
    old = m.PIERCE_ANGLE
    if psi is not None:
        m.PIERCE_ANGLE = psi
    try:
        tmp = os.path.join(RUNS, "_vedant_tmp.txt")
        os.makedirs(RUNS, exist_ok=True)
        pts, meta, dp = va.write_deck(result, tmp, {"H_z": H, "H_r": H, "d": D})
    finally:
        m.PIERCE_ANGLE = old
    rows = lk.rows_from_vedant(pts, meta, result["V"], H)
    return extend_tube(rows, D + TUBE_FACTOR * result["zw_mm"]), meta, dp


def uncorrected(result):
    """The same synthesis with the cathode built at theta, Rc (no Sec. III-F
    aberration correction), expressed in the form his geometry code reads."""
    r = dict(result)
    r["raw_inches"] = dict(result["raw_inches"])
    r["thetaT"] = result["theta"]
    r["RcT"] = result["Rc"]
    r["raw_inches"]["RcT"] = result["raw_inches"]["Rc"]
    r["RcT_mm"] = result["Rc_mm"]
    return r


def ann_theta(exact, model):
    """theta_T predicted by one of our networks for this gun's (P, rw, C).
    model: "m1" (trained on the 23 real guns) or "anne" (trained on Vedant's solver)."""
    from ..ann.stage2 import predict_payload
    from ..export.weights_to_json import load_model
    p = load_model(os.path.join(ds.repo_root(), "models", f"model_{model}.json"))
    return float(predict_payload(p, np.array([[exact["muP"], exact["rw_mm"], exact["C"]]]))[0, 0])


def geometry_from_theta(exact, thetaT_deg):
    """
    The gun Vedant's equations build when the angle is GIVEN instead of solved:
    theta_raw from theta_T (sin theta_raw = 0.905 sin theta_T), then Vaughan's
    Eqs. (3), (5), (15)-(21) exactly as his solver applies them after its solve,
    with his relativistic factor for this gun. This is how our ANN's angle turns
    into a gun: our original code predicts the angle but has no electrode builder.
    """
    from ..physics import lb
    ri = exact["raw_inches"]
    rc, rw, muP, kf = ri["rc"], ri["rw"], exact["muP"], exact["rel_factor"]
    th = math.asin(lb.ABERRATION_K * math.sin(math.radians(thetaT_deg)))
    a = math.sqrt(lb.VAUGHAN_CONE_CONST * kf * (1 - math.cos(th)) / muP)
    y = float(lb.y_from_a(a))
    R = rc * math.exp(-y) / rw
    Rc = rc / math.sin(th)
    Ra = Rc * math.exp(-y)
    rb = rc * math.exp(-y)
    ra = lb.APERTURE_FACTOR * rb
    zac = Rc - Ra
    disc = Ra ** 2 - ra ** 2
    za = Rc - math.sqrt(disc) if disc > 0 else Rc
    Rp = max(R - 1, 0.0)
    zw = zac + rw * (11 * Rp ** 0.5 + 1.32 * Rp + 0.0615 * Rp ** 2 - 0.00167 * Rp ** 3) / math.sqrt(muP)
    inches = {"rw": rw, "rc": rc, "Rc": Rc, "Ra": Ra, "zac": zac, "za": za, "rb_za": rb,
              "ra": ra, "zw": zw, "RcT": lb.ABERRATION_K * Rc}
    r = dict(exact)
    r.update(inches)
    r["raw_inches"] = inches
    for k, v in inches.items():
        r[k + "_mm"] = v * 25.4
    r["rw_mm"] = exact["rw_mm"]
    r.update(theta=math.degrees(th), thetaT=thetaT_deg, R=R, za_valid=disc > 0)
    return r


def tiwary_basu_rows(result):
    """
    Deck rows for the Tiwary-Basu electrodes on the ideal flow.

    Cathode: sphere Rc from the vertex to the rim at theta (uncorrected).
    Focus electrode: the V = 0 equipotential from the rim to its highest point,
      then straight up to the outer wall (1.75 x cathode radius, Vedant's Rmax).
    Anode: the V = Va equipotential from its highest point down to the aperture,
      ra = 1.2 rb(za) (Vaughan Eq. 18, Vedant's value); a straight face from the
      outer wall down to that highest point.
    Drift tube: Vedant's rule -- from the aperture parallel to the exit beam edge
      (tan phi2) to the plane of the centre of curvature, then flat to 1.3 x throw.
    Both curves are cut at their highest point because past it they turn back
    toward the axis (Fig. 2 of the paper), which a real electrode cannot do.
    """
    Rc, Ra, rc = result["Rc_mm"], result["Ra_mm"], result["rc_mm"]
    th = math.radians(result["theta"])
    V = result["V"]
    e = bfe.electrodes(Rc, Ra, result["theta"], d=D, r_max=1e9)
    Rmax = 1.75 * rc
    B = e["bfe"]
    B = B[: int(np.argmax(B[:, 1])) + 1]                   # rim -> apex
    A = e["anode"]
    A = A[: int(np.argmax(A[:, 1])) + 1]                   # beam edge -> apex
    ra = result["ra_mm"]
    A = A[A[:, 1] >= ra]                                   # outside the aperture
    if not len(A):
        raise ValueError("anode equipotential never clears the aperture radius")
    z_ap = A[0, 0]
    tan_phi2 = 0.17409 * math.sqrt(result["muP"] * abs(math.log(result["R"])))
    z12 = D + Rc
    r12 = ra - (z12 - z_ap) * tan_phi2
    Zmax = D + TUBE_FACTOR * result["zw_mm"]
    rows = [{"code": 0, "z": D, "r": 0.0, "volt": 0.0, "zc": D + Rc, "rad": Rc}]
    rows.append({"code": 1, "z": D + Rc * (1 - math.cos(th)), "r": rc, "volt": 0.0,
                 "exact_z": True})
    rows += [{"code": 1, "z": z, "r": r, "volt": 0.0} for z, r in B[1:]]
    rows.append({"code": -1, "z": B[-1, 0], "r": Rmax, "volt": 0.0})       # gap to anode
    rows.append({"code": 1, "z": A[-1, 0], "r": Rmax, "volt": V})
    rows += [{"code": 1, "z": z, "r": r, "volt": V} for z, r in A[::-1]]   # apex -> aperture
    rows += [{"code": 1, "z": z12, "r": r12, "volt": V},
             {"code": 1, "z": Zmax, "r": r12, "volt": V},
             {"code": 1, "z": Zmax, "r": 0.0, "volt": V}]
    geometry = {"bfe": B.tolist(), "anode": A.tolist(), "Rmax": Rmax,
                "bfe_tip": B[-1].tolist(), "anode_top": A[-1].tolist(),
                "gap_mm": float(math.hypot(A[-1, 0] - B[-1, 0], Rmax - A[-1, 1]))}
    return rows, geometry


def build(name, exact):
    """(deck text, cathode radius used, notes) for a variant."""
    title = f"PIERCE GUN {name.upper()}"
    if name == "reference":
        with open(lk.REFERENCE_DECK, encoding="ascii") as f:
            return f.read(), 57.80, {"source": "sir's reference deck, unchanged"}
    if name == "vedant":
        rows, meta, dp = _vedant_rows(exact)
        return (lk.deck_text(rows, title=title, h=H, d=D, crad=meta["Rc_mm"], rc=exact["rc_mm"]),
                meta["Rc_mm"], {"psi_deg": meta["psi_deg"], "cathode": "RcT (corrected)"})
    if name in ("ann_m1", "ann_e"):
        th = ann_theta(exact, "m1" if name == "ann_m1" else "anne")
        r = geometry_from_theta(exact, th)
        rows, meta, dp = _vedant_rows(r)
        return (lk.deck_text(rows, title=title, h=H, d=D, crad=meta["Rc_mm"], rc=exact["rc_mm"]),
                meta["Rc_mm"], {"psi_deg": meta["psi_deg"], "cathode": "RcT (corrected)",
                                "thetaT_ann_deg": th, "thetaT_vedant_deg": exact["thetaT"],
                                "designed_throw_mm": r["zw_mm"]})
    if name == "vedant_raw":
        r = uncorrected(exact)
        rows, meta, dp = _vedant_rows(r)
        return (lk.deck_text(rows, title=title, h=H, d=D, crad=meta["Rc_mm"], rc=exact["rc_mm"]),
                meta["Rc_mm"], {"psi_deg": meta["psi_deg"], "cathode": "Rc (uncorrected)"})
    if name == "tiwary_basu":
        rows, geo = tiwary_basu_rows(exact)
        return (lk.deck_text(rows, title=title, h=H, d=D, crad=exact["Rc_mm"], rc=exact["rc_mm"]),
                exact["Rc_mm"], {"cathode": "Rc (ideal flow)", **geo})
    if name.startswith("psi_"):
        psi = float(name[4:])
        rows, meta, dp = _vedant_rows(exact, psi=psi)
        return (lk.deck_text(rows, title=title, h=H, d=D, crad=meta["Rc_mm"], rc=exact["rc_mm"]),
                meta["Rc_mm"], {"psi_deg": meta["psi_deg"], "cathode": "RcT (corrected)",
                                "angle_to_axis_deg": psi - exact["thetaT"]})
    raise ValueError(name)


# --- comparison ----------------------------------------------------------------

def refined_waist(envelope):
    """Parabola through the envelope minimum and its neighbours (PIERCE samples
    the envelope every few mm). Returns (z, r) of the vertex, or (nan, nan) when
    the minimum is the last sample: the beam is still converging at the end of
    the domain and has no waist there."""
    e = np.array(envelope)
    i = int(np.argmin(e[:, 1]))
    if i == len(e) - 1:
        return math.nan, math.nan
    if 0 < i < len(e) - 1:
        z, r = e[i - 1:i + 2, 0], e[i - 1:i + 2, 1]
        a, b, c = np.polyfit(z, r, 2)
        if a > 0:
            zv = -b / (2 * a)
            return float(zv), float(c - b * b / (4 * a))
    return float(e[i, 0]), float(e[i, 1])


def compare(sim, exact, d=D):
    zw, rw = refined_waist(sim["envelope"]) if sim["envelope"] else (math.nan, math.nan)
    if sim["throw_mm"] is None:          # PIERCE found no waist inside the domain
        zw, rw = math.nan, math.nan
    nn = lambda v: math.nan if v is None else v
    P = exact["muP"]
    return {
        "design_current_A": exact["I"], "sim_current_A": sim["cathode_current_A"],
        "current_err_pct": 100 * (sim["cathode_current_A"] / exact["I"] - 1),
        "design_microperveance": P, "sim_microperveance": sim["microperveance"],
        "intercepted_A": sim["intercepted_A"],
        "transmission_pct": 100 * sim["beam_current_A"] / sim["cathode_current_A"]
        if sim["cathode_current_A"] else math.nan,
        "design_throw_mm": exact["zw_mm"], "sim_throw_mm_pierce": nn(sim["throw_mm"]) - d,
        "sim_throw_mm_refined": zw - d,
        "throw_err_pct": 100 * ((zw - d) / exact["zw_mm"] - 1),
        "design_waist_mm": exact["rw_mm"], "sim_waist_mm_pierce": nn(sim["waist_mm"]),
        "sim_waist_mm_refined": rw, "waist_err_pct": 100 * (rw / exact["rw_mm"] - 1),
        "waist_inside_domain": not math.isnan(rw),
        "converged": sim.get("converged"), "iterations": sim.get("iterations"),
        "wall_seconds": sim.get("wall_seconds"),
    }


def trajectory_losses(workdir, throw_mm):
    """Where trajectories end. A trajectory that stops before the end wall hit an
    electrode. Before the waist that is a gun defect (focus electrode, anode,
    aperture); after it, it is the expanding beam scraping the drift tube, which
    a real device prevents with its magnetic field and which says nothing about
    the gun. PIERCE gives intercepted current per electrode, not per place, so
    the split is reported as trajectory counts."""
    traj = lk.parse_plt(os.path.join(workdir, "pierce.plt"), keep=(2,))[2]
    ends = [t[-1] for t in traj]
    zend = max(z for z, _ in ends)
    lost = [z - D for z, _ in ends if z < zend - 2 * H]
    cut = throw_mm if throw_mm == throw_mm else math.inf       # nan -> no waist
    return {"n_trajectories": len(ends),
            "lost_in_gun": sum(z < cut for z in lost),
            "lost_after_waist": sum(z >= cut for z in lost),
            "lost_in_gun_at_mm": sorted(round(z, 1) for z in lost if z < cut)}


def plot_payload(workdir, every=2):
    """Boundary, trajectories and equipotentials, thinned for the web page."""
    p = lk.parse_plt(os.path.join(workdir, "pierce.plt"))
    thin = lambda seg, k: [[round(z, 3), round(r, 3)] for z, r in seg[::k]] + \
        ([[round(seg[-1][0], 3), round(seg[-1][1], 3)]] if len(seg) % k != 1 else [])
    return {"boundary": [thin(s, 1) for s in p[1]],
            "trajectories": [thin(s, every * 2) for s in p[2][::every]],
            "equipotentials": [thin(s, 4) for s in p[3]]}


def run_variant(name, exact, deck, crad, notes):
    """Simulate one prebuilt deck. Decks are built beforehand, one at a time:
    building drives Vedant's module through process-wide state (his
    PIERCE_ANGLE, stdout redirection), which is not thread-safe."""
    work = os.path.join(RUNS, name)
    try:
        sim = lk.run(deck, work)
    except lk.SimulationFailed as e:
        return name, {"name": name, "ok": False, "error": str(e), "notes": notes}
    res = {"name": name, "ok": True, "notes": notes, "cathode_radius_mm": crad,
           "pierce": {k: v for k, v in sim.items() if k != "current_history_A"},
           "current_history_A": sim.get("current_history_A"),
           "compare": compare(sim, exact)}
    res["compare"].update(trajectory_losses(work, res["compare"]["sim_throw_mm_refined"]))
    dest = os.path.join(OUT, name)
    lk.copy_outputs(work, dest, names=("pierce.in", "pierce.log", "stdout.txt"))
    with open(os.path.join(dest, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(dest, "plot.json"), "w", encoding="utf-8") as f:
        json.dump(plot_payload(work), f, separators=(",", ":"))
    return name, res


def tune_psi(exact, target_A=None, psi=(57.5, 62.5), tol=0.005, max_iter=6):
    """
    Simulation-in-the-loop: adjust Vedant's focus-electrode angle until PIERCE's
    cathode current equals the design current (perveance matching -- the step a
    gun designer does by hand in EGUN). Secant method on psi; each step is one
    PIERCE run. Returns the history and the final variant's result.
    """
    target_A = exact["I"] if target_A is None else target_A
    hist = []

    def f(p):
        name = f"tuned_psi_{p:.3f}"
        _, res = run_variant(name, exact, *build(f"psi_{p}", exact))
        hist.append({"psi_deg": p, **({"current_A": res["compare"]["sim_current_A"]}
                                      if res["ok"] else {"error": res["error"]})})
        if not res["ok"]:
            raise lk.SimulationFailed(res["error"])
        return res["compare"]["sim_current_A"] - target_A, res

    a, b = psi
    fa, _ = f(a)
    fb, rb = f(b)
    for _ in range(max_iter):
        if abs(fb) <= tol * target_A:
            break
        c = b - fb * (b - a) / (fb - fa)
        a, fa = b, fb
        b = round(c, 3)
        fb, rb = f(b)
    return {"target_A": target_A, "history": hist, "psi_deg": b, "result": rb,
            "converged": abs(fb) <= tol * target_A}


def build_shifted(exact, psi, dz):
    """Vedant's construction at focus-electrode angle psi, with every
    anode-potential point (anode face, aperture, drift tube) moved dz mm along
    the axis -- the cathode-anode spacing knob."""
    rows, meta, dp = _vedant_rows(exact, psi=psi)
    for r in rows:
        if r["volt"] > 0:
            r["z"] += dz
    deck = lk.deck_text(rows, title=f"PIERCE GUN TUNED {psi:.2f} {dz:+.2f}", h=H, d=D,
                        crad=meta["Rc_mm"], rc=exact["rc_mm"])
    return deck, meta["Rc_mm"], {"psi_deg": psi, "anode_shift_mm": dz,
                                 "cathode": "RcT (corrected)"}


def tune_2d(exact, x0=(57.085, 0.0), steps=(2.0, 2.0), tol=(0.005, 0.03), max_iter=4,
            workers=3):
    """
    Two knobs, two targets: (focus-electrode angle psi, anode axial shift dz)
    -> PIERCE cathode current = design current and beam waist = design waist.
    Newton with a forward-difference Jacobian; the three runs of each step go
    in parallel. Stops when current is within 0.5 % and waist within 3 %.
    """
    target = np.array([exact["I"], exact["rw_mm"]])
    hist = []

    def sims(points):
        built = [(f"tune2d_{p:.3f}_{z:+.3f}", build_shifted(exact, p, z)) for p, z in points]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            out = list(pool.map(lambda nb: run_variant(nb[0], exact, *nb[1]), built))
        vals = []
        for (p, z), (_, res) in zip(points, out):
            if not res["ok"]:
                raise lk.SimulationFailed(res["error"])
            c = res["compare"]
            y = np.array([c["sim_current_A"], c["sim_waist_mm_refined"]])
            hist.append({"psi_deg": p, "dz_mm": z, "current_A": y[0], "waist_mm": y[1],
                         "throw_mm": c["sim_throw_mm_refined"],
                         "lost_in_gun": c["lost_in_gun"]})
            if not np.isfinite(y).all():
                raise lk.SimulationFailed(f"no waist inside the domain at psi {p}, dz {z}")
            vals.append((y, res))
        return vals

    x = np.array(x0, float)
    for _ in range(max_iter):
        pts = [tuple(x), (x[0] + steps[0], x[1]), (x[0], x[1] + steps[1])]
        (y0, r0), (y1, _), (y2, _) = sims(pts)
        err = (y0 - target) / target
        if abs(err[0]) <= tol[0] and abs(err[1]) <= tol[1]:
            return {"converged": True, "x": x.tolist(), "result": r0, "history": hist}
        J = np.column_stack([(y1 - y0) / steps[0], (y2 - y0) / steps[1]])
        dx = np.linalg.solve(J, target - y0)
        dx = np.clip(dx, [-6.0, -8.0], [6.0, 8.0])         # trust region
        x = np.round(x + dx, 3)
    (y0, r0), = sims([tuple(x)])
    err = (y0 - target) / target
    return {"converged": bool(abs(err[0]) <= tol[0] and abs(err[1]) <= tol[1]),
            "x": x.tolist(), "result": r0, "history": hist}


def write_tunings():
    """Run both tunings and record them (reports/simulation/tuned_*.json)."""
    exact = va.synthesize(**GUN)
    t1 = tune_psi(exact)
    with open(os.path.join(OUT, "tuned_psi.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "secant on psi, PIERCE in the loop, target = design current",
                   "converged": t1["converged"], "psi_deg": t1["psi_deg"],
                   "variant": t1["result"]["name"], "history": t1["history"],
                   "compare": t1["result"]["compare"]}, f, indent=2)
    t2 = tune_2d(exact, x0=(t1["psi_deg"], 0.0))
    h = t2["history"]
    for q in h:
        q["err_I_pct"] = 100 * (q["current_A"] / exact["I"] - 1)
        q["err_w_pct"] = 100 * (q["waist_mm"] / exact["rw_mm"] - 1)
    best = min(h, key=lambda q: math.hypot(q["err_I_pct"], q["err_w_pct"]))
    name = f"tune2d_{best['psi_deg']:.3f}_{best['dz_mm']:+.3f}"
    with open(os.path.join(OUT, name, "summary.json"), encoding="utf-8") as f:
        bc = json.load(f)["compare"]
    with open(os.path.join(OUT, "tuned_2d.json"), "w", encoding="utf-8") as f:
        json.dump({"method": "Newton on (psi, anode shift), forward-difference Jacobian, "
                             "PIERCE in the loop; targets design current and design waist; "
                             "tolerance 0.5 % current, 3 % waist",
                   "converged": t2["converged"],
                   "best": {"psi_deg": best["psi_deg"], "anode_shift_mm": best["dz_mm"],
                            "variant": name, "compare": bc},
                   "history": h}, f, indent=2, default=float)
    print("psi only:", t1["psi_deg"], t1["result"]["compare"]["sim_current_A"])
    print("2-D best:", best, "converged", t2["converged"])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--tune", action="store_true",
                   help="only run the two simulation-in-the-loop tunings")
    a = p.parse_args(argv)
    if a.tune:
        return write_tunings()
    names = ["reference", "ann_m1", "ann_e", "vedant", "vedant_raw", "tiwary_basu"] + \
        [f"psi_{x:g}" for x in PSI_SWEEP]
    if a.only:
        names = [n for n in names if n in a.only] or a.only
    exact = va.synthesize(**GUN)
    decks = {n: build(n, exact) for n in names}               # sequential, see run_variant
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        results = dict(pool.map(lambda n: run_variant(n, exact, *decks[n]), names))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "summary.json")
    old = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f).get("variants", {})
    old.update(results)
    design = {k: exact[k] for k in ("V", "I", "rw_mm", "Jc", "muP", "theta", "thetaT",
                                    "Rc_mm", "RcT_mm", "Ra_mm", "zac_mm", "zw_mm", "rc_mm")}
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"gun": GUN, "grid_mm": H, "design": design, "variants": old}, f, indent=2)
    for n, r in results.items():
        if not r["ok"]:
            print(f"{n:14s} FAILED: {r['error']}")
            continue
        c = r["compare"]
        print(f"{n:14s} I {c['sim_current_A']:7.2f} A ({c['current_err_pct']:+6.1f} %)  "
              f"lost in gun {c['lost_in_gun']}/{c['n_trajectories']}  throw {c['sim_throw_mm_refined']:6.1f} mm "
              f"({c['throw_err_pct']:+5.1f} %)  waist {c['sim_waist_mm_refined']:5.2f} mm "
              f"({c['waist_err_pct']:+5.1f} %)  {c['wall_seconds']:.0f} s")


if __name__ == "__main__":
    main()
