"""
Any gun -> a deck sir's PIERCE (target.exe) runs -> optionally run it and compare.

    python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --out my_gun/pierce.in
    python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --psi 65.66 --anode-shift 2.27 --run
    python -m src.sim.deck --V 20000 --I 1 --rw 0.5 --Jc 5 --electrodes tiwary-basu --run

The synthesis is Vedant's exact solver. The geometry is Vedant's construction
(focus electrode a straight line at --psi to the beam edge, default 67.5) or the
Tiwary-Basu (1988) equipotential electrodes. The deck carries the &CONTROL
values of sir's reference deck and the closing '/' that Vedant's writer omits.
Grid: chosen from the cathode radius (~50 cells across it; 0.5 mm for the CEERI
gun, as in sir's deck) unless --grid is given. The drift tube runs to 2 x the
designed throw so the waist forms inside the domain.

With --run, target.exe is run in the output folder (by default
simulation/runs/<name>) and the simulated current, throw and waist are compared
with the design. Open the folder and run sir's pierce_plotter.py / new3d.py there
for his plots.
"""

import argparse
import os
import sys

from ..ann import dataset as ds
from ..physics import vaughan_adapter as va
from . import experiments as ex
from . import lktron as lk


def build_deck(V, I, rw, Jc, electrodes="vedant", psi=None, anode_shift=0.0, grid=None):
    exact = va.synthesize(V, I, rw, Jc)
    if not exact["feasible"]:
        raise ValueError(f"no Pierce gun for these inputs: {exact['reason']}")
    h = grid or lk.choose_grid(exact["rc_mm"], zmax_mm=ex.TUBE_FACTOR * exact["zw_mm"] + 1)
    ex.H, ex.D = h, h                           # grid step; cathode vertex one cell in
    if electrodes == "tiwary-basu":
        rows, notes = ex.tiwary_basu_rows(exact)
        crad = exact["Rc_mm"]
    else:
        rows, meta, dp = ex._vedant_rows(exact, psi=psi)
        crad = meta["Rc_mm"]
        notes = {"psi_deg": meta["psi_deg"]}
    for r in rows:
        if r["volt"] > 0:
            r["z"] += anode_shift
    title = f"PIERCE GUN {V/1e3:g}KV {I:g}A {electrodes.upper()}"
    deck = lk.deck_text(rows, title=title, h=h, d=ex.D, crad=crad, rc=exact["rc_mm"])
    return deck, exact, {"grid_mm": h, **notes}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--V", type=float, required=True, help="beam voltage, V")
    p.add_argument("--I", type=float, required=True, help="beam current, A")
    p.add_argument("--rw", type=float, required=True, help="beam waist radius, mm")
    p.add_argument("--Jc", type=float, required=True, help="cathode loading, A/cm^2")
    p.add_argument("--electrodes", choices=("vedant", "tiwary-basu"), default="vedant")
    p.add_argument("--psi", type=float, help="focus electrode angle to the beam edge, deg "
                                              "(Vedant's construction; default 67.5)")
    p.add_argument("--anode-shift", type=float, default=0.0,
                   help="move the anode and drift tube along the axis, mm")
    p.add_argument("--grid", type=float, help="grid step H, mm")
    p.add_argument("--out", help="deck path (default simulation/runs/<name>/pierce.in)")
    p.add_argument("--run", action="store_true", help="run target.exe and compare")
    a = p.parse_args(argv)
    try:
        deck, exact, notes = build_deck(a.V, a.I, a.rw, a.Jc, a.electrodes, a.psi,
                                        a.anode_shift, a.grid)
    except (ValueError, va.DeckRefused) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
    name = f"gun_{a.V/1e3:g}kV_{a.I:g}A_{a.electrodes}" + (f"_psi{a.psi:g}" if a.psi else "")
    out = a.out or os.path.join(ds.repo_root(), "simulation", "runs", name, "pierce.in")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="ascii", newline="\n") as f:
        f.write(deck)
    print(f"deck written: {out}  (grid {notes['grid_mm']} mm)")
    print(f"design: I {exact['I']:g} A, microperveance {exact['muP']:.4f}, theta_T "
          f"{exact['thetaT']:.3f} deg, throw {exact['zw_mm']:.2f} mm, waist {exact['rw_mm']} mm")
    if not a.run:
        return
    work = os.path.dirname(os.path.abspath(out))
    sim = lk.run(deck, work)
    c = ex.compare(sim, exact, d=ex.D)
    c.update(ex.trajectory_losses(work, c["sim_throw_mm_refined"]))
    print(f"PIERCE: I {c['sim_current_A']:.2f} A ({c['current_err_pct']:+.1f} %), "
          f"microperveance {c['sim_microperveance']:.4f}, lost in gun {c['lost_in_gun']}/"
          f"{c['n_trajectories']} trajectories")
    if c["waist_inside_domain"]:
        print(f"        throw {c['sim_throw_mm_refined']:.1f} mm ({c['throw_err_pct']:+.1f} %), "
              f"waist {c['sim_waist_mm_refined']:.2f} mm ({c['waist_err_pct']:+.1f} %)")
    else:
        print("        no waist inside the domain (beam still converging at the end wall)")
    print(f"        outputs in {work}; run sir's pierce_plotter.py there for his plots")


if __name__ == "__main__":
    main()
