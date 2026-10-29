"""
Write Vedant's EGUN point deck for a design.

    python -m src.moo.deck --study shin_eik               # the study's knee design
    python -m src.moo.deck --study ceeri_klystron --index 7
    python -m src.moo.deck --V 14000 --I 0.5 --rw 0.2255 --Jc 5 --out my_gun.txt

The design is first re-synthesised with his EXACT solver (never the emulator),
and the deck is written by his own code (get_design_params_mode1 ->
compute_gun_points_mm -> print_gun_points) with his defaults for the free
geometry parameters (k1 0.02, d 0.2 mm, H 0.05 mm, ...); override any of them
with --set key=value using his prompt names (k1, d, l, H_z, H_r, g, nH_gap).

Refuses -- non-zero exit, no file -- when his solver finds the design infeasible,
or when theta_raw > 56.44 deg, where Vaughan's Eq.(20) for the anode plane has
no real root (the deck's aperture point would be invented). Sharma et al. (2001)
would fix the latter and is not implemented.
"""

import argparse
import json
import os
import sys

import pandas as pd

from ..physics import vaughan_adapter as va
from .run import out_dir


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--study")
    p.add_argument("--index", type=int, help="row of pareto.csv (default: knee)")
    p.add_argument("--V", type=float); p.add_argument("--I", type=float)
    p.add_argument("--rw", type=float, help="mm"); p.add_argument("--Jc", type=float)
    p.add_argument("--out")
    p.add_argument("--set", action="append", default=[], help="key=value geometry override")
    a = p.parse_args(argv)

    if a.study:
        d = out_dir(a.study)
        front = pd.read_csv(os.path.join(d, "pareto.csv"))
        if a.index is None:
            with open(os.path.join(d, "knee.json"), encoding="utf-8") as f:
                a.index = json.load(f)["index_in_pareto_csv"]
        row = front.iloc[a.index]
        V, I, rw, Jc = row.V, row.I, row.rw_mm, row.Jc
        out = a.out or os.path.join(d, f"egun_deck_{a.index}.txt")
    else:
        if None in (a.V, a.I, a.rw, a.Jc):
            p.error("give --study, or all of --V --I --rw --Jc")
        V, I, rw, Jc = a.V, a.I, a.rw, a.Jc
        out = a.out or "egun_deck.txt"

    overrides = dict(kv.split("=", 1) for kv in a.set)
    r = va.synthesize(V, I, rw, Jc)
    try:
        pts, meta, dp = va.write_deck(r, out, overrides,
                                      report_path=os.path.splitext(out)[0] + "_report.txt")
    except va.DeckRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        sys.exit(2)
    print(f"deck written: {out}")
    print(f"  exact solver: theta {r['theta']:.4f} deg, theta_T {r['thetaT']:.4f} deg, "
          f"Rc {r['Rc_mm']:.4f} mm, zac {r['zac_mm']:.4f} mm, throw {r['zw_mm']:.4f} mm")
    print(f"  focus electrode built at {meta['psi_deg']:.3f} deg to the beam edge")


if __name__ == "__main__":
    main()
