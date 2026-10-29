"""
Gate G3: re-run Pareto designs through Vedant's EXACT solver.

    python -m src.moo.verify                    # both studies
    python -m src.moo.verify shin_eik --n 12

For each study, picks n designs spread evenly along reports/moo/<name>/pareto.csv
(always including the knee), synthesises each with his mpmath solver, and
compares theta, theta_T, Rc, z_ac and the throw with what the optimiser saw.
Writes reports/moo/<name>/verify.csv. Exits NON-ZERO if any verified design is
infeasible for his solver or differs by more than 2 % in any compared quantity.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from ..physics import vaughan_adapter as va
from .run import out_dir

TOL = 0.02
KEYS = ("theta", "thetaT", "Rc_mm", "zac_mm", "zw_mm")


def pick(n_rows, n, knee):
    idx = set(np.unique(np.round(np.linspace(0, n_rows - 1, min(n, n_rows))).astype(int)))
    idx.add(knee)
    return sorted(idx)


def verify_study(name, n=12, workers=None):
    d = out_dir(name)
    front = pd.read_csv(os.path.join(d, "pareto.csv"))
    with open(os.path.join(d, "knee.json"), encoding="utf-8") as f:
        knee = json.load(f)["index_in_pareto_csv"]
    rows = front.iloc[pick(len(front), n, knee)]
    designs = [{"V": r.V, "I": r.I, "rw_mm": r.rw_mm, "Jc": r.Jc,
                **{k: r[k] for k in KEYS}} for _, r in rows.iterrows()]
    out = pd.DataFrame(va.verify(designs, tol=TOL, keys=KEYS, workers=workers))
    out.insert(0, "pareto_index", rows.index.to_numpy())
    out.insert(1, "is_knee", rows.index.to_numpy() == knee)
    out.to_csv(os.path.join(d, "verify.csv"), index=False, float_format="%.10g")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("studies", nargs="*", default=["shin_eik", "ceeri_klystron"])
    p.add_argument("--n", type=int, default=12)
    a = p.parse_args(argv)
    failed = False
    for name in a.studies:
        out = verify_study(name, a.n)
        n_bad = int((~out.ok).sum())
        print(f"{name}: verified {len(out)} designs with the exact solver; "
              f"infeasible {int((~out.feasible).sum())}; "
              f"max rel error {np.nanmax(out.max_err):.2e}; failures {n_bad}")
        if len(out) < 10 or n_bad:
            failed = True
    print("G3", "FAIL" if failed else "PASS")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
