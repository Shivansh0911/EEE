"""
The Design tab's JavaScript must agree with the Python it ports.

1. web/js/closure.js `synthesize` against src/physics/closure.py on 100 random
   designs: every reported quantity to < 1e-9 relative, feasibility identical.
2. web/js/closure.js `deck` against the deck Vedant's own code writes (via the
   adapter), fed the same exact-solver result: identical text.
3. The calibration formula against src/ann/stage2.apply_calibration.

Skips, saying so, if Node is not installed.
"""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from src.ann import dataset as ds
from src.ann.stage2 import apply_calibration
from src.physics import closure, gate_g1, kf
from src.physics import vaughan_adapter as va

ROOT = ds.repo_root()
JS = os.path.join(ROOT, "web", "js", "closure.js")
TOL = 1e-9
KEYS = ("muP", "C", "rc_mm", "theta", "thetaT", "Rc_mm", "Ra_mm", "zac_mm", "za_mm",
        "rb_za_mm", "ra_mm", "zw_mm", "RcT_mm", "R", "rel_factor")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node not installed")

DRIVER = r"""
const cl = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
let out;
if (job.kind === 'synth') {
  out = job.designs.map(d => cl.synthesize(d[0], d[1], d[2], d[3], job.table));
} else if (job.kind === 'deck') {
  out = job.results.map(r => { const d = cl.deck(r); return d.text || ('REFUSED ' + d.refused); });
} else {
  out = job.rows.map(r => cl.calibrate(job.params, r[0], r[1], r[2], r[3]));
}
process.stdout.write(JSON.stringify(out));
"""


def node(job):
    with tempfile.TemporaryDirectory() as d:
        drv, inp = os.path.join(d, "drv.js"), os.path.join(d, "job.json")
        with open(drv, "w") as f:
            f.write(DRIVER)
        with open(inp, "w") as f:
            json.dump(job, f)
        res = subprocess.run(["node", drv, JS, inp], capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def table_json():
    t = dict(kf.load_table())
    t["kf"] = t["kf"].tolist()
    return t


def test_synthesis_parity():
    V, I, rw, Jc = gate_g1.sample(100, seed=11)
    # add the infeasible / edge cases so refusals are compared too
    extra = np.array([[10e3, 0.1, 1.0, 50], [10e3, 5, 0.05, 1], [50e3, 30, 0.2, 20]])
    V, I, rw, Jc = (np.concatenate([a, extra[:, k]]) for k, a in enumerate((V, I, rw, Jc)))
    py = closure.synthesize(V, I, rw, Jc)
    js = node({"kind": "synth", "table": table_json(),
               "designs": np.column_stack([V, I, rw, Jc]).tolist()})
    worst = 0.0
    for i, j in enumerate(js):
        assert j["feasible"] == bool(py["feasible"][i])
        if not j["feasible"]:
            assert j["why"] == int(py["why"][i])
            continue
        assert j["za_valid"] == bool(py["za_valid"][i])
        for k in KEYS:
            worst = max(worst, abs(j[k] - py[k][i]) / abs(py[k][i]))
    assert worst < TOL, worst


DECK_CASES = [(14e3, 0.5, 0.2255, 5), (150e3, 120, 9.8, 5), (20e3, 1, 0.5, 5),
              (30e3, 2.0, 0.6, 8), (8e3, 0.3, 0.25, 3)]


def test_deck_text_is_identical_to_his():
    exact = [va.synthesize(*c) for c in DECK_CASES]
    results, theirs = [], []
    for ex in exact:
        results.append({k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                        for k, v in ex.items() if k not in ("raw_inches", "warnings")})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "deck.txt")
            try:
                va.write_deck(ex, p)
                with open(p, encoding="utf-8") as f:
                    theirs.append(f.read())
            except Exception as e:                      # his refusal
                theirs.append(None)
    ours = node({"kind": "deck", "results": results})
    n_compared = 0
    for t, o in zip(theirs, ours):
        if t is None:
            assert o.startswith("REFUSED")
        else:
            assert o == t
            n_compared += 1
    assert n_compared >= 4


def test_calibration_parity():
    with open(os.path.join(ROOT, "reports", "synthetic_roles.json"), encoding="utf-8") as f:
        params = json.load(f)["closure_calibrator"]
    rng = np.random.default_rng(2)
    rows = np.column_stack([rng.uniform(10, 60, 20), rng.uniform(0.3, 3.6, 20),
                            rng.uniform(0.5, 4.4, 20), rng.uniform(5, 300, 20)])
    py = apply_calibration(params, rows[:, 0], rows[:, 1:])
    js = np.array(node({"kind": "cal", "params": params, "rows": rows.tolist()}))
    assert np.max(np.abs(js - py) / py) < 1e-12
