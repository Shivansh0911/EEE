"""
The Design tab's PIERCE deck (web/js/pierce.js) must be the deck the Python
writes (src/sim/deck.py), character for character, for both electrode
constructions. Both sides are fed the same exact-solver result, so the test
checks the port, not the emulator. Skips if Node is not installed.
"""

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pytest

from src.ann import dataset as ds
from src.physics import vaughan_adapter as va
from src.sim import deck as simdeck
from src.sim import experiments as ex

ROOT = ds.repo_root()
JS = os.path.join(ROOT, "web", "js", "pierce.js")
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node not installed")

DRIVER = r"""
const P = require(process.argv[2]);
const job = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const out = job.map(j => { const d = P.pierceDeck(j.r, j.opts); return d.text || ('REFUSED ' + d.refused); });
process.stdout.write(JSON.stringify(out));
"""

CASES = [
    ((150e3, 120, 9.8, 5), {"electrodes": "vedant"}),
    ((150e3, 120, 9.8, 5), {"electrodes": "vedant", "psi": 65.656, "anodeShift": 2.265}),
    ((150e3, 120, 9.8, 5), {"electrodes": "tiwary-basu"}),
    ((20e3, 1, 0.5, 5), {"electrodes": "vedant"}),
    ((20e3, 1, 0.5, 5), {"electrodes": "tiwary-basu"}),
]


def jsonable(r):
    return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v)
            for k, v in r.items() if k not in ("raw_inches", "warnings")}


def test_pierce_deck_text_identical():
    saved = (ex.H, ex.D)
    try:
        theirs, jobs = [], []
        for (V, I, rw, Jc), o in CASES:
            deck, exact, _ = simdeck.build_deck(V, I, rw, Jc, o["electrodes"], o.get("psi"),
                                                o.get("anodeShift", 0.0))
            theirs.append(deck)
            jobs.append({"r": jsonable(exact), "opts": o})
    finally:
        ex.H, ex.D = saved
    with tempfile.TemporaryDirectory() as d:
        drv, inp = os.path.join(d, "drv.js"), os.path.join(d, "job.json")
        open(drv, "w").write(DRIVER)
        json.dump(jobs, open(inp, "w"))
        res = subprocess.run(["node", drv, JS, inp], capture_output=True, text=True, check=True)
    ours = json.loads(res.stdout)
    for t, o, c in zip(theirs, ours, CASES):
        assert o == t, f"deck differs for {c}"
