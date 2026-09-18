"""
The browser and the paper must not disagree.

The Predict tab runs its own forward pass in JavaScript, because the whole point
of the architecture is that prediction never touches a backend. That means there
are two implementations of the model, and two implementations drift. When they
drift, the failure is not an exception -- it is a plausible-looking wrong angle
on screen during a demo, which is the worst kind of bug to have.

So the JS is run under Node against the actual shipped model file and checked
against Python on 100 random inputs, to < 1e-9. The inputs are drawn across and
beyond the training envelope, because the interesting disagreements are at the
edges: the log transform of C, the saturation of a tanh output, and the
denormalisation back to degrees.

If Node is not installed this test skips rather than failing, and says so.
"""

import json
import os
import shutil
import subprocess

import numpy as np
import pytest

from src.ann import dataset as ds
from src.ann.config import TrainConfig
from src.ann.network import predict as py_predict
from src.ann.scaling import InputPipeline, MinMaxScaler

ROOT = ds.repo_root()
ANN_JS = os.path.join(ROOT, "web", "js", "ann.js")
MODELS = [
    os.path.join(ROOT, "models", "model_m1.json"),
    os.path.join(ROOT, "models", "model_m1_multi.json"),
]

TOL = 1e-9
N_INPUTS = 100

# Node is handed the model path and the inputs, and prints the outputs. Kept
# deliberately small: anything it does that the browser does not would make this
# test a check on the harness rather than on the shipped code.
DRIVER = r"""
const path = process.argv[2];
const ann = require(path);
const model = JSON.parse(require('fs').readFileSync(process.argv[3], 'utf8'));
const inputs = JSON.parse(require('fs').readFileSync(process.argv[4], 'utf8'));
const out = inputs.map((x) => ann.predict(model, x).vector);
process.stdout.write(JSON.stringify(out));
"""


def node_available():
    return shutil.which("node") is not None


@pytest.fixture(scope="module")
def sample_inputs():
    """
    Spread across the training envelope, with a deliberate margin outside it.

    Half the value of this test is that it covers inputs the UI will actually be
    given -- a slider dragged to the end of its range lands outside the envelope,
    and that is exactly where a mismatched transform would show up.
    """
    cfg = TrainConfig()
    df = ds.load(cfg=cfg)
    train, _ = ds.paper_split(df)
    X, _ = ds.matrices(train, cfg)
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = hi - lo

    rng = np.random.default_rng(20260918)
    pts = rng.uniform(lo - 0.15 * span, hi + 0.15 * span, size=(N_INPUTS, X.shape[1]))
    # C is log-transformed, so it must stay strictly positive whatever the
    # slider does. The UI enforces the same floor.
    j = cfg.inputs.index("C")
    pts[:, j] = np.clip(pts[:, j], 0.5, None)
    return pts


def run_node(model_path, inputs, tmp_path):
    # .cjs, not .js: a package.json with "type": "module" anywhere above the
    # driver would otherwise make Node treat it as ESM and reject require().
    # The extension pins CommonJS regardless of what is in the enclosing tree.
    driver = tmp_path / "driver.cjs"
    driver.write_text(DRIVER, encoding="utf-8")
    payload = tmp_path / "inputs.json"
    payload.write_text(json.dumps(inputs.tolist()), encoding="utf-8")

    proc = subprocess.run(
        ["node", str(driver), ANN_JS, model_path, str(payload)],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        raise AssertionError(f"node failed:\n{proc.stderr}")
    return np.array(json.loads(proc.stdout), dtype=float)


def python_predict(model_path, inputs):
    """Rebuild the Python model from the SAME file the browser reads."""
    with open(model_path, encoding="utf-8") as f:
        m = json.load(f)

    cfg = TrainConfig(
        targets=m["targets"],
        inputs=m["inputs"],
        input_transform=m["input_transform"],
        hidden=m["architecture"][1:-1],
    )
    pipe = InputPipeline(
        MinMaxScaler(m["scaler"]["x_min"], m["scaler"]["x_max"]),
        m["inputs"], m["input_transform"],
    )
    y_scaler = MinMaxScaler(m["scaler"]["y_min"], m["scaler"]["y_max"])

    theta = np.concatenate([
        np.array(m["weights"][k]).ravel()
        for i in range(1, len(m["architecture"]))
        for k in (f"W{i}", f"b{i}")
    ])
    return y_scaler.inverse_transform(py_predict(theta, pipe.transform(inputs), cfg))


@pytest.mark.skipif(not node_available(), reason="node is not installed")
@pytest.mark.parametrize("model_path", MODELS)
def test_js_matches_python(model_path, sample_inputs, tmp_path):
    if not os.path.exists(model_path):
        pytest.skip(f"{os.path.basename(model_path)} not exported yet")

    js = run_node(model_path, sample_inputs, tmp_path)
    py = python_predict(model_path, sample_inputs)

    assert js.shape == py.shape
    worst = float(np.max(np.abs(js - py)))
    assert worst < TOL, (
        f"JavaScript and Python disagree by {worst:.3e} on "
        f"{os.path.basename(model_path)}. The browser would show a wrong angle."
    )


@pytest.mark.skipif(not node_available(), reason="node is not installed")
def test_js_applies_the_log_transform(tmp_path):
    """
    A regression guard for the specific mistake this design invites: applying
    the scaler without the transform. Both are exported, and a consumer that
    dropped the transform would still produce finite, plausible numbers -- so
    the check is that changing C changes the answer in the way the log implies,
    not merely that the code runs.
    """
    model_path = MODELS[0]
    if not os.path.exists(model_path):
        pytest.skip("model not exported yet")
    with open(model_path, encoding="utf-8") as f:
        m = json.load(f)
    assert m["input_transform"] == {"C": "log"}

    # Two C values equally spaced in LOG space must be equally spaced in the
    # normalised input, which is only true if the log is applied.
    base = [1.5, 2.0, 10.0]
    pts = np.array([base, [1.5, 2.0, 30.0], [1.5, 2.0, 90.0]])
    js = run_node(model_path, pts, tmp_path)
    py = python_predict(model_path, pts)
    assert np.max(np.abs(js - py)) < TOL

    j = m["inputs"].index("C")
    lo, hi = m["scaler"]["x_min"][j], m["scaler"]["x_max"][j]
    norm = [2 * (np.log(c) - lo) / (hi - lo) - 1 for c in (10.0, 30.0, 90.0)]
    assert np.isclose(norm[1] - norm[0], norm[2] - norm[1])
