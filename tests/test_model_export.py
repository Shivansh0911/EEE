"""
The exported model JSON is a contract, so it is tested like one.

B.3 of the master build document specifies exactly what the browser will read.
If a key goes missing or a matrix is transposed, the failure surfaces in the
browser as a plausible-looking wrong angle during a demo rather than as an
exception -- which is the worst kind of bug to have. So the shape is asserted
here, and a forward pass rebuilt purely from the JSON is checked against the
Python network it was exported from.

The JS/Python parity test (P7/P8) will run the same check against Node.
"""

import json
import os

import numpy as np
import pytest

from src.ann import dataset as ds
from src.ann.config import TrainConfig
from src.ann.network import n_params, nguyen_widrow, predict
from src.ann.scaling import InputPipeline, MinMaxScaler
from src.export.weights_to_json import build_payload

MODEL_PATH = os.path.join(ds.repo_root(), "models", "model_m1.json")
MULTI_PATH = os.path.join(ds.repo_root(), "models", "model_m1_multi.json")

REQUIRED_KEYS = {
    "schema_version", "architecture", "activation", "output_activation",
    "weights", "scaler", "targets", "training_envelope", "metrics", "provenance",
    "input_transform", "inputs",
}


@pytest.fixture(scope="module")
def payload():
    cfg = TrainConfig()
    rng = np.random.default_rng(3)
    theta = nguyen_widrow(cfg.layer_sizes, rng)
    X = rng.uniform(0.3, 300.0, size=(12, cfg.n_in))
    Y = rng.uniform(5.0, 70.0, size=(12, cfg.n_out))
    xs = InputPipeline.fit(X, cfg.inputs, cfg.input_transform)
    ys = MinMaxScaler.fit(Y)
    p = build_payload(theta, cfg, xs, ys, {"test_mre_pct": {}}, {"model_id": "TEST"})
    return p, theta, cfg, xs, ys


def test_payload_has_every_key_the_browser_reads(payload):
    p, *_ = payload
    assert REQUIRED_KEYS <= set(p)


def test_architecture_and_weight_shapes_agree(payload):
    p, theta, cfg, *_ = payload
    assert p["architecture"] == [3, 3, 2, 2]
    assert sum(np.size(np.array(v)) for v in p["weights"].values()) == n_params(
        cfg.layer_sizes) == 26
    for i, (a, b) in enumerate(zip(p["architecture"], p["architecture"][1:]), start=1):
        assert np.array(p["weights"][f"W{i}"]).shape == (a, b)
        assert np.array(p["weights"][f"b{i}"]).shape == (b,)


def test_activation_names_are_the_javascript_ones(payload):
    p, *_ = payload
    assert p["activation"] == "tanh" and p["output_activation"] == "tanh"


def test_input_transform_is_exported(payload):
    """
    The scaler's x ranges are in transformed space, so a consumer that applied
    the scaler without the transform would get plausible, wrong numbers. The
    transform has to travel with the model.
    """
    p, _, cfg, *_ = payload
    assert p["input_transform"] == {"C": "log"}
    assert p["inputs"] == cfg.inputs


def test_forward_pass_rebuilt_from_json_matches_the_python_network(payload):
    """
    The whole point of the file: someone reading only the JSON must get the same
    answer, INCLUDING the input transform. Tolerance is 1e-12, far tighter than
    the 1e-9 the JS parity test uses, because nothing here crosses a language
    boundary.
    """
    p, theta, cfg, xs, ys = payload
    rng = np.random.default_rng(11)
    X = rng.uniform(0.3, 300.0, size=(50, 3))

    Xt = X.copy()
    for name, kind in p["input_transform"].items():
        assert kind == "log"
        Xt[:, p["inputs"].index(name)] = np.log(Xt[:, p["inputs"].index(name)])
    Xn = 2 * (Xt - np.array(p["scaler"]["x_min"])) / (
        np.array(p["scaler"]["x_max"]) - np.array(p["scaler"]["x_min"])) - 1
    a = Xn
    n_layers = len(p["architecture"]) - 1
    for i in range(1, n_layers + 1):
        a = np.tanh(a @ np.array(p["weights"][f"W{i}"]) + np.array(p["weights"][f"b{i}"]))

    expected = predict(theta, xs.transform(X), cfg)
    assert np.max(np.abs(a - expected)) < 1e-12


@pytest.mark.parametrize("path,targets", [
    (MODEL_PATH, ["theta_cone_deg"]),
    (MULTI_PATH, ["theta_cone_deg", "Rc_mm"]),
])
def test_shipped_models_are_well_formed_and_honest(path, targets):
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not exported yet")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    assert REQUIRED_KEYS <= set(m)
    assert m["provenance"]["tier_b_rows"] == 0, (
        "the shipped model claims synthetic training rows, but Tier B is gated "
        "on a physics validation that fails"
    )
    assert m["provenance"]["n_train"] == 23 and m["provenance"]["n_test"] == 7
    assert m["targets"] == targets
    assert m["input_transform"] == {"C": "log"}
    for lo, hi in m["training_envelope"].values():
        assert lo < hi
    # The distribution, not just the lucky draw, must reach the browser.
    d = m["metrics"]["restart_distribution"]
    assert d["test_mre_pct_median"] is not None
    assert d["test_mre_pct_min"] <= d["test_mre_pct_median"] <= d["test_mre_pct_max"]
