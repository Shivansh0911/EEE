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
from src.ann.scaling import MinMaxScaler
from src.export.weights_to_json import build_payload

MODEL_PATH = os.path.join(ds.repo_root(), "models", "model_m1.json")

REQUIRED_KEYS = {
    "schema_version", "architecture", "activation", "output_activation",
    "weights", "scaler", "targets", "training_envelope", "metrics", "provenance",
}


@pytest.fixture(scope="module")
def payload():
    cfg = TrainConfig()
    rng = np.random.default_rng(3)
    theta = nguyen_widrow(cfg.layer_sizes, rng)
    X = rng.uniform(0.3, 300.0, size=(12, cfg.n_in))
    Y = rng.uniform(5.0, 70.0, size=(12, cfg.n_out))
    xs, ys = MinMaxScaler.fit(X), MinMaxScaler.fit(Y)
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


def test_forward_pass_rebuilt_from_json_matches_the_python_network(payload):
    """
    The whole point of the file: someone reading only the JSON must get the same
    answer. Tolerance is 1e-12, far tighter than the 1e-9 the JS parity test
    will use, because nothing here crosses a language boundary.
    """
    p, theta, cfg, xs, ys = payload
    rng = np.random.default_rng(11)
    X = rng.uniform(0.3, 300.0, size=(50, 3))

    Xn = 2 * (X - np.array(p["scaler"]["x_min"])) / (
        np.array(p["scaler"]["x_max"]) - np.array(p["scaler"]["x_min"])) - 1
    a = Xn
    n_layers = len(p["architecture"]) - 1
    for i in range(1, n_layers + 1):
        a = np.tanh(a @ np.array(p["weights"][f"W{i}"]) + np.array(p["weights"][f"b{i}"]))

    expected = predict(theta, xs.transform(X), cfg)
    assert np.max(np.abs(a - expected)) < 1e-12


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="M1 not exported yet")
def test_shipped_model_is_well_formed_and_honest():
    with open(MODEL_PATH, encoding="utf-8") as f:
        m = json.load(f)
    assert REQUIRED_KEYS <= set(m)
    assert m["provenance"]["tier_b_rows"] == 0, (
        "the shipped model claims synthetic training rows, but Tier B is gated "
        "on a physics validation that fails"
    )
    assert m["provenance"]["n_train"] == 23 and m["provenance"]["n_test"] == 7
    assert m["targets"] == ["theta_cone_deg", "Rc_mm"]
    for lo, hi in m["training_envelope"].values():
        assert lo < hi
