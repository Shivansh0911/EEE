"""
Our ANN reproduces Satwik's Colab forward pass, with his weights and his
normalisation.

His notebook (notebooks/satwik_colab_reference.ipynb, kept untouched) saves no
outputs, so his trained weights exist nowhere on disk. They are recreated here
by executing his own code cells, in order, on his own data file
(data/raw/ann_testing_original.csv, the "ANN testing - Sheet1.csv" he uploads).
Only three cells are skipped: the google.colab upload (replaced by reading the
same CSV), the matplotlib plot, and the interactive input() prediction.

What is the same:   3-3-2-1 tanh network, tanh output, [-1, 1] min-max scaling
                    fitted on training rows, parameter packing W1 b1 W2 b2 W3 b3.
What differs, and is deliberate:
  * split   -- he trains on the first 12 rows and tests on the rest (3 of 14);
               we use Panahi's 23/7 split of the 30-row Table 2 (D1, P5).
  * Jacobian -- his is forward-difference (eps 1e-5, 26 extra forward passes
               per epoch); ours is analytic (src/ann/jacobian.py). Checked
               below: they agree to finite-difference accuracy.
  * restarts -- he runs one seed (100); we run Nguyen-Widrow restarts.
  * C       -- he scales C linearly; our default log-scales it first (D8).
"""

import contextlib
import io
import json
import os

import numpy as np
import pandas as pd
import pytest

from src.ann import dataset as ds
from src.ann.config import TrainConfig
from src.ann.jacobian import analytic_jacobian
from src.ann.network import forward
from src.ann.scaling import MinMaxScaler

ROOT = ds.repo_root()
NB = os.path.join(ROOT, "notebooks", "satwik_colab_reference.ipynb")
CSV = os.path.join(ROOT, "data", "raw", "ann_testing_original.csv")


@pytest.fixture(scope="module")
def his():
    with open(NB, encoding="utf-8") as f:
        cells = [c for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    ns = {"df": pd.read_csv(CSV)}
    with contextlib.redirect_stdout(io.StringIO()):
        exec("import pandas as pd\nimport numpy as np", ns)
        for c in cells:
            src = "".join(c["source"])
            if "google.colab" in src or "plt." in src or "input(" in src:
                continue
            exec(src, ns)
    return ns


def test_his_training_ran(his):
    assert his["params"].size == 23           # 3*3+3 + 3*2+2 + 2*1+1
    assert his["X_train"].shape == (12, 3)
    assert len(his["mse_history"]) > 0


def test_our_forward_pass_equals_his(his):
    cfg = TrainConfig(targets=["theta_cone_deg"], input_transform={})
    for Xn in (his["X_train_norm"], his["X_test_norm"]):
        theirs = his["forward"](Xn, *his["unpack_params"](his["params"]))
        ours = forward(his["params"], Xn, cfg.layer_sizes)
        assert np.max(np.abs(ours - theirs)) < 1e-14


def test_our_scaler_equals_his_normalisation(his):
    xs = MinMaxScaler.fit(his["X_train"])
    assert np.allclose(xs.transform(his["X_test"]), his["X_test_norm"], atol=1e-15)
    ys = MinMaxScaler.fit(his["y_train"][:, None])
    assert np.allclose(ys.transform(his["y_train"][:, None])[:, 0], his["y_train_norm"])


def test_our_denormalised_prediction_equals_his(his):
    cfg = TrainConfig(targets=["theta_cone_deg"], input_transform={})
    ys = MinMaxScaler.fit(his["y_train"][:, None])
    Xn = his["X_test_norm"]
    theirs = ((his["forward"](Xn, *his["unpack_params"](his["params"])) + 1) / 2
              * (his["y_max"] - his["y_min"]) + his["y_min"])
    ours = ys.inverse_transform(forward(his["params"], Xn, cfg.layer_sizes))
    assert np.max(np.abs(ours - theirs)) < 1e-12


def test_his_numeric_jacobian_matches_our_analytic_one(his):
    cfg = TrainConfig(targets=["theta_cone_deg"], input_transform={})
    Xn, yn = his["X_train_norm"], his["y_train_norm"]
    Jhis = his["calculate_jacobian"](his["params"], Xn, yn)
    Jours = analytic_jacobian(his["params"], Xn, yn[:, None], cfg)
    assert Jhis.shape == Jours.shape
    # Forward difference, eps = 1e-5: truncation error O(eps * f''), which grows
    # with his trained weights, so compare in norm relative to the Jacobian.
    rel = np.linalg.norm(Jhis - Jours) / np.linalg.norm(Jours)
    assert rel < 1e-3
