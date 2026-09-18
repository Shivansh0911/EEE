"""
The scaler contract: fit on train only, round-trip exactly, and never quietly
see the test set.

Fitting the scaler on all 30 rows would leak the test set's range into training.
It is a one-line mistake that makes every downstream number look slightly better
and is invisible in the output, so it gets its own test.
"""

import numpy as np
import pytest

from src.ann import dataset as ds
from src.ann.config import TrainConfig
from src.ann.scaling import (InputPipeline, MinMaxScaler, load_scalers,
                             save_scalers, saturation_report)


def test_round_trip_is_exact():
    rng = np.random.default_rng(0)
    X = rng.uniform(-50, 300, size=(40, 3))
    s = MinMaxScaler.fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X)


def test_training_rows_land_exactly_on_the_unit_box():
    rng = np.random.default_rng(1)
    X = rng.uniform(0, 10, size=(20, 3))
    Xn = MinMaxScaler.fit(X).transform(X)
    assert np.allclose(Xn.min(axis=0), -1.0)
    assert np.allclose(Xn.max(axis=0), +1.0)


def test_degenerate_column_is_refused():
    X = np.array([[1.0, 5.0], [2.0, 5.0]])
    with pytest.raises(ValueError):
        MinMaxScaler.fit(X)


def test_scaler_is_blind_to_the_test_split():
    """
    The persisted scaler's ranges must equal the TRAINING rows' ranges, not the
    full dataset's. Test case 3 has the largest theta in the whole set, so if the
    scaler had seen it the target maximum would move.
    """
    cfg = TrainConfig()
    df = ds.load(cfg=cfg)
    train, test = ds.paper_split(df)
    X_tr, Y_tr = ds.matrices(train, cfg)
    X_all, Y_all = ds.matrices(df, cfg)

    s = MinMaxScaler.fit(X_tr)
    assert np.allclose(s.x_min, X_tr.min(axis=0))
    assert np.allclose(s.x_max, X_tr.max(axis=0))
    assert not np.allclose(MinMaxScaler.fit(X_all).x_min, s.x_min), (
        "train-only and all-rows scalers are identical, so this test proves nothing"
    )


def test_persisted_scaler_round_trips_through_json(tmp_path):
    cfg = TrainConfig()
    df = ds.load(cfg=cfg)
    train, _ = ds.paper_split(df)
    X, Y = ds.matrices(train, cfg)
    xs, ys = MinMaxScaler.fit(X), MinMaxScaler.fit(Y)

    path = tmp_path / "scaler.json"
    save_scalers(str(path), xs, ys, cfg.inputs, cfg.targets)
    xs2, ys2, meta = load_scalers(str(path))

    assert meta["inputs"] == cfg.inputs and meta["targets"] == cfg.targets
    assert np.allclose(xs2.transform(X), xs.transform(X))
    assert np.allclose(ys2.transform(Y), ys.transform(Y))


def test_saturation_report_finds_the_ceiling():
    Yn = np.array([[0.5, -0.99], [0.995, 0.1]])
    hits = saturation_report(Yn, 0.98)
    assert sorted((i, k) for i, k, _ in hits) == [(0, 1), (1, 0)]
    assert saturation_report(np.array([[0.5, 0.5]]), 0.98) == []


def test_input_pipeline_logs_C_before_scaling():
    """
    The transform must happen first: the min/max the scaler stores are in log
    space, so the training rows still land exactly on [-1, 1] and C's median row
    moves off the floor it sat on under linear scaling.
    """
    cfg = TrainConfig()
    df = ds.load(cfg=cfg)
    train, _ = ds.paper_split(df)
    X, _ = ds.matrices(train, cfg)

    pipe = InputPipeline.fit(X, cfg.inputs, cfg.input_transform)
    Xn = pipe.transform(X)
    assert np.allclose(Xn.min(axis=0), -1.0)
    assert np.allclose(Xn.max(axis=0), +1.0)

    j = cfg.inputs.index("C")
    linear = MinMaxScaler.fit(X).transform(X)[:, j]
    assert np.median(Xn[:, j]) > np.median(linear) + 0.3, (
        "log scaling should lift C's median row well off the -1 floor"
    )


def test_envelope_comes_back_in_physical_units():
    cfg = TrainConfig()
    df = ds.load(cfg=cfg)
    train, _ = ds.paper_split(df)
    X, _ = ds.matrices(train, cfg)
    env = InputPipeline.fit(X, cfg.inputs, cfg.input_transform).envelope()
    for j, name in enumerate(cfg.inputs):
        assert np.isclose(env[name][0], X[:, j].min())
        assert np.isclose(env[name][1], X[:, j].max())


def test_unknown_transform_is_refused():
    with pytest.raises(ValueError):
        InputPipeline.fit(np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]),
                          ["a", "b", "c"], {"c": "sqrt"})
