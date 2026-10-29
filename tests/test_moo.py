"""
Optimiser, Gate G3, and the deck CLI.
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from src.ann import dataset as ds
from src.moo import verify
from src.moo.problem import GunProblem, evaluate, knee_index, load_study
from src.physics import vaughan_adapter as va

ROOT = ds.repo_root()
STUDIES = ("shin_eik", "ceeri_klystron")


@pytest.mark.parametrize("name", STUDIES + ("template",))
def test_study_configs_load_and_evaluate(name):
    st = load_study(name)
    b = st["bounds"]
    X = np.array([[np.mean(b[v]) for v in ("V", "rw_mm", "Jc")]])
    F, G, r = evaluate(X, st)
    assert F.shape == (1, len(st["objectives"]))
    assert GunProblem(st).n_ieq_constr == G.shape[1]


def test_published_designs_satisfy_their_study_constraints():
    for name in STUDIES:
        st = load_study(name)
        ref = st["reference_design"]
        _, G, _ = evaluate(np.array([[ref["V"], ref["rw_mm"], ref["Jc"]]]), st)
        assert (G <= 0).all(), name


def test_infeasible_design_violates_and_is_penalised():
    st = load_study("shin_eik")
    F, G, r = evaluate(np.array([[14e3, 3.0, 50.0]]), st)     # rc < rw
    assert not r["feasible"][0]
    assert G[0, 0] > 0 and (F >= 1e8).all()


def test_aperture_constraint_rejects_theta_raw_above_56_44():
    st = load_study("shin_eik")
    st["bounds"] = {"V": [5e3, 60e3], "rw_mm": [0.1, 1], "Jc": [1, 40]}
    st["I"] = 30.0
    F, G, r = evaluate(np.array([[50e3, 0.2, 20.0]]), st)
    assert r["feasible"][0] and not r["za_valid"][0]
    assert (G > 0).any()


def test_unknown_objective_is_refused():
    st = load_study("template")
    st["objectives"] = ["Jc", "beauty"]
    with pytest.raises(ValueError):
        evaluate(np.array([[20e3, 0.8, 4.0]]), st)


def test_knee_rule_is_min_distance_to_normalised_ideal():
    F = np.array([[0.0, 10.0], [4.0, 4.0], [10.0, 0.0], [6.0, 6.0]])
    assert knee_index(F) == 1


@pytest.mark.parametrize("name", STUDIES)
def test_study_outputs_exist_and_are_feasible(name):
    d = os.path.join(ROOT, "reports", "moo", name)
    for f in ("pareto.csv", "vaughan_inputs.csv", "verify.csv", "knee.json", "pareto.png",
              "run.json"):
        assert os.path.exists(os.path.join(d, f)), f
    run = json.load(open(os.path.join(d, "run.json"), encoding="utf-8"))
    assert run["evaluations"] >= 50000
    front = pd.read_csv(os.path.join(d, "pareto.csv"))
    st = load_study(name)
    _, G, _ = evaluate(front[["V", "rw_mm", "Jc"]].to_numpy(), st)
    assert (G <= 1e-9).all()
    knee = json.load(open(os.path.join(d, "knee.json"), encoding="utf-8"))
    assert set(knee["sensitivity_pm5pct"]) == {f"{v}{s}5%" for v in ("V", "rw_mm", "Jc")
                                               for s in "+-"}
    assert knee["expected_if_measured"]["band_pct"] > 0


@pytest.mark.parametrize("name", STUDIES)
def test_gate_g3_verified_designs(name):
    v = pd.read_csv(os.path.join(ROOT, "reports", "moo", name, "verify.csv"))
    assert len(v) >= 10 and v.is_knee.any()
    assert v.feasible.all() and v.ok.all()
    assert v.max_err.max() <= 0.02


def test_gate_g3_fails_on_a_wrong_prediction():
    d = {"V": 14e3, "I": 0.5, "rw_mm": 0.2255, "Jc": 5.0,
         "theta": 14.7048, "thetaT": 16.2891 * 1.05, "Rc_mm": 7.0286, "zac_mm": 4.2528,
         "zw_mm": 12.0923}
    out = va.verify([d], tol=0.02, keys=verify.KEYS, workers=1)
    assert not out[0]["ok"]


def test_deck_cli_refuses_above_56_44_deg(tmp_path):
    out = tmp_path / "deck.txt"
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.moo.deck", "--V", "50000",
                        "--I", "30", "--rw", "0.2", "--Jc", "20", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 2 and "56.44" in p.stderr
    assert not out.exists()


def test_deck_cli_writes_his_deck(tmp_path):
    out = tmp_path / "deck.txt"
    p = subprocess.run([sys.executable, "-W", "ignore", "-m", "src.moo.deck", "--V", "14000",
                        "--I", "0.5", "--rw", "0.2255", "--Jc", "5", "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    text = out.read_text()
    assert text.startswith(" 'PIERCE GUN TEST'") and "&CATHODE" in text


@pytest.mark.parametrize("name", STUDIES)
def test_knee_deck_exists(name):
    d = os.path.join(ROOT, "reports", "moo", name)
    knee = json.load(open(os.path.join(d, "knee.json"), encoding="utf-8"))
    assert os.path.exists(os.path.join(d, f"egun_deck_{knee['index_in_pareto_csv']}.txt"))
