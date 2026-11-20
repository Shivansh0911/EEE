"""
Driver for Dr. Lalit Kumar's PIERCE simulator (src/sim/lktron.py) and the
recorded simulation results.

The run test needs simulation/lktron/target.exe (git-ignored, sir's binary) and
skips without it.
"""

import json
import os
import tempfile

import pytest

from src.ann import dataset as ds
from src.physics import vaughan_adapter as va
from src.sim import experiments as ex
from src.sim import lktron as lk

ROOT = ds.repo_root()
REF = os.path.join(ROOT, "simulation", "lktron", "reference_ceeri")
SIM = os.path.join(ROOT, "reports", "simulation")


def test_parse_sirs_reference_log():
    r = lk.parse_log(os.path.join(REF, "pierce.log"))
    assert r["completed"] and r["converged"]
    assert r["cathode_current_A"] == pytest.approx(121.7416)
    assert r["microperveance"] == pytest.approx(2.096)
    assert r["throw_mm"] == pytest.approx(85.5)
    assert r["waist_mm"] == pytest.approx(7.4336)
    assert r["n_trajectories"] == 57 and r["intercepted_A"] == 0.0
    assert len(r["envelope"]) == 21


def test_parse_sirs_reference_plot():
    p = lk.parse_plt(os.path.join(REF, "pierce.plt"))
    assert len(p[2]) == 57 and len(p[3]) == 25 and len(p[1]) >= 1


def test_vedants_own_deck_lacks_the_cathode_terminator():
    """Recorded defect: his print_gun_points never closes &CATHODE with '/',
    and PIERCE stops with 'End of file'. Our writer adds it."""
    r = va.synthesize(14e3, 0.5, 0.2255, 5)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "deck.txt")
        va.write_deck(r, p)
        last = open(p).read().rstrip().splitlines()[-1]
    assert last.strip() != "/"


def test_our_deck_is_pierce_compatible():
    r = va.synthesize(**ex.GUN)
    rows, meta, dp = ex._vedant_rows(r)
    deck = lk.deck_text(rows, title="T", h=0.5, d=0.5, crad=meta["Rc_mm"], rc=r["rc_mm"])
    lines = deck.rstrip().splitlines()
    assert lines[-1].strip() == "/" and "&CATHODE" in lines[-3]
    assert "DELIN = 0.2" in deck and "imesh= 0" in deck and "NMAX = 2000" in deck
    n = sum(1 for l in lines if l[:2].strip() in ("1", "0", "-1") and "," in l[:4])
    assert f"MBP = {n}," in deck
    assert rows[0]["code"] == 0 and rows[2]["code"] == -1 and rows[6]["code"] == -1
    assert all(rw["volt"] == r["V"] for rw in rows[7:])


def test_duplicate_grid_points_are_merged():
    rows = [{"code": 1, "z": 1.0, "r": 1.0, "volt": 0.0},
            {"code": 1, "z": 1.1, "r": 0.9, "volt": 0.0},      # same 0.5 mm node
            {"code": 1, "z": 3.0, "r": 1.0, "volt": 0.0}]
    deck = lk.deck_text(rows, title="T", h=0.5, d=0.5, crad=10, rc=2)
    assert "MBP = 2," in deck


def test_grid_choice_matches_sirs_deck():
    assert lk.choose_grid(27.65) == 0.5


@pytest.mark.skipif(not lk.available(), reason="target.exe (sir's binary) not present")
def test_rerunning_sirs_deck_reproduces_his_result():
    with open(lk.REFERENCE_DECK, encoding="ascii") as f:
        deck = f.read()
    with tempfile.TemporaryDirectory() as d:
        r = lk.run(deck, d)
    assert r["cathode_current_A"] == pytest.approx(121.7416)
    assert r["throw_mm"] == pytest.approx(85.5)
    assert r["waist_mm"] == pytest.approx(7.4336)


def test_recorded_results_are_consistent():
    s = json.load(open(os.path.join(SIM, "summary.json"), encoding="utf-8"))
    v = s["variants"]
    assert v["reference"]["compare"]["sim_current_A"] == pytest.approx(121.7416)
    for name in ("reference", "ann_m1", "ann_e", "vedant", "tiwary_basu", "psi_57.5", "psi_62.5"):
        assert v[name]["ok"]
    # current rises monotonically with the focus-electrode angle
    cur = [v[n]["compare"]["sim_current_A"] for n in ("psi_57.5", "psi_62.5", "vedant",
                                                       "psi_72.5")]
    assert cur == sorted(cur)
    t1 = json.load(open(os.path.join(SIM, "tuned_psi.json"), encoding="utf-8"))
    assert t1["converged"] and abs(t1["compare"]["current_err_pct"]) < 0.5
    t2 = json.load(open(os.path.join(SIM, "tuned_2d.json"), encoding="utf-8"))["best"]
    assert t2["compare"]["waist_inside_domain"]
    assert abs(t2["compare"]["current_err_pct"]) < 1 and abs(t2["compare"]["waist_err_pct"]) < 3
    assert t2["compare"]["lost_in_gun"] == 0


def test_geometry_from_theta_reproduces_vedant_given_his_angle():
    exact = va.synthesize(**ex.GUN)
    g = ex.geometry_from_theta(exact, exact["thetaT"])
    for k in ("Rc_mm", "Ra_mm", "zac_mm", "za_mm", "zw_mm", "RcT_mm", "ra_mm"):
        assert g[k] == pytest.approx(exact[k], rel=1e-10)
