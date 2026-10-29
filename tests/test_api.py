"""The optimisation API: health, precomputed fronts, and a small live run."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from src.moo.problem import load_study  # noqa: E402

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"ok": True}


def test_studies_listed():
    names = {s["name"] for s in client.get("/studies").json()}
    assert {"shin_eik", "ceeri_klystron", "template"} <= names


def test_pareto_served_with_verification():
    j = client.get("/pareto/shin_eik").json()
    assert len(j["pareto"]) > 0 and j["verify"] and all(v["ok"] for v in j["verify"])


def test_unknown_study_is_404():
    assert client.get("/pareto/nope").status_code == 404


def test_live_optimize_small():
    st = load_study("template")
    st["constraints"]["deck_constructible"] = False      # keep the test quick
    j = client.post("/optimize", json={"study": st, "n_gen": 5, "pop_size": 20}).json()
    assert j["pareto"] and j["backend"] == "lb-closure"


def test_live_optimize_rejects_bad_objective():
    st = load_study("template")
    st["objectives"] = ["Jc", "beauty"]
    assert client.post("/optimize", json={"study": st, "n_gen": 2}).status_code == 422
