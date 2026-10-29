"""
The optimisation API (Render). Prediction and single-design synthesis run in the
browser; only the Pareto search needs a server.

    uvicorn api.app:app --port 8000          # from the repo root

GET  /health              liveness, used by the site's wake-up probe
GET  /studies             the study configs shipped in src/moo/studies
GET  /pareto/{study}      precomputed front, knee, and Gate G3 verification
POST /optimize            a live NSGA-II run on a posted study config
                          (generations capped so a free instance answers in time)
"""

import json
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.moo.problem import GEOM, VARS, knee_index, load_study

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "src", "moo", "studies")
MAX_GEN_LIVE = 120

app = FastAPI(title="Pierce gun design API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"],
                   allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/studies")
def studies():
    return [load_study(f[:-5]) for f in sorted(os.listdir(STUDIES)) if f.endswith(".json")]


def _csv_records(path):
    import csv
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _num(v):
    try:
        return float(v)
    except ValueError:
        return {"True": True, "False": False}.get(v, v)


@app.get("/pareto/{study}")
def pareto(study: str):
    d = os.path.join(ROOT, "reports", "moo", os.path.basename(study))
    if not os.path.exists(os.path.join(d, "pareto.csv")):
        raise HTTPException(404, f"no results for study {study!r}")
    with open(os.path.join(d, "knee.json"), encoding="utf-8") as f:
        knee = json.load(f)
    ver = os.path.join(d, "verify.csv")
    return {"study": load_study(os.path.basename(study)), "pareto": _csv_records(os.path.join(d, "pareto.csv")),
            "knee": knee, "verify": _csv_records(ver) if os.path.exists(ver) else None}


class OptimizeRequest(BaseModel):
    study: dict
    n_gen: int = 60
    pop_size: int = 100
    seed: int = 1


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize
    from src.moo.problem import GunProblem, evaluate
    st = req.study
    for key in ("I", "bounds", "objectives", "constraints"):
        if key not in st:
            raise HTTPException(422, f"study config is missing {key!r}")
    n_gen = min(req.n_gen, MAX_GEN_LIVE)
    try:
        res = minimize(GunProblem(st), NSGA2(pop_size=min(req.pop_size, 200)),
                       ("n_gen", n_gen), seed=req.seed, verbose=False)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if res.X is None:
        return {"pareto": [], "note": "no feasible design; loosen the constraints"}
    X = np.atleast_2d(res.X)
    F, _, r = evaluate(X, st)
    ki = knee_index(F)
    rows = []
    for i in range(len(X)):
        row = {v: float(X[i, j]) for j, v in enumerate(VARS)}
        row.update({"obj_" + n: float(F[i, j]) for j, n in enumerate(st["objectives"])})
        row.update({k: (bool(r[k][i]) if k == "za_valid" else float(r[k][i]))
                    for k in GEOM})
        rows.append(row)
    return {"pareto": rows, "knee_index": ki, "n_gen": n_gen,
            "evaluations": int(res.algorithm.evaluator.n_eval), "backend": "lb-closure",
            "note": "emulator results; verify with `python -m src.moo.verify` before building"}
