# Our project (EEE F266, supervisor Lalit Kumar) — short context

Full text: `midsem_report.md`. Repo: https://github.com/Shivansh0911/EEE.git (read README.md, 00_MASTER_BUILD_PROMPTS.md, reports/DECISIONS.md, BENCHMARK.md first).
- Scope: pencil beams (solid, axisymmetric). Chain gun → magnetic focusing → collector; we do the gun.
- Replicates and extends Panahi et al., Results in Physics 73 (2025) 108256: 3-3-2-x tanh ANN, Levenberg–Marquardt, inputs (P, r_w, C) → half-beam cone angle θ.
- Tier A: 30 real guns from the paper's Table 2 (23 train / 7 test). Tier B: synthetic (generated, 1000 rows, C≥8) — it HURT (M3 9.77 % vs M1 7.93 %) because the generator carried ~10 % bias vs measurement.
- Honest results: M1 median test MRE 8.33 % (paper claims 3.74 %, not reproducible); classical Vaughan is 10.1 % vs measured.
- Satwik's Colab notebook (`satwik_colab.ipynb`): simplified 3-3-2-1 ANN, [-1,1] scaling, LM with finite-difference Jacobian, first 12 rows train. Our `src/ann/` supersedes it (analytic Jacobian, 23/7 split, restarts). Keep his notebook as a reference, verify equivalence.
- `already_built_integration_branch.zip`: the integration I (Claude) already built — lb.py, kf.py, LBClosure emulator, adapter to Vedant's solver, NSGA-II with throw/aperture, verify gate, 94 passing tests, Pareto results for Shin EIK and CEERI klystron. START FROM IT (unzip over the repo on a new branch) rather than rewriting; review it critically.
