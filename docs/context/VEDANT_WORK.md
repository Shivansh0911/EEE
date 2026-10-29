# What Vedant built (last semester) — and why it matters

Source files in this folder: `vedant_code/pierce_gun_design_fixed.py` (1559 lines, the final solver) and `vedant_chat_full.txt` (his whole Claude session, 146 KB: bug fixes, validations, HTML port, EGUN deck). Treat the chat as history/evidence; the code is the truth.

## One line
A validated, interactive **Pierce electron-gun synthesis tool**: give it (V, I, r_w, Jc) → it returns the complete gun geometry and an EGUN-ready point deck. It is the *exact classical method* that our ANN approximates.

## Method (all in the code header)
- Vaughan (1981) "Synthesis of the Pierce gun" + Yang-Jia-Zhu (2006) amendments + Liao (2003) starting angle.
- Perveance μP = I/V^1.5, cathode radius rc = sqrt(I/(π Jc)), convergence C = (rc/rw)².
- Langmuir–Blodgett spherical flow (series α(γ)); half-angle θ from a **bracketed bisection** (`solve_theta`) with Yang's damped iteration (`yang_iterate`) as an **independent cross-check** (warns if they differ > 0.1°).
- Beam-spread constant 0.17409 (derived from physical constants, not fitted).
- **Relativistic diode factor kf** (`relativistic_factor`): −0.2 % at 14 kV, −2 % at 150 kV, −6 % at 500 kV.
- Aberration compensation: built cathode radius RcT = 0.905·Rc.
- Anode aperture ra = 1.2·rb(za); Eq.20 for za becomes insoluble above θ = 56.44° → he flags `za_valid=False` instead of failing (θ, Rc, Ra, z_ac, throw stay valid). Raises `GunInfeasible` for impossible inputs (rc ≤ rw, α outside series validity).
- Throw zw (Eq.12/21), beam-edge table, drift-tube clearance.
- Pierce 67.5° focus electrode construction (rebuilt after the old version couldn't reach 67.5°); EGUN point deck with integer-grid snapping (`compute_gun_points_mm`, `print_gun_points`), synthesis report, wireframe plot, parallel Jc sweep (`run_sweep`, CSV + 4-panel PNG).
- A browser (HTML/JS) port checked digit-for-digit against the Python.

## Validation he did (quote these)
- Kant & Nagaraju 2021, 8 MW S-band CEERI klystron gun: all 10 tabulated outputs within 0.51 %.
- Ardebili et al. 2025 Table 1: nine guns, μP 1.90–3.67, C 5.33–306: mean deviation 0.26° (aberration-corrected angle).
- 180-point random sweep: 99 initially bad → fixed validity limits (series valid for −α ≤ 2.8, rc > rw required); regression: 85 identical, 50 fixed, 45 explicitly rejected, 0 regressions.
- Beam-level check on the SLAC 5045 klystron gun (350 kV, 414 A): perveance and throw match; limits noted — it produces *a* valid Pierce gun, not designs relying on non-Pierce electrode shaping.
- Shin et al. 35 GHz EIK case (14 kV, 0.5 A, rw 0.2255 mm, 5 A/cm²) reproduced; comparisons to IECAS Ka EIK and CEERI/NIT MB-EIK.

## What he did NOT have (our opening)
1. **No optimisation** — one design in, one design out; the 180-point sweep is only a 1-D scan of Jc.
2. **Slow** (~1.8 s/design with mpmath) → a real 50 000-design search is ~25 h.
3. **No machine-learning surrogate**, no data, no uncertainty statement vs measurement.
4. Anode aperture empirical (Sharma 2001 not implemented, paywalled).
5. θ semantics: his tool reports the compensated angle and the raw angle; our ANN's targets are the paper's Table-2 angle (= compensated). Keep them distinct.

## What we have that he lacks
ANN surrogate (Panahi 2025, replicated and honestly benchmarked), 3-tier dataset with provenance, NSGA-II, a deployed web tool, a physics finding (LB constant 14.665 not 29.33; lens coefficient 0.417 not 0.62), and an honest error floor (all models 7–9 % vs measured; Vaughan itself 10.1 %).

## How the combination shines (the story for the supervisor)
Vedant = **exact analysis** (slow, trusted, buildable output). Ours = **fast surrogate + search**. Together: the optimiser explores 50 000 designs in ~20 s using a surrogate that reproduces his solver to ~0.005 % (θ) / 0.02 % (lengths); the top-10 are *re-verified by his exact solver* (the P6 gate); the winner leaves as an EGUN deck a gun designer can simulate. That closes the loop: spec in → Pareto front → verified design → manufacturable deck.
