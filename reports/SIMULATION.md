# Simulating synthesised guns in sir's PIERCE, and the two focus-electrode papers

Numbers below come from `reports/SIMULATION_RESULTS.md`, `reports/simulation/*.json`
(written by `python -m src.sim.experiments` and `python -m src.sim.report`) and the
tests named next to them.

## 1. What `target.exe` is

`simulation/lktron/target.exe` is **LKTRON PIERCE**, Dr. Lalit Kumar's gun code
(banner: "@Copyright Dr. LALIT KUMAR", version 29 Oct 2026). It reads `pierce.in`,
solves Poisson's equation on an axisymmetric grid, launches Child-law trajectories
from the cathode and iterates to self-consistency, then writes `pierce.log`
(current, perveance, beam envelope, throw, waist, interception), `pierce.plt`
(electrodes, trajectories, equipotentials) and `pierce.3D` (potential and field
maps). Sir's `pierce_plotter.py` and `new3d.py` draw those two files.

Its input format **is Vedant's point deck**: rows of `code, z, r, V, z_c, r_c,
radius` followed by `&CATHODE`. That is the sense in which Vedant's code "implements
target.exe": his synthesis writes the boundary that PIERCE simulates.

The binary is sir's and is git-ignored (the repository is public). Everything in
`src/sim` finds it at `simulation/lktron/target.exe`.

## 2. Running our decks in it: three things had to be fixed

| Problem | Symptom in PIERCE | Fix (`src/sim/lktron.py`) |
|---|---|---|
| Vedant's `print_gun_points` never closes `&CATHODE` with `/` | `Fortran runtime error: End of file` | the deck writer adds it (`tests/test_lktron.py` records his omission) |
| Vedant's `&CONTROL/&PRINTS/&PLOT` values | crash re-reading its own `pierce.plt` ("Bad value during integer read") | use the values from sir's reference deck |
| PIERCE allows at most 500 mesh intervals along z | `FATAL ERROR : THE PROBLEM SIZE EXCEEDS THE LIMIT` | grid chosen from the cathode radius and coarsened to fit the limit |

Two choices of ours: the drift tube runs to 2 x the designed throw (Vedant's
1.3 x ends the domain before PIERCE's waist in most runs, and a "waist" on the end
wall is not a waist), and interceptions are split by where the trajectory ends,
before the waist (a gun defect) or after it (the expanding beam scraping the tube,
which a magnet prevents in a real device).

**Check:** re-running sir's own deck through the driver reproduces his run exactly,
121.7416 A, throw 85.5 mm, waist 7.4336 mm (`tests/test_lktron.py`, live run).

## 3. Results on the CEERI gun (150 kV, 120 A, waist 9.8 mm, 5 A/cm²)

CEERI is used because it is the gun sir's reference deck simulates. Grid 0.5 mm,
as in his deck.

| Run | Current vs 120 A | Lost in gun | Waist vs 9.8 mm |
|---|---|---|---|
| Sir's reference deck | +1.5 % | 0 / 57 | −24.4 % |
| Vedant's deck as built (ψ 67.5°, corrected cathode R_cT) | +26.5 % | 1 / 58 | +23.0 % |
| Vedant's construction, uncorrected cathode R_c | +40.4 % | 19 / 57 | +24.7 % |
| Tiwary–Basu electrodes | −14.5 % | 0 / 57 | −33.6 % |
| ψ 57.5° / 62.5° / 72.5° | +1.0 / +13.4 / +38.4 % | 0 / 0 / 32 | −33.2 / −3.1 % / no waist |
| ψ tuned to the current (57.08°) | **+0.2 %** | 0 / 58 | −34.7 % |
| ψ 65.66° and anode +2.27 mm, tuned | **−0.7 %** | 0 / 58 | **−1.9 %** |

What it says:

1. **Vedant's deck, as synthesised, draws 27 % too much current in a field solve.**
   The synthesis equations do not see the focus-electrode field in detail; the
   simulator does.
2. **The focus-electrode angle sets the current, smoothly.** 121 → 136 → 152 →
   166 A from 57.5° to 72.5°. Pierce's 67.5° is the planar-diode result and is not
   right for this convergent gun. Vedant's code comments already note a
   simulation-refined 130 kV design that moved to 60.5°.
3. **The aberration-corrected cathode (R_cT) is the right one to build.** With the
   uncorrected R_c, 19 trajectories hit the gun.
4. **Simulation in the loop closes the gap.** A secant search on ψ (3 PIERCE runs)
   brings the current to +0.2 %. Adding the anode position as a second knob
   (Newton, 13 runs) reaches −0.7 % current and −1.9 % waist with nothing lost.
   It stopped short of its preset tolerance (0.5 % current, 3 % waist): steps
   smaller than the 0.5 mm grid do not move any electrode point, so the last two
   steps gave identical results. That is the grid's resolution, not a solver
   failure.
5. **Vaughan's throw formula is optimistic in every run**, sir's deck included
   (+13 % to +31 %). Neither knob fixes it; it is a property of the synthesis.

## 4. Paper 1: Tiwary & Basu (IEEE TED 35(7), 1988), focus electrode and anode synthesis

**What it does.** The potential inside the beam is Langmuir and Blodgett's,
V ∝ α(γ)^(4/3). A logarithmic conformal map w = ln z makes the cathode, anode and
beam edge straight lines, so the outside potential is the real part of the same
function of s = ln(ρ/R_c) + j(θ − θ₀). The V = 0 curve is the focus electrode,
the V = V_a curve is the anode face.

**How it connects to Vedant.** His focus electrode is a straight line at 67.5° to
the beam edge. The paper's V = 0 curve starts at exactly that angle, so Vedant's
electrode is its first-order tangent; the paper supplies the rest of the curve,
and the anode face too.

**Implementation** (`src/physics/bfe.py`). The paper finds the curves by Newton
forward interpolation along radial lines. Here they are traced exactly by
inverting the α series in the complex plane: α(−s) = t·e^(−j3π/8) for the focus
electrode, α(−s) = (K + jτ)^(3/4) for the anode. Checks (`tests/test_bfe.py`):

- leaves the cathode rim at 67.50°;
- the anode curve passes through the anode sphere radius exactly (the paper's own
  interpolation missed it by 0.26–0.41 %);
- the paper's Fig. 2 is reproduced (`reports/figures/tiwary_basu_fig2.png`). Laid
  over the scanned figure, the cathode arcs and beam edges coincide, and the curves
  match near the beam, drifting apart far from it, where the paper says its
  interpolation is least accurate;
- the potential is harmonic in (ln ρ, θ) to 1e-6.

**What it gets wrong, measured.** The map is conformal in the plane, but a gun is
axisymmetric. The axisymmetric Laplacian keeps two terms, V_u + cot θ·V_θ, that
the map drops, and they are 50–80 % of the size of the terms that cancel
(`laplace_residual`). The paper reports residues under 0.5 %, by a measure it does
not define. In PIERCE the Tiwary–Basu electrodes give **14.5 % less current than
designed**, with no interception. They are a usable starting shape, and they are
in the Design tab, but they are not a better first guess than Vedant's line
(+26.5 %) by any margin that matters; both need simulation.

## 5. Paper 2: Gao et al. (AIP Advances 15, 015030, 2025), electrode angle

A COMSOL study of a 60 kV welding gun: the angle of the focus (Wehnelt)
electrode's inner wall is swept and the current, waist and emittance read off;
67.5° is cited for space-charge-limited guns and 60° found best for theirs
(temperature-limited). Their angle is measured from the axis; Vedant's ψ is
measured from the beam edge, so theirs = ψ − θ_T.

The same study was run on our gun in PIERCE (section 3, points 2 and 4), with the
step the paper does not take: tuning the angle automatically until the simulated
gun meets its specification.

## 6. Tools

```bash
python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --run          # Vedant's deck, run, compare
python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --psi 65.66 --anode-shift 2.27 --run
python -m src.sim.deck --V 20000 --I 1 --rw 0.5 --Jc 5 --electrodes tiwary-basu --run
python -m src.sim.experiments            # all variants (about 1 min on 7 cores)
python -m src.sim.experiments --tune     # both tunings (about 10 min)
python -m src.sim.report                 # figures, SIMULATION_RESULTS.md, web data
```

In each run folder, `python ../../lktron/pierce_plotter.py` gives sir's plots.
The Design tab downloads the same `pierce.in` (`web/js/pierce.js`, identical to the
Python, `tests/test_pierce_js_parity.py`); the Simulation tab shows the recorded
runs.

## 7. Limits

- One gun with a reference deck (CEERI). The 20 kV test gun runs, but neither
  construction gives a usable beam for it, so it is a tool check, not a result.
- Grid 0.5 mm; PIERCE samples the envelope every 6 mm, so throw and waist are
  refined by a parabola through the minimum. Changing the emission settings
  DZ/DX from Vedant's (1.5, 1.0) to sir's ratios (2.0, 0.6) moved Vedant's
  current from 155.3 A to 151.8 A: about 2 % of the simulator's answer depends on
  settings we did not tune.
- PIERCE uses Child-law emission (space-charge limited), no thermal velocities,
  no magnetic field.
- Sharma et al. (2001) aperture synthesis is still not implemented.
