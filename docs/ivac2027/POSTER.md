# Poster content: "Optimisation of Convergent Electron Gun using ANN" (IVAC 2027)

This is a brief for a designer. Format: **A0 portrait (841 × 1189 mm)**. All figure paths are relative to the repository root (`EEE/`). Every number on the poster comes from the files listed in `docs/ivac2027/REPORT.md`. Please do not round or change the numbers.

---

## Header band (full width, about 12 % of height)

- **Title:** Optimisation of Convergent Electron Gun using ANN
- **Authors:** Shivansh Shekher Ojha, Satwik Shrivastava, Lalit Kumar *(supervisor; confirm co-authorship and order)*
- **Affiliation:** Department of Electrical and Electronics Engineering, BITS Pilani, Hyderabad Campus, India
- **Logos:** BITS Pilani (left) and IVAC 2027 (right), if the conference template allows them.
- **One-line takeaway under the title (large):** *A 23-parameter network gives the beam angle, Vaughan's equations build the gun, and a field simulator in the loop corrects it.*

## Layout (three columns below the header)

| Column 1 (left) | Column 2 (centre) | Column 3 (right) |
|---|---|---|
| 1 Introduction | 4 Simulation in PIERCE | 6 Optimisation results |
| 2 Method pipeline (diagram) | 5 Parametric study: anode shift and cathode R_c (the largest block, with 4 figures) | 7 Conclusion |
| 3a The ANN | | 8 References |
| 3b Gun synthesis + Tiwary–Basu electrode | | 9 Acknowledgements |

Put a **callout strip** of four large numbers across the bottom of column 2 or the top of column 3 (see "Key-number callouts").

---

## 1 Introduction (≤ 60 words)

- A convergent Pierce gun's geometry (cathode, anode, focus electrode, throw) follows from one number: the half-beam cone angle θ.
- The classical synthesis gives θ iteratively. On 30 published guns it is about 10 % from the measured angle.
- Goal: get θ quickly with an ANN, build the gun, and let a self-consistent simulator correct what the equations miss.

## 2 Method pipeline (diagram, ≤ 40 words of labels)

**Description for the designer.** Draw a left-to-right flow of five boxes with arrows, plus a feedback arrow from box 5 back to box 4:

1. **Inputs**: P (µperv), r_w (mm), C
2. **ANN**: 3-3-2-1 tanh network → θ
3. **Synthesis**: Vaughan (1981) equations, solved exactly → cathode R_cT, anode, aperture, throw
4. **Focus electrode**: Tiwary–Basu (1988), V = 0 equipotential, 67.5° to the beam edge → PIERCE input deck
5. **PIERCE (target.exe)**: field and space charge, self-consistent → current, waist, focus, interception

Feedback arrow labelled: "tune anode spacing / electrode angle until the simulated perveance equals the design (secant search)".

Optional inset under box 2: `docs/ivac2027/screens/predict_gun_svg.png`, the gun drawn by the website from the predicted θ.

## 3a The ANN (≤ 60 words)

- 3 → 3 → 2 → 1, tanh, **23 parameters**, written from scratch in NumPy.
- Levenberg–Marquardt with an analytic Jacobian (agrees with finite differences to < 1e-6).
- Dataset: 30 real guns from published Table 2; 23 train / 7 test; C used as ln C.
- Test error **7.93 %** (median **8.33 %** over 30 restarts). The published 3.74 % was **not reproducible**.

**Figure:** `reports/figures/fig7_restart_histogram.png`
Caption: *Test error over 30 restarts of the same network and split. The result is a distribution, not one number. The paper's 3.74 % lies in its best few per cent.*

Second, smaller figure (optional): `reports/figures/fig3_regression_test.png`
Caption: *Predicted against measured θ, seven held-out guns.*

## 3b Gun synthesis and Tiwary–Basu electrode (≤ 60 words)

- Vaughan's (1981) synthesis equations, solved exactly: Langmuir–Blodgett cone constant 14.665 µperv.
- The cathode is built aberration-corrected (sin θ = 0.905 sin θ_T), and the anode aperture is 1.2 r_b.
- Focus electrode: the V = 0 equipotential of Tiwary & Basu's (1988) logarithmic conformal map, traced exactly. It leaves the cathode rim at 67.50° and reproduces the paper's Fig. 2.

**Figure:** `docs/ivac2027/screens/predict_bfe_card.png`
Caption: *Website: focus electrode (V = 0) and anode face (V = V_a) for the predicted θ; 67.5° to the beam edge = 37.4° to the axis in this example.*

Small inset (optional): `reports/figures/tiwary_basu_fig2.png`. Caption: *Fig. 2 of Tiwary & Basu (1988) recomputed by our implementation (three guns).*

## 4 Simulation in PIERCE (≤ 60 words)

- Every gun is written as a deck for Dr. Lalit Kumar's LKTRON PIERCE (`target.exe`), which solves the field and space charge self-consistently.
- Driver check: the supervisor's reference deck is reproduced exactly, **121.7416 A**.
- Our gun on 8 guns, untuned: six draw **+3 % to +56 %** more perveance than designed. Cases 26 and 29 (low perveance) fail because the beam hits the gun.

**Figure:** `reports/figures/sim_trajectories.png` (crop to the top two panels if space is short)
Caption: *PIERCE trajectories (red) and equipotentials (green), CEERI 150 kV / 120 A gun: the supervisor's reference deck, the synthesised deck, the Tiwary–Basu deck, and two tuned decks.*

Table for this block (small):

| Gun | Design µP | PIERCE µP | Error |
|---|---|---|---|
| CEERI 150 kV | 2.0656 | 2.128 | +3.0 % |
| Case 3 | 2.50 | 2.702 | +8.1 % |
| Case 6 | 2.27 | 2.507 | +10.4 % |
| Case 12 | 1.20 | 1.870 | +55.8 % |
| Case 15 | 1.01 | 1.275 | +26.2 % |
| Case 17 | 0.74 | 1.139 | +53.9 % |
| Case 26 | 0.29 | — | beam hits gun (19/20) |
| Case 29 | 0.40 | — | beam hits gun (26/27) |

## 5 Parametric study: anode shift and cathode R_c (≤ 60 words of text + 4 figures)

- **Anode shift (−4 to +6 mm):** at every shift, the simulated perveance lies between the planar 1/d² scaling and the Langmuir–Blodgett prediction. Moving the anode away lowers the current and shrinks the waist.
- **Cathode R_c (θ ± 8°):** the real focus lies **1.6–2.6 ×** farther from the cathode than its centre of curvature. Too flat a cathode puts the beam on the gun.

**Figures (2 × 2 grid):**
1. `reports/figures/sweep_anode_sir_gun.png`: *CEERI 150 kV gun: current, focus point and waist against anode shift.*
2. `reports/figures/sweep_rc_sir_gun.png`: *CEERI gun: aim point vs actual focus, waist and current against R_c. At R_c = 67.5 mm, 38 of 56 electrons hit the gun.*
3. `reports/figures/sweep_anode_case15.png`: *Case 15 (10 kV): same anode study. At −4 mm, 10 of 52 electrons hit the gun.*
4. `reports/figures/sweep_rc_case15.png`: *Case 15: same R_c study.*

Optional screenshot beside the grid: `docs/ivac2027/screens/sim_sweep_rc.png`. Caption: *The website steps through the recorded target.exe runs with a slider.*

## 6 Optimisation results (≤ 60 words)

- **Anode spacing, secant search with PIERCE in the loop:** five guns within **1.3 %** of the design perveance in 3–8 runs (CEERI −1.29 %, case 6 +0.13 %, case 12 −0.42 %, case 15 +0.20 %, case 17 +0.66 %).
- **Electrode angle ψ (CEERI):** 67.5° → 57.08° gives **+0.2 %** current.
- **ψ + anode (Newton, 13 runs):** **−0.7 %** current, **−1.9 %** waist.
- **NSGA-II:** 50 000 evaluations per study. Pareto designs re-verified by the exact solver.

**Figure:** `reports/figures/sim_psi_sweep.png`
Caption: *Current and waist against the focus-electrode angle (CEERI gun). Pierce's 67.5° gives 26.5 % too much current for this convergent gun.*

Optional: `reports/moo/ceeri_klystron/pareto.png`. Caption: *NSGA-II Pareto front, CEERI study (classical-model optimum, not measured).*

## 7 Conclusion (≤ 60 words)

- A 23-parameter ANN predicts θ to about 8 % on real held-out guns, roughly the level of the classical methods, but in microseconds.
- Synthesised guns draw too much current in a field solve. A few simulator-in-the-loop runs correct it.
- Simulated waists are smaller than designed, so current and waist need two knobs (angle and anode).

## 8 References (small type)

1. R. Panahi, S.A.H. Feghhi, Sh. Sanaye Hajari, M. Khorsandi, *Results in Physics* 73 (2025) 108256.
2. J.R.M. Vaughan, "Synthesis of the Pierce gun," *IEEE Trans. Electron Devices* ED-28(1) (1981) 37–41.
3. U. Tiwary, B.N. Basu, *IEEE Trans. Electron Devices* 35(7) (1988) 1184–1187.
4. U. Tiwary, B. Basu, *IEEE Trans. Electron Devices* 34(5) (1987) 1218–1222.
5. C. Yang, B. Jia, Z. Zhu, *IEEE Trans. Electron Devices* 53(11) (2006) 2849–2852.
6. I. Langmuir, K.B. Blodgett, *Phys. Rev.* 24 (1924) 49–59.
7. H. Gao, G. Ding, X. Zhou et al., *AIP Advances* 15 (2025) 015030.
8. J.R. Pierce, *Theory and Design of Electron Beams*, Van Nostrand, 1954.

## 9 Acknowledgements (one line)

We thank Dr. Lalit Kumar for the use of his LKTRON PIERCE simulator, and Vedant for his implementation of Vaughan's synthesis, used with permission.

---

## Key-number callouts (large type, four boxes)

| Callout | Sub-label |
|---|---|
| **7.93 %** | ANN θ error, 7 held-out real guns (median 8.33 % over 30 restarts) |
| **+3 % … +56 %** | extra perveance drawn by the synthesised guns in PIERCE |
| **≤ 1.3 %** | perveance error after anode tuning, 5 guns, 3–8 simulator runs |
| **−0.7 % / −1.9 %** | current / waist error, CEERI 150 kV gun, angle + anode tuned |

## Colour and visual notes

- The figures already use blue `#2a6fdb` (simulation), grey `#8a8f98` (1/d² / aim point) and orange `#d9822b` (Langmuir–Blodgett). Use the same blue for "simulation" everywhere on the poster, and the same orange for "theory".
- In the PIERCE trajectory plots, trajectories are red and equipotentials green. Keep their legend in one place.
- Light background. The website screenshots have a dark theme, so give them a thin dark frame and treat them as "screen" images, apart from the plots.
- Mark negative results in one consistent neutral style (e.g. a grey "not reproduced" / "fails" tag), not red alarm colours.
- Minimum body text 28 pt at A0, and section heads 48–60 pt.

## Figures to export (checklist)

- [ ] `reports/figures/fig7_restart_histogram.png` (3a)
- [ ] `reports/figures/fig3_regression_test.png` (3a, optional)
- [ ] `docs/ivac2027/screens/predict_gun_svg.png` (2, optional inset)
- [ ] `docs/ivac2027/screens/predict_bfe_card.png` (3b)
- [ ] `reports/figures/tiwary_basu_fig2.png` (3b, optional)
- [ ] `reports/figures/sim_trajectories.png` (4)
- [ ] `reports/figures/sweep_anode_sir_gun.png` (5)
- [ ] `reports/figures/sweep_rc_sir_gun.png` (5)
- [ ] `reports/figures/sweep_anode_case15.png` (5)
- [ ] `reports/figures/sweep_rc_case15.png` (5)
- [ ] `docs/ivac2027/screens/sim_sweep_rc.png` (5, optional)
- [ ] `reports/figures/sim_psi_sweep.png` (6)
- [ ] `reports/moo/ceeri_klystron/pareto.png` (6, optional)
- [ ] Pipeline diagram (2), to be drawn by the designer from the description above

The matplotlib figures were saved at 140 dpi (`src/sim/sweeps.py`). For A0 print, regenerate them at 300 dpi or as SVG/PDF by changing `dpi=` / the file extension in the `figures()` functions. The re-plot reads the saved JSON (`python -m src.sim.sweeps --gun sir_gun --report-only`) and does not re-run target.exe. Do **not** use `docs/ivac2027/screens/sim_live_card_local.png` on the poster: it shows a local Windows path.
