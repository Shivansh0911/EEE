# Optimisation of Convergent Electron Gun using ANN

**Shivansh Shekher Ojha, Satwik Shrivastava, and Lalit Kumar**

Department of Electrical and Electronics Engineering, Birla Institute of Technology and Science (BITS) Pilani, Hyderabad Campus, India

*Prepared for IVAC 2027 (IEEE International Vacuum Electronics Conference). Author order and whether the supervisor is listed as co-author should be confirmed by Dr. Lalit Kumar.*

---

## Abstract (full version, about 390 words)

The half-beam cone angle θ fixes the geometry of a convergent Pierce gun. The classical synthesis that yields θ is iterative, and on 30 published guns its result is about 10 % from the measured angle. We present a design chain in which a small artificial neural network (ANN) supplies θ, the gun is synthesised around it, and a self-consistent gun simulator is used in the loop to correct the synthesis. The network is a 3-3-2-1 tanh perceptron (23 parameters) trained from scratch with Levenberg–Marquardt on 23 real guns. Its inputs are the perveance P, the waist radius r_w and ln C, where C is the area convergence. On the seven held-out guns it predicts θ with a mean relative error of 7.93 %, and the median over 30 restarts is 8.33 %. The 3.74 % reported for the same architecture and split could not be reproduced. Around θ, the cathode, anode and throw come from Vaughan's (1981) synthesis equations, solved exactly. The cathode is aberration-corrected (sin θ = 0.905 sin θ_T) and the anode aperture is 1.2 times the local beam radius. The focus electrode is the V = 0 equipotential of Tiwary and Basu's (1988) conformal-map synthesis, implemented exactly. Each gun is written as an input deck for the LKTRON PIERCE code and simulated.

We tested this on the 150 kV, 120 A CEERI klystron gun and seven 10 kV literature guns. In the simulator, six of the synthesised guns draw 3 % to 56 % more than their design perveance. The other two, both low-convergence guns, fail because the beam strikes the electrodes. With the simulator in the loop, a secant search on the anode–cathode spacing brings five guns within 1.3 % of the design perveance in 3–8 runs. For the reference gun, tuning the focus-electrode angle alone gives +0.2 % current. Tuning the angle and the anode position together gives −0.7 % current and −1.9 % waist radius.

Two parametric sweeps were run. For anode shifts of −4 to +6 mm, the simulated perveance lies between the planar 1/d² and the Langmuir–Blodgett predictions at every step. When the cathode radius of curvature is varied, the simulated focus lies 1.6–2.6 times farther from the cathode than the cathode's centre of curvature (98.6 mm against 51.9 mm at the network's angle). Simulated waists are smaller than designed in every tuned gun, so the anode alone cannot set both current and waist. The ANN makes synthesis fast enough for a 50 000-evaluation NSGA-II search, and the field simulation decides the final geometry.

**Keywords:** Pierce electron gun; convergent beam; artificial neural network; gun synthesis; PIERCE simulation.

---

## Short version (about 150 words)

The half-beam cone angle θ fixes a convergent Pierce gun's geometry. We predict θ with a 23-parameter neural network trained by Levenberg–Marquardt on 23 real guns. On seven held-out guns its error is 7.93 % (8.33 % median over 30 restarts), and the 3.74 % published for this architecture was not reproducible. The gun is built around θ from Vaughan's (1981) equations, solved exactly, with a Tiwary–Basu (1988) equipotential focus electrode. It is then simulated in the LKTRON PIERCE code. As synthesised, six guns draw 3–56 % more than their design perveance, and two low-convergence guns fail with the beam striking the electrodes. Secant tuning of the anode position, with the simulator in the loop, brings five guns within 1.3 % in 3–8 runs. Tuning the electrode angle and the anode together gives −0.7 % current and −1.9 % waist for a 150 kV klystron gun. Sweeps of anode position and cathode curvature map current, focus and interception.

---

### Source of every number (for checking)

| Number | File |
|---|---|
| ~10 % classical vs measured (10.49 % exact solver, 10.15 % paper's iterative column, 30 guns) | `reports/INTEGRATION_SUMMARY.md` |
| 3-3-2-1, 23 parameters, LM, ln C | `reports/BENCHMARK.md` §1, `reports/DECISIONS.md` D8 |
| 7.93 % (selected M1), 8.33 % median over 30 restarts, 3.74 % not reproduced | `reports/BENCHMARK.md` §2, §4 |
| 0.905, aperture 1.2 r_b | `src/physics/vendor/pierce_gun_design_fixed.py` header; `reports/INTEGRATION_DESIGN.md` §2 |
| +3 % … +56 % untuned; cases 26 and 29 fail; five guns within 1.3 % in 3–8 runs | `reports/simulation/ourgun/summary.json` (`guns`, `tuned`) |
| +0.2 % (ψ tuned); −0.7 % current, −1.9 % waist (ψ and anode) | `reports/simulation/tuned_psi.json`, `tuned_2d.json` |
| perveance between 1/d² and L-B; focus 1.6–2.6× the aim point; 98.6 mm vs 51.9 mm | `reports/simulation/sweeps/sir_gun.json`, `case15.json` |
| NSGA-II, 50 000 evaluations | `reports/INTEGRATION_SUMMARY.md` (Optimisation) |
