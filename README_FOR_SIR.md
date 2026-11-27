# Running the project on your computer, with your target.exe

**Optimisation of a Convergent Electron Gun using ANN**, EEE F266.
Shivansh Shekher Ojha and Satwik. Supervisor: Dr. Lalit Kumar.

## Steps

1. Unzip the folder anywhere, for example `D:\electron_gun_ann`.
2. Copy your `target.exe` into `simulation\lktron\` inside that folder.
   (It can also go in the folder itself. Or you can give its path: `START_HERE.bat --exe "D:\LKTRON\target.exe"`.)
3. Double-click **`START_HERE.bat`**.
   - A terminal window opens and prints a link, `http://127.0.0.1:8765/#sim`.
   - The browser opens on that link. If it does not, open the link yourself.
   - Keep the terminal window open while you use the site. Closing it stops the site.

**First run only.** If Python's numpy, pandas and mpmath are missing, the launcher installs them into a `.venv` folder. This needs internet once. Python 3.10 or newer must be installed from python.org, with "Add python.exe to PATH" ticked.

## What the site shows

| Tab | What it is |
|---|---|
| **Predict** | You give P (microperveance), r_w (waist radius, mm) and C (area convergence). The site shows:<br>• our neural network's beam angle θ;<br>• the Pierce gun built from θ (cathode, anode, beam) and its geometry table;<br>• the beam-focusing electrode by Tiwary & Basu (1988), with its angle: 67.5° to the beam edge, and the corresponding angle to the axis. |
| **Simulation** | **Our gun in target.exe.** Move the slider to step through recorded target.exe runs:<br>• **Anode shift:** the anode moves along the axis. You see the current against 1/d² and Langmuir–Blodgett, and where the beam focuses.<br>• **Cathode R_c:** the cathode's radius of curvature changes, with the anode re-placed by Langmuir–Blodgett. You see the aim point, the actual focus, and electrons hitting the gun (orange circles).<br>**Run our gun in target.exe now:** type any P, r_w, C, V, anode shift and angle change, then press Run. The site builds our gun, runs **your target.exe** on this computer (under a minute), and draws the trajectories. |
| Dataset, Results | The training data, network accuracy and the validation gates. |

Each live run keeps all of target.exe's files in `simulation\runs\local\<date_time_inputs>\`: `pierce.in`, `pierce.log`, `pierce.plt` and the rest. So your `pierce_plotter.py` and `new3d.py` can be run on them as usual.

## From the terminal (optional)

| Command | What it runs |
|---|---|
| `python -m src.sim.ourgun --tune` | Our gun on your reference gun and the paper's 7 held-out guns, then the anode tuned with target.exe in the loop |
| `python -m src.sim.sweeps` | The anode-shift and R_c sweeps on your reference gun |
| `python -m src.sim.sweeps --gun case15` | The same sweeps on paper case 15 |
| `python demo\run_reference.py` | Your own reference deck, run through our driver; it reproduces 121.7416 A |

`target.exe` is never modified or redistributed by this project. It is only called.
