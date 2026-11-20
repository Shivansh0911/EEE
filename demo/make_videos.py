"""
Build the three demo videos.

    python demo/make_videos.py <clone_dir> <work_dir> <out_dir> [1 2 3]

<clone_dir> is a clone of this branch in which demo/capture.py recorded the
runs (demo_rec/*.json) and target.exe produced demo_rec/run_*/; the website is
driven from <clone_dir>/web/index.html.
"""

import glob
import os
import subprocess
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videokit import (BG, F, INK, MONO, SANS_B, Scenes, clear_highlight, highlight, slow_fill)  # noqa: E402

CLONE, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WHICH = sys.argv[4:] or ["1", "2", "3"]
sys.path.insert(0, CLONE)
REC = os.path.join(CLONE, "demo_rec")
URL = "file:///" + os.path.join(CLONE, "web", "index.html").replace("\\", "/")
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def rec(name):
    return os.path.join(REC, name + ".json")


def text_image(path, out, title, max_lines=34):
    with open(path, encoding="latin-1") as f:
        lines = f.read().splitlines()[:max_lines]
    img = Image.new("RGB", (1240, 40 + 19 * len(lines) + 20), (12, 13, 17))
    d = ImageDraw.Draw(img)
    d.text((16, 8), title, font=F(SANS_B, 20), fill=INK)
    for i, ln in enumerate(lines):
        d.text((16, 44 + 19 * i), ln[:128], font=F(MONO, 15), fill=(200, 210, 225))
    img.save(out)
    return out


SHOW_DECK = """() => {
  const v = id => parseFloat(document.getElementById(id).value);
  const r = EGUN.closure.synthesize(v('ds-V') * 1000, v('ds-I'), v('ds-rw'), v('ds-Jc'), EGUN_DATA.kfTable);
  const d = EGUN.pierce.pierceDeck(r, {electrodes: document.getElementById('ds-elec').value,
    psi: v('ds-psi'), anodeShift: v('ds-shift')});
  let o = document.createElement('pre'); o.id = 'demo-deck';
  o.style.cssText = 'position:fixed;top:30px;left:60px;right:60px;bottom:120px;z-index:9999;' +
    'background:#0c0d11;color:#cfd8e6;border:1px solid #5aa2ff;border-radius:8px;padding:16px;' +
    'font:13px/17px Consolas,monospace;overflow:hidden;';
  o.textContent = 'pierce.in  (downloaded from the Design tab)\\n\\n' + d.text;
  document.body.appendChild(o); }"""
HIDE_DECK = "() => { const o = document.getElementById('demo-deck'); if (o) o.remove(); }"


def tab(page, name):
    page.click(f"#tab-{name}")
    page.wait_for_timeout(400)


# =============================================================================
def video1():
    s = Scenes(WORK, "v1")
    s.card("Our Pierce gun tool: the neural network, the old gun drawing, and today's "
           "cathode and focus-electrode work",
           "EEE F266 study project. Shivansh Shekher Ojha, Satwik Shrivastava. "
           "Supervisor: Lalit Kumar.",
           "Demo one of three. This video runs our own code: the neural network that "
           "predicts the half beam cone angle of a Pierce electron gun, the website with the old "
           "gun drawing, and the new cathode and focus electrode work added today.",
           kicker="Demo 1 of 3")
    s.terminal(rec("v1_train"),
               "First we train our neural network, model M one, from scratch. It is the paper's "
               "three, three, two, one network, trained by Levenberg Marquardt on the twenty three "
               "real guns of Panahi's table two. Ten random restarts are scored by five fold cross "
               "validation inside the training guns. On the seven held out real guns the error is "
               "seven point nine three percent. The paper claims three point seven four, which we "
               "could not reproduce.",
               title="python -m src.ann.train --model m1")
    s.terminal(rec("v1_stage2"),
               "Next, the synthetic data stage. Model A N N E learns Vedant's exact solver and "
               "matches it to zero point zero eight percent. Against real measured guns, the "
               "classical solver alone is ten point four percent off. The calibrated line at two "
               "point eight eight percent is not claimed: its gain comes from the waist radius, "
               "which cannot physically change the angle. So model M one stays our headline.",
               title="python -m src.ann.stage2")

    def predict_inputs(p):
        highlight(p, "#panel-predict .card")

    def predict_type(p):
        slow_fill(p, "#in-P", "2")
        slow_fill(p, "#in-rw", "1.5")
        slow_fill(p, "#in-C", "40")
        highlight(p, ".readout")

    s.browser(URL + "#predict", [
        ("This is our website. The Predict tab is the original tool. Three inputs, perveance, "
         "beam waist and convergence ratio, go into the trained network, and the half beam cone "
         "angle comes out, computed right here in the browser.", predict_inputs),
        ("Let us type a gun: perveance two, waist one point five millimetres, convergence forty. "
         "The network predicts the angle instantly.", predict_type),
        ("Below it is the old Pierce gun drawing: the cathode, the focus electrode, the anode, and "
         "the beam converging to its waist. It is a sketch built from the predicted angle, and it "
         "still uses the old Langmuir Blodgett constant, so its anode is only approximate.",
         lambda p: highlight(p, "#gun-svg")),
        ("The comparison card puts the network next to the three classical methods from the paper.",
         lambda p: highlight(p, "#compare-body")),
        ("The Dataset tab lists every row with its source: thirty real guns from the paper, and "
         "synthetic rows clearly marked as not measured.", lambda p: (clear_highlight(p), tab(p, "dataset"))),
        ("The Method tab explains the network, the input scaling, Levenberg Marquardt training, "
         "and the physics behind it.", lambda p: (tab(p, "method"), highlight(p, "#arch-svg"))),
    ])

    def design_open(p):
        tab(p, "design")
        slow_fill(p, "#ds-V", "20")
        slow_fill(p, "#ds-I", "1")
        slow_fill(p, "#ds-rw", "0.5")
        slow_fill(p, "#ds-Jc", "5")
        highlight(p, "#panel-design .grid.predict .card")

    def tb(p):
        p.select_option("#ds-elec", "tiwary-basu")
        p.wait_for_timeout(300)
        highlight(p, "#ds-svg")

    def knobs(p):
        p.select_option("#ds-elec", "vedant")
        highlight(p, ".ds-electrodes")
        slow_fill(p, "#ds-psi", "60")
        slow_fill(p, "#ds-shift", "0.5")

    s.browser(URL + "#design", [
        ("Now the new work. The Design tab takes the four real engineering inputs: voltage, "
         "current, waist radius and cathode loading. Here, twenty kilovolts, one amp, half a "
         "millimetre and five amps per square centimetre.", design_open),
        ("Vedant's method computes theta T, the angle the cathode is built with, and theta raw, "
         "the analysis angle that every length comes from, plus what a real measured gun would "
         "likely show.", lambda p: highlight(p, "#panel-design .readout")),
        ("The drawing is the cathode and electrodes exactly as they will be simulated. The orange "
         "arc is the cathode, built with the aberration corrected radius. The straight focus "
         "electrode leaves the cathode rim at Pierce's sixty seven point five degrees.",
         lambda p: highlight(p, "#ds-svg")),
        ("Today we added the focus electrode synthesis of Tiwary and Basu, nineteen eighty eight. "
         "Choosing it replaces the straight electrode and the anode face with the curved "
         "equipotentials of the ideal electron flow.", tb),
        ("With Vedant's construction, the electrode angle and the anode position can now be "
         "changed. These are the two knobs we tuned in sir's simulator.", knobs),
        ("One click downloads a pierce dot in file that runs directly in sir's target dot exe. "
         "This is that file.", lambda p: (clear_highlight(p), p.evaluate(SHOW_DECK))),
        ("The geometry table lists every dimension: cathode and anode radii, spacing, aperture, "
         "throw, and the electric field in the gap.",
         lambda p: (p.evaluate(HIDE_DECK), highlight(p, "#ds-rows"))),
    ])
    s.terminal(rec("v1_tests"),
               "Finally, the complete check. Every automated test passes: the physics checks, the "
               "checks that the browser gives the same numbers as Python, and a live re run of "
               "sir's simulator.",
               title="python -m pytest -q", show_from=None)
    s.card("Next", "Demo 2: our code plus Vedant's code.   Demo 3: our guns in sir's simulator.",
           "That was our own code. Demo two shows how Vedant's code fits in, and demo three runs "
           "our guns in sir's simulator.", kicker="End of demo 1")
    return s.render(os.path.join(OUT, "demo1_our_code.mp4"))


# =============================================================================
def video2():
    s = Scenes(WORK, "v2")
    s.card("Our code + Vedant's code",
           "His exact Vaughan (1981) synthesis, used unchanged, with our fast copy of it, our "
           "data, and the website built around it.",
           "Demo two. Vedant's code is an exact implementation of Vaughan's nineteen eighty one "
           "Pierce gun synthesis. We use it unchanged, and build around it.", kicker="Demo 2 of 3")
    s.terminal(rec("v2_vedant"),
               "Here Vedant's solver runs on its own, exactly as his program does, for a one "
               "hundred fifty kilovolt, one hundred twenty amp klystron gun. It prints the angle, "
               "twenty eight point eight degrees, the corrected angle thirty two point two, every "
               "radius and spacing, and the throw. Then the same gun goes through our emulator of "
               "his solver: the answers agree to about one part in ten million, and in batches the "
               "emulator is hundreds of times faster.",
               title="python demo/vedant_direct.py 150000 120 9.8 5")
    s.terminal(rec("v2_g1"),
               "Gate G one checks that properly: one hundred random guns through both. The mean "
               "difference is about seven parts in a billion, and both agree on every gun that is "
               "impossible to build.",
               title="python -m src.physics.gate_g1")
    s.terminal(rec("v2_deck"),
               "His code also writes the electrode point deck. Our wrapper answers his interactive "
               "prompts with his own default values, so the deck comes straight from his "
               "construction, with no change to his file.",
               title="python -m src.moo.deck --V 14000 --I 0.5 --rw 0.2255 --Jc 5")
    s.image(text_image(os.path.join(REC, "shin_deck.txt"), os.path.join(WORK, "deck.png"),
                       "demo_rec/shin_deck.txt, written by Vedant's own code"),
            "This is that deck: the cathode arc, the focus electrode, the anode and the drift tube, "
            "as points snapped to the simulation grid.")

    def ceeri(p):
        tab(p, "design")
        p.click("#ds-presets button:nth-child(2)")
        highlight(p, "#panel-design .readout")

    s.browser(URL + "#design", [
        ("On the website the same solver runs in the browser. The three boxes at the top say how "
         "far to trust the numbers: the emulator against Vedant's solver, the band for a real "
         "measured gun, and Vaughan's own ten percent error against real guns.",
         lambda p: highlight(p, "#ds-stats")),
        ("Choosing a gun, here the same klystron gun, gives Vedant's full design instantly.", ceeri),
        ("The geometry table is his output: cathode sphere radius fifty seven point four "
         "millimetres, cathode to anode spacing twenty nine point eight, throw seventy seven "
         "point eight.", lambda p: highlight(p, "#ds-rows")),
        ("Vedant's original deck format is still offered. The PIERCE deck next to it is the "
         "version that actually runs in sir's simulator, which demo three shows.",
         lambda p: highlight(p, "#ds-pdeck")),
    ])
    s.card("How the two fit",
           "Vedant's solver is the exact reference. Our emulator copies it to about one part in "
           "ten million and makes it fast enough for the website and the simulator pipeline. His "
           "deck construction becomes the input to sir's simulator.",
           "So Vedant's solver is the exact reference, our emulator copies it and makes it fast, "
           "and his electrode construction becomes the input to sir's simulator.",
           kicker="End of demo 2")
    return s.render(os.path.join(OUT, "demo2_our_code_plus_vedant.mp4"))


# =============================================================================
def video3():
    from anim import trajectory_animation
    from src.sim import lktron as lk
    s = Scenes(WORK, "v3")
    s.card("Our guns in sir's simulator (target.exe)",
           "Dr. Lalit Kumar's LKTRON PIERCE: electric field, space charge and electron "
           "trajectories, solved together.",
           "Demo three. Target dot exe is sir's PIERCE simulator. It reads a pierce dot in deck, "
           "solves the electric field and the space charge together, traces the electrons, and "
           "repeats until it all agrees. That tells us what a synthesised gun really does.",
           kicker="Demo 3 of 3")
    s.image(text_image(os.path.join(CLONE, "simulation", "lktron", "reference_ceeri", "pierce.in"),
                       os.path.join(WORK, "refdeck.png"), "sir's reference deck: pierce.in"),
            "This is sir's own deck for the klystron gun. Vedant's deck uses the same format, but "
            "it does not run as written: it is missing the closing slash of the cathode block, and "
            "its solver settings make PIERCE crash. Our deck writer fixes both.")

    def runinfo(name):
        c = lk.parse_log(os.path.join(REC, name, "pierce.log"))
        return c

    v = runinfo("run_vedant")
    s.terminal(rec("v3_vedant_run"),
               "Our pipeline now runs Vedant's deck for that gun in target dot exe. The design asks "
               f"for one hundred twenty amps. PIERCE finds {v['cathode_current_A']:.1f} amps, "
               "about twenty six percent too much.",
               title="python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --run")
    s.frames(trajectory_animation(os.path.join(REC, "run_vedant"),
                                  "What PIERCE computed: Vedant's deck as synthesised",
                                  [f"current {v['cathode_current_A']:.1f} A (design 120 A)",
                                   "focus electrode at 67.5 deg to the beam edge"]),
             "Here is what PIERCE computed. First the electrodes, then the equipotential lines of "
             "the field, then the electron trajectories leaving the cathode and converging to the "
             "waist. The inset shows the current settling over PIERCE's iterations, well above the "
             "design line.", min_dur=14)
    t = runinfo("run_tuned")
    s.terminal(rec("v3_tuned_run"),
               "The focus electrode angle controls the current. We tuned the angle and the anode "
               "position with PIERCE in the loop. At sixty five point seven degrees, with the anode "
               "two point three millimetres further away, the current comes within one percent of "
               "the design, and the waist within two percent.",
               title="python -m src.sim.deck ... --psi 65.656 --anode-shift 2.265 --run")
    s.frames(trajectory_animation(os.path.join(REC, "run_tuned"),
                                  "What PIERCE computed: the tuned gun",
                                  [f"current {t['cathode_current_A']:.1f} A (design 120 A)",
                                   "focus electrode 65.7 deg, anode +2.27 mm"]),
             "The tuned gun. The current settles on the design line, and no electron hits the gun.",
             min_dur=12)
    plots = sorted(glob.glob(os.path.join(REC, "run_tuned", "pierce_plot*.png")))
    if plots:
        s.image(plots[0], "Sir's own plotter script reads the same output files and draws the "
                "electrodes, trajectories and equipotentials.", title="sir's pierce_plotter.py")
    s.terminal(rec("v3_tb_run"),
               "The curved Tiwary and Basu electrodes, from today's paper, give about fourteen "
               "percent less current than designed. Neither construction is right first time, "
               "which is exactly why the simulator matters.",
               title="python -m src.sim.deck ... --electrodes tiwary-basu --run")
    s.browser(URL + "#sim", [
        ("The Simulation tab on the website keeps all of these runs.",
         lambda p: highlight(p, "#sim-stats")),
        ("The table compares each run with the design: current, electrons lost in the gun, throw "
         "and waist.", lambda p: highlight(p, "#sim-table")),
        ("The viewer draws the trajectories of any run in sir's colours. This is Vedant's deck as "
         "synthesised.", lambda p: (p.select_option("#sim-pick", "1"), highlight(p, "#sim-svg"))),
        ("And this is the tuned gun.", lambda p: p.select_option("#sim-pick", "8")),
        ("The sweep shows the current rising steadily with the focus electrode angle: the study of "
         "Gao and colleagues, twenty twenty five, repeated on our gun.",
         lambda p: highlight(p, "#sim-sweep")),
    ])
    s.card("Summary",
           "Vedant's synthesis gives the gun; sir's simulator tests it; tuning with the simulator "
           "in the loop makes it meet the design. Limits: one reference gun, a 0.5 mm grid, and "
           "Vaughan's throw formula stays about 30 % short in every run.",
           "In short: Vedant's synthesis gives the gun, sir's simulator tests it, and tuning with "
           "the simulator in the loop makes it meet the design. The limits: one reference gun, a "
           "half millimetre grid, and the throw from Vaughan's formula stays about thirty percent "
           "short in every run.", kicker="End of demo 3")
    return s.render(os.path.join(OUT, "demo3_target_exe_simulation.mp4"))


if __name__ == "__main__":
    for k in WHICH:
        print("built", {"1": video1, "2": video2, "3": video3}[k]())
