"""
Build the three demo videos.

    python demo/make_videos.py <clone_dir> <work_dir> <out_dir> [1 2 3]

<clone_dir> is a clone of this branch in which demo/capture.py recorded the
runs (demo_rec/*.json) and src.sim.experiments produced simulation/runs/*; the
website is driven from <clone_dir>/web/index.html.

1  our code (ANN, website, old gun drawing), then the two focus-electrode papers
2  our code + Vedant's code
3  every simulation in sir's PIERCE (target.exe), with sir's plotters
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from videokit import (CAPTION_CSS, F, INK, MONO, SANS_B, SET_CAPTION, Scenes,  # noqa: E402
                      clear_highlight, highlight, slow_fill)

CLONE, WORK, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
WHICH = sys.argv[4:] or ["1", "2", "3"]
sys.path.insert(0, CLONE)
REC = os.path.join(CLONE, "demo_rec")
RUNS = os.path.join(CLONE, "simulation", "runs")
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
    page.wait_for_timeout(300)


def crop_plot(src, dst):
    """Sir's plotter draws on a tall 9:16 page; keep the gun plot."""
    im = Image.open(src).convert("RGB")
    g = ImageOps.invert(im.convert("L")).point(lambda v: 255 if v > 12 else 0)
    x0, y0, x1, y1 = g.crop((0, 0, im.width, int(im.height * 0.3))).getbbox()
    im.crop((max(0, x0 - 20), max(0, y0 - 10), min(im.width, x1 + 20), y1 + 10)).save(dst)
    return dst


def montage(items, out):
    """Sir's pierce_plotter output for several runs, in a labelled grid."""
    tiles = []
    for name, label in items:
        p = os.path.join(RUNS, name, "pierce_plot_plot1.png")
        t = Image.open(crop_plot(p, os.path.join(WORK, f"crop_{name}.png")))
        t.thumbnail((610, 210))
        tile = Image.new("RGB", (620, 250), (255, 255, 255))
        tile.paste(t, ((620 - t.width) // 2, 34))
        ImageDraw.Draw(tile).text((10, 6), label, font=F(SANS_B, 18), fill=(20, 20, 20))
        tiles.append(tile)
    rows = (len(tiles) + 1) // 2
    img = Image.new("RGB", (2 * 630, rows * 258), (30, 33, 42))
    for i, t in enumerate(tiles):
        img.paste(t, ((i % 2) * 630 + 5, (i // 2) * 258 + 4))
    img.save(out)
    return out


# =============================================================================
def video1():
    s = Scenes(WORK, "v1")
    s.card("Our Pierce gun tool, and the two papers",
           "Part 1: our neural network and website, with the old gun drawing. Part 2: the two "
           "focus-electrode papers, implemented. EEE F266, Shivansh Shekher Ojha, Satwik "
           "Shrivastava; supervisor Lalit Kumar.",
           "Demo one of three. Part one runs our own code: the neural network for the half beam "
           "cone angle, and the website with the old gun drawing. Part two implements the two "
           "focus electrode papers sir gave us.", kicker="Demo 1 of 3")
    s.terminal(rec("v1_train"),
               "First we train our neural network, model M one, from scratch: the paper's three, "
               "three, two, one network, trained by Levenberg Marquardt on twenty three real guns. "
               "On the seven held out real guns the error is seven point nine three percent. The "
               "paper claims three point seven four, which we could not reproduce.",
               title="python -m src.ann.train --model m1")
    s.terminal(rec("v1_stage2"),
               "Next, the synthetic data stage. Model A N N E learns Vedant's exact solver to zero "
               "point zero eight percent. Against real guns the classical solver alone is ten point "
               "four percent off. The two point eight eight line is not claimed, because its gain "
               "comes from the waist radius, which cannot change the angle. M one stays our headline.",
               title="python -m src.ann.stage2")

    def predict_type(p):
        slow_fill(p, "#in-P", "2", 40)
        slow_fill(p, "#in-rw", "1.5", 40)
        slow_fill(p, "#in-C", "40", 40)
        highlight(p, ".readout")

    s.browser(URL + "#predict", [
        ("This is our website. In the Predict tab, three inputs, perveance, beam waist and "
         "convergence ratio, go into the trained network, and the cone angle comes out, right in "
         "the browser.", lambda p: highlight(p, "#panel-predict .card")),
        ("Typing a gun: perveance two, waist one point five millimetres, convergence forty. The "
         "network predicts the angle instantly.", predict_type),
        ("This is the old Pierce gun drawing: cathode, focus electrode, anode, and the beam "
         "converging to its waist. It is a sketch from the predicted angle, using the old constant, "
         "so its anode is only approximate.", lambda p: highlight(p, "#gun-svg")),
        ("The comparison card puts the network next to the three classical methods of the paper.",
         lambda p: highlight(p, "#compare-body")),
        ("The Dataset tab lists every row with its source: thirty real guns, and synthetic rows "
         "marked as not measured.", lambda p: (clear_highlight(p), tab(p, "dataset"))),
        ("The Method tab explains the network, the scaling, the training and the physics.",
         lambda p: (tab(p, "method"), highlight(p, "#arch-svg"))),
    ])
    s.card("Part 2: the two focus-electrode papers",
           "Paper 1: Tiwary and Basu, IEEE TED 1988, synthesis of the focus electrode and anode. "
           "Paper 2: Gao et al., AIP Advances 2025, effect of the focus-electrode angle.",
           "Part two. Sir asked us to implement the focus electrode synthesis of two papers. Both "
           "connect directly to Vedant's code, whose focus electrode is a straight line at Pierce's "
           "sixty seven point five degrees.", kicker="Demo 1, part 2")
    s.terminal(rec("v1_paper1"),
               "Paper one finds the focus electrode and the anode as the zero volt and anode voltage "
               "equipotentials of the ideal flow, by a conformal map. We trace them exactly. On the "
               "paper's three guns, the electrode leaves the cathode at sixty seven point five "
               "degrees, so Vedant's straight line is its tangent, and our anode hits the anode "
               "sphere exactly, where the paper's interpolation was off by up to zero point four "
               "percent.", title="python demo/paper1_check.py")
    s.image(os.path.join(REC, "paper1_overlay.png"),
            "Our curves, in red, drawn over the paper's own figure two. The cathode and anode arcs "
            "coincide, and near the beam our electrodes follow the paper's.",
            title="Tiwary & Basu (1988) Fig. 2, with our curves in red")

    def design_open(p):
        tab(p, "design")
        p.click("#ds-presets button:nth-child(2)")
        highlight(p, "#ds-svg")

    def tb(p):
        p.select_option("#ds-elec", "tiwary-basu")
        p.wait_for_timeout(200)
        highlight(p, "#ds-svg")

    def knob(p):
        p.select_option("#ds-elec", "vedant")
        highlight(p, ".ds-electrodes")
        slow_fill(p, "#ds-psi", "57.5", 60)

    s.browser(URL + "#design", [
        ("In the Design tab a gun is synthesised by Vedant's method. Here the electrodes follow "
         "Vedant: a straight focus electrode at sixty seven point five degrees.", design_open),
        ("Switching to paper one replaces it with the curved equipotential electrode and the "
         "curved anode face.", tb),
        ("Paper two studies the angle of that electrode. Here it can be set, for example fifty "
         "seven point five degrees, and the anode can be moved.", knob),
        ("Either way, one click gives a pierce dot in file that runs in sir's target dot exe.",
         lambda p: (clear_highlight(p), p.evaluate(SHOW_DECK))),
    ])
    s.image(os.path.join(CLONE, "reports", "figures", "sim_psi_sweep.png"),
            "Paper two swept the electrode angle in a commercial simulator. We ran the same study "
            "on our gun in sir's simulator: the current rises steadily with the angle, and Pierce's "
            "sixty seven point five degrees gives too much current. Demo three shows every run.",
            title="Paper 2's study in sir's PIERCE: current and waist against electrode angle")
    s.terminal(rec("v1_tests"),
               "Finally the complete check: every automated test passes, including the paper one "
               "checks and a live re run of sir's simulator.", title="python -m pytest -q")
    s.card("Next", "Demo 2: our code plus Vedant's code.   Demo 3: every simulation in sir's PIERCE.",
           "Demo two shows how Vedant's code fits in, and demo three runs every gun in sir's "
           "simulator.", kicker="End of demo 1")
    return s.render(os.path.join(OUT, "demo1_our_code_and_papers.mp4"))


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
    best = json.load(open(os.path.join(CLONE, "reports", "simulation", "tuned_2d.json")))["best"]
    B = best["variant"]

    def cur(n):
        return lk.parse_log(os.path.join(RUNS, n, "pierce.log"))["cathode_current_A"]

    def html(n, k):
        return "file:///" + os.path.join(RUNS, n, k).replace("\\", "/")

    s = Scenes(WORK, "v3")
    s.card("Every simulation, in sir's PIERCE (target.exe)",
           "Sir's deck, Vedant's deck, Vedant's with the uncorrected cathode, the Tiwary-Basu "
           "electrodes, the electrode-angle sweep, and the tuned gun.",
           "Demo three. Target dot exe is sir's PIERCE simulator: it solves the field and the "
           "space charge together and traces the electrons. Here every gun our code and Vedant's "
           "code produce is simulated in it.", kicker="Demo 3 of 3")
    s.image(text_image(os.path.join(CLONE, "simulation", "lktron", "reference_ceeri", "pierce.in"),
                       os.path.join(WORK, "refdeck.png"), "sir's reference deck: pierce.in"),
            "This is sir's own deck for the klystron gun. Vedant's deck has the same format but "
            "did not run: it misses the closing slash of the cathode block, and its solver settings "
            "crash PIERCE. Our deck writer fixes both.")
    s.terminal(rec("v3_experiments"),
               "One command builds nine decks, from our code, from Vedant's code and from the "
               "combined pipeline, and runs them all in target dot exe in parallel. The design asks "
               "for one hundred twenty amps. Sir's deck gives plus one point five percent, our ANN "
               "guns plus twenty seven and plus twenty two, Vedant's deck plus twenty six and a "
               "half, and the combined runs range from minus fourteen to plus thirty eight.",
               title="python -m src.sim.experiments")

    def anim(name, title, txt, dur):
        s.frames(trajectory_animation(os.path.join(RUNS, name), "PIERCE: " + title,
                                      [f"current {cur(name):.1f} A (design 120 A)"]), txt, min_dur=dur)

    anim("reference", "sir's reference deck (the check)",
         f"Sir's own deck first, as a check: {cur('reference'):.1f} amps, reproducing his run "
         "exactly. First the electrodes, then the field lines, then the electrons converging to "
         "the waist.", 9)
    s.card("1. Our code", "The angle from our neural networks, turned into a gun and simulated.",
           "Part one: our code. Our networks predict the angle; Vaughan's equations turn that "
           "angle into a gun, and PIERCE simulates it.", kicker="Demo 3")
    anim("ann_m1", "our ANN M1's angle (32.17 deg)",
         f"Our model M one predicts thirty two point one seven degrees for this gun, almost "
         f"exactly Vedant's angle. PIERCE gives {cur('ann_m1'):.1f} amps, twenty seven percent too "
         "much.", 8)
    anim("ann_e", "our ANN-E's angle (31.90 deg)",
         f"Model A N N E predicts thirty one point nine degrees: {cur('ann_e'):.1f} amps, twenty "
         "two percent too much.", 7)
    s.card("2. Vedant's code", "His exact angle and his electrode construction, simulated.",
           "Part two: Vedant's code, his exact angle and his electrode construction.",
           kicker="Demo 3")
    anim("vedant", "Vedant's deck as synthesised (67.5 deg electrode)",
         f"Vedant's deck: {cur('vedant'):.1f} amps, twenty six and a half percent too much, the "
         "same as our ANN guns. So the error is not the angle; it is the electrode construction.", 9)
    anim("vedant_raw", "Vedant's construction with the uncorrected cathode",
         "Without his aberration correction of the cathode, nineteen trajectories hit the gun, so "
         "the corrected cathode is the right one to build.", 8)
    s.card("3. Combined code", "Our pipeline around Vedant's construction: the two papers, and "
           "tuning with sir's simulator in the loop.",
           "Part three: the combined code. Our pipeline around Vedant's construction: the two "
           "papers, and tuning with sir's simulator in the loop.", kicker="Demo 3")
    anim("tiwary_basu", "Paper 1: Tiwary-Basu curved electrodes",
         f"Paper one's curved electrodes: {cur('tiwary_basu'):.1f} amps, fourteen percent too "
         "little, but nothing hits the gun.", 8)
    s.image(montage([("ann_m1", "1. our code: ANN M1"), ("ann_e", "1. our code: ANN-E"),
                     ("vedant", "2. Vedant's code"), ("vedant_raw", "2. Vedant, uncorrected cathode"),
                     ("tiwary_basu", "3. Paper 1: Tiwary-Basu"), ("psi_57.5", "3. Paper 2 sweep: 57.5 deg"),
                     ("psi_72.5", "3. Paper 2 sweep: 72.5 deg"), (B, "3. tuned: angle + anode")],
                    os.path.join(WORK, "montage.png")),
            "Sir's own plotter script on all of them: our code, Vedant's code, and the combined "
            "runs. In paper two's sweep, at seventy two and a half degrees the beam spreads and hits "
            "the gun.", title="sir's pierce_plotter.py on every run")
    s.terminal(rec("v3_tune"),
               "Then the simulator goes in the loop. First the electrode angle alone is tuned until "
               "the current matches, then the angle and the anode position together, until both "
               "the current and the beam waist match the design.",
               title="python -m src.sim.experiments --tune")
    s.frames(trajectory_animation(os.path.join(RUNS, B),
                                  f"PIERCE: tuned gun, {best['psi_deg']:.1f} deg, anode "
                                  f"+{best['anode_shift_mm']:.2f} mm",
                                  [f"current {cur(B):.1f} A (design 120 A)",
                                   f"waist {best['compare']['sim_waist_mm_refined']:.2f} mm "
                                   "(design 9.8 mm)"]),
             f"The tuned gun: {cur(B):.1f} amps, within one percent of the design, a waist within "
             "two percent, and no electron hits the gun.", min_dur=9)

    def rotate(p):
        p.mouse.move(640, 380)
        p.mouse.down()
        p.mouse.move(820, 330, steps=40)
        p.mouse.up()

    def axial(p):
        p.goto(html(B, "plot_index2_Normalized_Axial_Electric_Field.html"))
        p.wait_for_timeout(1500)
        p.add_style_tag(content=CAPTION_CSS)
        p.evaluate(SET_CAPTION, "And the axial electric field: strongest in the cathode to anode "
                                "gap, where the electrons are accelerated.")
        p.mouse.move(640, 380)
        p.mouse.down()
        p.mouse.move(470, 400, steps=40)
        p.mouse.up()

    s.browser(html(B, "plot_index1_Normalized_Potential.html"), [
        ("Sir's other script, new 3 D, draws the potential over the whole grid as a surface: zero "
         "at the cathode, rising to the anode voltage.", rotate),
        ("And the axial electric field: strongest in the cathode to anode gap, where the electrons "
         "are accelerated.", axial),
    ])
    s.browser(URL + "#sim", [
        ("All of this is on the website's Simulation tab: the summary,",
         lambda p: highlight(p, "#sim-stats")),
        ("every run against the design,", lambda p: highlight(p, "#sim-table")),
        ("the trajectories of any run,",
         lambda p: (p.select_option("#sim-pick", label="Tiwary & Basu (1988) electrodes"),
                    highlight(p, "#sim-svg"))),
        ("and paper two's angle sweep.", lambda p: highlight(p, "#sim-sweep")),
    ])
    s.terminal(rec("v3_tuned_run"),
               "And for any new gun, one command writes the deck, runs target dot exe, and compares "
               "the result with the design.",
               title="python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --psi 65.656 "
                     "--anode-shift 2.265 --run")
    s.card("Summary",
           "Vedant's synthesis builds the gun; sir's simulator tests it; both papers are "
           "implemented and simulated; tuning with the simulator in the loop meets the design "
           "current and waist. Limits: one reference gun, 0.5 mm grid, Vaughan's throw stays about "
           "30 % short.",
           "In short: Vedant's synthesis builds the gun, sir's simulator tests it, both papers are "
           "implemented and simulated, and tuning with the simulator in the loop meets the design. "
           "The limits: one reference gun, a half millimetre grid, and Vaughan's throw formula "
           "stays about thirty percent short.", kicker="End of demo 3")
    return s.render(os.path.join(OUT, "demo3_all_simulations_target_exe.mp4"))


if __name__ == "__main__":
    for k in WHICH:
        print("built", {"1": video1, "2": video2, "3": video3}[k]())
