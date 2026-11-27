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
        t = Image.open(crop_plot(p, os.path.join(WORK, f"crop_{name.replace('/', '_')}.png")))
        t = t.crop((0, 0, int(t.width * 0.72), t.height))     # drop the legend box
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

    def rotate(dx, dy):
        def act(p):
            p.wait_for_timeout(600)
            p.mouse.move(640, 380)
            p.mouse.down()
            p.mouse.move(640 + dx, 380 + dy, steps=50)
            p.mouse.up()
        return act

    s.browser(html(B, "plot_index1_Normalized_Potential.html"), [
        ("Sir's other script, new 3 D, draws the potential over the whole grid as a surface: zero "
         "at the cathode, rising to the anode voltage.", rotate(110, 15)),
    ])
    s.browser(html(B, "plot_index2_Normalized_Axial_Electric_Field.html"), [
        ("The same script draws the axial and radial electric field on the same grid, from "
         "PIERCE's own output file.", rotate(-90, 10)),
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


# =============================================================================
def lines_image(blocks, out, width=1240):
    """Text panels side by side: blocks = [(title, [lines], {line_index: colour})]."""
    n = len(blocks)
    col_w = width // n
    h = 60 + 19 * max(len(b[1]) for b in blocks) + 20
    img = Image.new("RGB", (width, h), (12, 13, 17))
    d = ImageDraw.Draw(img)
    for k, (title, lines, marks) in enumerate(blocks):
        x = k * col_w + 14
        d.text((x, 10), title, font=F(SANS_B, 19), fill=INK)
        for i, ln in enumerate(lines):
            d.text((x, 46 + 19 * i), ln[: col_w // 8], font=F(MONO, 14),
                   fill=marks.get(i, (200, 210, 225)))
        if k:
            d.line([k * col_w, 0, k * col_w, h], fill=(44, 49, 61), width=2)
    img.save(out)
    return out


def video4():
    s = Scenes(WORK, "v4")
    s.card("Our website: the beam-focusing electrode from Tiwary & Basu (1988)",
           "Inputs P, r_w and C go into our neural network for the half-beam cone angle; Tiwary and "
           "Basu's synthesis then gives the focus electrode, its angle, and the anode face.",
           "This demo shows our website after adding Tiwary and Basu's nineteen eighty eight paper. "
           "From the input parameters it now calculates the beam focusing electrode and its angle.",
           kicker="Demo: Tiwary & Basu on our website")
    s.terminal(rec("v1_paper1"),
               "First, the paper's method, checked on the paper's own three guns. The focus electrode "
               "is the zero volt equipotential of the ideal electron flow. It leaves the cathode rim at "
               "sixty seven point five degrees to the beam edge, and our anode curve hits the anode "
               "sphere exactly.", title="python demo/paper1_check.py")
    s.image(os.path.join(REC, "paper1_overlay.png"),
            "Our curves in red, drawn over the paper's own figure two: they follow the paper's "
            "electrodes near the beam.", title="Tiwary & Basu (1988) Fig. 2, with our curves in red")

    def gun1(p):
        slow_fill(p, "#in-P", "2", 40)
        slow_fill(p, "#in-rw", "1.5", 40)
        slow_fill(p, "#in-C", "40", 40)
        highlight(p, ".readout")

    def gun2(p):
        slow_fill(p, "#in-P", "0.3", 40)
        slow_fill(p, "#in-rw", "0.62", 40)
        slow_fill(p, "#in-C", "207.6", 40)
        highlight(p, "#gun-svg")

    s.browser(URL + "#predict", [
        ("This is the Predict tab. The three inputs are perveance, beam waist radius and "
         "convergence ratio.", lambda p: highlight(p, "#panel-predict .card")),
        ("For perveance two, waist one point five millimetres and convergence forty, our network "
         "predicts a half beam cone angle of forty point nine degrees.", gun1),
        ("From these inputs and that angle, Tiwary and Basu's method builds the gun. The orange arc "
         "is the cathode. Its radius is root C times the waist, and its curvature follows from the "
         "angle.", lambda p: highlight(p, "#gun-svg")),
        ("The white curve is the beam focusing electrode: the zero volt equipotential of the paper. "
         "It leaves the cathode rim at sixty seven point five degrees to the beam edge, which here "
         "is twenty six point six degrees to the axis. The grey curve is the anode face.",
         lambda p: highlight(p, "#gun-svg")),
        ("The table lists every number: the cathode and anode radii, the spacing, the electrode "
         "angle to the beam edge and to the axis, and where the electrode and anode end.",
         lambda p: highlight(p, "#geom-rows")),
        ("Change the gun and everything is recomputed instantly. A high convergence gun: perveance "
         "zero point three, waist zero point six two, convergence two hundred and seven. The angle "
         "is now twenty eight point four degrees, so the electrode sits at thirty nine point one "
         "degrees to the axis.", gun2),
        ("The buttons load real held out guns from the paper's table, with their measured angle "
         "for comparison.", lambda p: (p.click("#preset-row button"), highlight(p, "#messages"))),
    ])

    def design_tb(p):
        tab(p, "design")
        p.click("#ds-presets button:nth-child(2)")
        p.select_option("#ds-elec", "tiwary-basu")
        p.wait_for_timeout(200)
        highlight(p, "#ds-svg")

    s.browser(URL + "#design", [
        ("In the Design tab, the same paper's curved electrodes can be built into a full gun and "
         "downloaded as a deck for sir's simulator.", design_tb),
    ])
    s.card("Summary",
           "Inputs -> our network's angle -> Tiwary & Basu: the focus electrode (V = 0) at 67.5 deg "
           "to the beam edge = 67.5 - theta to the axis, and the anode face (V = Va). The browser's "
           "numbers match our Python implementation to one part in a billion.",
           "So the website now takes the input parameters, predicts the angle with our network, and "
           "uses only Tiwary and Basu's paper to give the beam focusing electrode and its angle. The "
           "browser's numbers match our tested Python version to one part in a billion.",
           kicker="End")
    return s.render(os.path.join(OUT, "demoA_website_tiwary_basu.mp4"))


# =============================================================================
def video5():
    from anim import trajectory_animation
    from src.sim import lktron as lk
    best = json.load(open(os.path.join(CLONE, "reports", "simulation", "tuned_2d.json")))["best"]
    B = best["variant"]

    def cur(n):
        return lk.parse_log(os.path.join(RUNS, n, "pierce.log"))["cathode_current_A"]

    s = Scenes(WORK, "v5")
    s.card("How our code runs on sir's code: target.exe (LKTRON PIERCE)",
           "What target.exe is, how our code drives it, the fixes it needed, and every output.",
           "This demo shows how our code works with sir's simulator, target dot exe, and what it "
           "produced.", kicker="Demo: our code on sir's simulator")
    s.card("What target.exe is",
           "LKTRON PIERCE by Dr. Lalit Kumar. Input: pierce.in, the electrode outline as points. It "
           "solves the electric field on a grid, emits electrons from the cathode by Child's law, "
           "traces them, adds their space charge, and repeats until it converges. Output: current, "
           "throw, waist, trajectories, fields.",
           "Target dot exe is sir's PIERCE simulator. Its input, pierce dot in, is the electrode "
           "outline as a list of points with their voltages. It solves the electric field on a grid, "
           "releases electrons from the cathode, traces them, adds the electrons' own charge to the "
           "field, and repeats until nothing changes. It reports the current, where the beam "
           "focuses, and whether electrons hit the metal.", kicker="Background")
    s.image(text_image(os.path.join(CLONE, "simulation", "lktron", "reference_ceeri", "pierce.in"),
                       os.path.join(WORK, "refdeck.png"), "sir's deck: pierce.in"),
            "This is sir's deck for the klystron gun. The top blocks set the grid and the solver. "
            "The fourteen rows are the electrode points: cathode and focus electrode at zero volts, "
            "anode and drift tube at one hundred fifty kilovolts. The last block describes the "
            "cathode.")
    s.terminal(rec("v5_reference"),
               "Step one: run sir's own deck through our code. Our driver writes the file, runs target "
               "dot exe, and reads back the results. One hundred twenty one point seven four amps, "
               "throw eighty five point five millimetres, waist seven point four three: identical to "
               "sir's own run.", title="python demo/run_reference.py")
    log = open(os.path.join(RUNS, "reference", "pierce.log"), encoding="latin-1").read().splitlines()
    a = next(i for i, l in enumerate(log) if "MAJOR" in l)
    b = next(i for i, l in enumerate(log) if "Electrode  Potential" in l)
    excerpt = [l.rstrip()[:118] for l in log[a:b + 5] if l.strip()][:40]
    s.image(lines_image([("pierce.log, written by target.exe", excerpt, {})],
                        os.path.join(WORK, "log.png")),
            "This is what target dot exe writes. Each row is one round of the solution: the current "
            "starts at three hundred thirty amps and settles at one hundred twenty one point seven "
            "within twenty rounds. Below it, the beam radius along the axis, the throw, the waist, "
            "and the current that hits each electrode.")
    ved = open(os.path.join(REC, "shin_deck.txt"), encoding="latin-1").read().splitlines()
    ours = open(os.path.join(RUNS, "vedant", "pierce.in"), encoding="latin-1").read().splitlines()
    vt, ot = ved[-5:], ours[-6:]
    s.image(lines_image([("Vedant's deck (his writer): end", ved[3:9] + ["..."] + vt +
                          ["   <- no closing '/' : target.exe stops"],
                          {len(ved[3:9]) + 1 + len(vt): (239, 112, 112)}),
                         ("our deck writer: end", ours[3:9] + ["..."] + ot,
                          {len(ours[3:9]) + len(ot): (78, 201, 160)})],
                        os.path.join(WORK, "decks.png")),
            "Step two: Vedant's code writes the same kind of deck, but it did not run. It never closes "
            "the cathode block with a slash, so target dot exe stops at the end of the file, and its "
            "solver settings crash the program. Our deck writer adds the slash, uses sir's settings, "
            "and keeps the grid within the program's limit of five hundred cells.")
    s.card("The pipeline",
           "gun specs (V, I, rw, Jc) -> Vedant's exact synthesis -> his electrode construction -> "
           "our deck writer (fixed) -> target.exe -> our reader (pierce.log, pierce.plt) -> compare "
           "with the design -> tune the electrode angle and anode -> repeat",
           "So the pipeline is: gun specifications, Vedant's synthesis and electrode construction, "
           "our fixed deck writer, target dot exe, our reader of its output files, a comparison with "
           "the design, and if needed, tuning and running again.", kicker="How it fits together")
    s.terminal(rec("v3_vedant_run"),
               f"Step three: one command does all of it for a gun. For Vedant's design, target dot exe "
               f"finds {cur('vedant'):.1f} amps against the designed one hundred twenty: twenty six "
               "and a half percent too much.",
               title="python -m src.sim.deck --V 150000 --I 120 --rw 9.8 --Jc 5 --run")
    s.frames(trajectory_animation(os.path.join(RUNS, "vedant"), "target.exe output: Vedant's deck",
                                  [f"current {cur('vedant'):.1f} A (design 120 A)"]),
             "This is target dot exe's own output, animated: the electrodes, the field lines, then the "
             "electrons. The current settles well above the design line.", min_dur=9)
    s.terminal(rec("v3_experiments"),
               "Step four: every version of the gun, simulated together. Our A N N guns come out plus "
               "twenty seven and plus twenty two percent, Vedant's plus twenty six and a half, the "
               "Tiwary Basu electrodes minus fourteen and a half. Whoever supplies the angle, the "
               "error is in the electrode shape.", title="python -m src.sim.experiments")
    s.image(montage([("ann_m1", "our code: ANN M1"), ("ann_e", "our code: ANN-E"),
                     ("vedant", "Vedant's code"), ("vedant_raw", "Vedant, uncorrected cathode"),
                     ("tiwary_basu", "Tiwary-Basu electrodes"), ("psi_57.5", "electrode at 57.5 deg"),
                     ("psi_72.5", "electrode at 72.5 deg"), (B, "tuned: angle + anode")],
                    os.path.join(WORK, "montage5.png")),
            "Sir's own plotter script on each of these runs.",
            title="sir's pierce_plotter.py on every run")
    s.terminal(rec("v3_tune"),
               "Step five: target dot exe in the loop. The electrode angle and the anode position are "
               "changed, run, measured, and changed again, until the current and the beam waist match "
               "the design.", title="python -m src.sim.experiments --tune")
    s.frames(trajectory_animation(os.path.join(RUNS, B),
                                  f"target.exe output: tuned gun, {best['psi_deg']:.1f} deg, anode "
                                  f"+{best['anode_shift_mm']:.2f} mm",
                                  [f"current {cur(B):.1f} A (design 120 A)",
                                   f"waist {best['compare']['sim_waist_mm_refined']:.2f} mm "
                                   "(design 9.8 mm)"]),
             f"The result: {cur(B):.1f} amps, within one percent of the design, a waist within two "
             "percent, and no electron hits the gun.", min_dur=9)

    def rotate(p):
        p.wait_for_timeout(600)
        p.mouse.move(640, 380)
        p.mouse.down()
        p.mouse.move(750, 395, steps=50)
        p.mouse.up()

    s.browser("file:///" + os.path.join(RUNS, B, "plot_index1_Normalized_Potential.html").replace("\\", "/"), [
        ("Sir's other script, new 3 D, shows target dot exe's potential over the whole grid: zero "
         "at the cathode, rising to the anode voltage.", rotate),
    ])
    s.browser(URL + "#sim", [
        ("All the runs are on the website's Simulation tab.", lambda p: highlight(p, "#sim-stats")),
        ("Each run against the design.", lambda p: highlight(p, "#sim-table")),
        ("And the trajectories of any run.", lambda p: highlight(p, "#sim-svg")),
    ])
    s.card("Summary",
           "Sir's run reproduced exactly; Vedant's deck fixed so target.exe runs it; one command "
           "from gun specs to a simulated, compared gun; every gun ~25 % over the design current "
           "until the electrodes are tuned with target.exe in the loop: then -0.7 % current, "
           "-1.9 % waist.",
           "In short: we reproduce sir's own run exactly, we fixed Vedant's deck so target dot exe "
           "runs it, and one command takes any gun from its specifications to a simulated result. "
           "Every synthesised gun drew about twenty five percent too much current, until the "
           "electrodes were tuned with target dot exe in the loop.", kicker="End")
    return s.render(os.path.join(OUT, "demoB_our_code_on_target_exe.mp4"))


# =============================================================================
def video6():
    s = Scenes(WORK, "v6")
    s.card("Our website: the Pierce gun, and the beam-focusing electrode by Tiwary & Basu",
           "The Predict tab keeps the gun it always drew, and adds the beam-focusing electrode and "
           "its angle from Tiwary and Basu (1988).",
           "This demo shows our website. The Predict tab keeps the Pierce gun it always drew, and now "
           "also calculates the beam focusing electrode and its angle by Tiwary and Basu's method.",
           kicker="Demo: website")

    def gun1(p):
        slow_fill(p, "#in-P", "2", 40)
        slow_fill(p, "#in-rw", "1.5", 40)
        slow_fill(p, "#in-C", "40", 40)
        highlight(p, ".readout")

    def gun2(p):
        slow_fill(p, "#in-P", "0.3", 40)
        slow_fill(p, "#in-rw", "0.62", 40)
        slow_fill(p, "#in-C", "207.6", 40)
        highlight(p, "#bfe-card")

    s.browser(URL + "#predict", [
        ("The inputs are perveance, beam waist radius and convergence ratio. For perveance two, "
         "waist one point five millimetres and convergence forty, our network predicts a half beam "
         "cone angle of forty point nine degrees.", gun1),
        ("Our Pierce gun drawing, as before: the cathode, the anode and the beam converging to its "
         "waist, with the geometry that follows from the angle.", lambda p: highlight(p, "#gun-svg")),
        ("The geometry table: cathode radius, radius of curvature, anode position and spacing.",
         lambda p: highlight(p, "#geom-rows")),
        ("The new card adds the beam focusing electrode, from Tiwary and Basu's paper. It is the "
         "zero volt equipotential of the ideal electron flow, drawn for this gun.",
         lambda p: highlight(p, "#bfe-svg")),
        ("The electrode leaves the cathode at sixty seven point five degrees to the beam edge. For "
         "this gun that is twenty six point six degrees to the axis. The table also gives where the "
         "electrode ends and the anode face.", lambda p: highlight(p, "#bfe-rows")),
        ("Change the inputs, and both the gun and the focusing electrode update. Here a high "
         "convergence gun gives twenty eight point four degrees, so the electrode is at thirty nine "
         "point one degrees to the axis.", gun2),
    ])
    s.terminal(rec("v1_paper1"),
               "The paper's method is checked on the paper's own three guns: the electrode leaves the "
               "cathode at sixty seven point five degrees, and our anode curve hits the anode sphere "
               "exactly.", title="python demo/paper1_check.py")
    s.image(os.path.join(REC, "paper1_overlay.png"),
            "And our curves, in red, drawn over the paper's own figure two.",
            title="Tiwary & Basu (1988) Fig. 2, with our curves in red")
    s.card("Summary", "Same Pierce gun as before, plus the beam-focusing electrode and its angle "
           "from Tiwary & Basu, computed in the browser from the input parameters.",
           "So the website keeps our Pierce gun, and adds the beam focusing electrode and its angle "
           "from Tiwary and Basu, computed from the input parameters.", kicker="End")
    return s.render(os.path.join(OUT, "demoC_website_gun_and_tiwary_basu.mp4"))


def video7():
    from anim import trajectory_animation
    from src.sim import lktron as lk
    og = os.path.join(CLONE, "simulation", "runs", "ourgun")
    summ = json.load(open(os.path.join(CLONE, "reports", "simulation", "ourgun", "summary.json")))
    tv = {t["case"]: t["variant"] for t in summ["tuned"]}

    def cur(n):
        return lk.parse_log(os.path.join(og, n, "pierce.log"))

    s = Scenes(WORK, "v7")
    s.card("Our gun, simulated in sir's PIERCE (target.exe)",
           "Our network's angle + the gun synthesised by Vaughan's equations + Tiwary & Basu's "
           "focus electrode, written as a pierce.in deck and run in target.exe.",
           "This demo runs our gun in sir's simulator, target dot exe. Our gun is our network's "
           "angle, the gun built around it by Vaughan's equations, and Tiwary and Basu's beam "
           "focusing electrode.", kicker="Demo: our gun in target.exe")
    s.card("What target.exe does",
           "Reads pierce.in (electrode points and voltages); solves the electric field on a grid; "
           "emits electrons from the cathode; traces them; adds their space charge; repeats until "
           "it converges. Reports the current, the beam waist and lost electrons.",
           "Target dot exe reads the electrode outline, solves the electric field, emits and traces "
           "the electrons, adds their own charge, and repeats until it converges. It tells us the "
           "current the gun really draws, where the beam focuses, and whether electrons hit the "
           "metal.", kicker="Background")
    s.terminal(rec("v5_reference"),
               "First, a check: sir's own deck run through our code reproduces his result exactly, "
               "one hundred twenty one point seven four amps.", title="python demo/run_reference.py")
    s.terminal(rec("v7_ourgun"),
               "Now our gun, on sir's klystron gun and on the paper's seven held out real guns. On "
               "sir's gun, our gun draws three percent more than designed. On the paper's guns it "
               "draws eight to fifty six percent more, and two low convergence guns fail. Then the "
               "anode position is tuned with target dot exe in the loop, and five guns match the "
               "design within about one percent.", title="python -m src.sim.ourgun --tune")
    sg = cur("sir_gun")
    s.frames(trajectory_animation(os.path.join(og, "sir_gun"),
                                  "target.exe: our gun on sir's klystron gun",
                                  [f"current {sg['cathode_current_A']:.1f} A (design 120 A)",
                                   "focus electrode: Tiwary & Basu curve"]),
             f"Target dot exe's output for our gun on sir's klystron gun: {sg['cathode_current_A']:.1f} "
             "amps against one hundred twenty designed, and no electron hits the gun.", min_dur=9)
    c15 = cur("case15")
    s.frames(trajectory_animation(os.path.join(og, "case15"), "target.exe: our gun, paper case 15",
                                  [f"current {c15['cathode_current_A']*1000:.1f} mA, design "
                                   f"{1.01e-6 * 1e4 ** 1.5 * 1000:.1f} mA"]),
             "A smaller gun from the paper, case fifteen: twenty six percent more current than "
             "designed.", min_dur=7)
    t15 = cur(tv["case15"])
    s.frames(trajectory_animation(os.path.join(og, tv["case15"]),
                                  "target.exe: case 15 after tuning the anode (+1.10 mm)",
                                  [f"current {t15['cathode_current_A']*1000:.1f} mA, design "
                                   f"{1.01e-6 * 1e4 ** 1.5 * 1000:.1f} mA"]),
             "After moving the anode one point one millimetres, with target dot exe in the loop: the "
             "current matches the design within zero point two percent.", min_dur=8)
    s.image(montage([("ourgun/sir_gun", "our gun: sir's klystron gun"),
                     ("ourgun/case15", "our gun: case 15"),
                     ("ourgun/case26", "our gun: case 26 (fails)")],
                    os.path.join(WORK, "montage7.png")),
            "Sir's plotter on our runs. Case twenty six, a low convergence gun, fails: the beam hits "
            "the gun, so this construction does not suit such guns.",
            title="sir's pierce_plotter.py on our gun")
    s.card("Summary",
           "Our gun runs in target.exe; untuned it draws 3 % to 56 % more current than designed; "
           "with the anode tuned by target.exe in the loop, 5 guns match the design perveance within "
           "1.3 % with no electron lost. Low-convergence guns fail; the simulated waist is smaller "
           "than designed.",
           "In short: our gun runs in sir's simulator. Untuned, it draws more current than designed; "
           "with the anode tuned by the simulator in the loop, five guns match the design within "
           "about one percent, with no electron lost. Low convergence guns fail, and the simulated "
           "waist is smaller than designed, which is the next thing to improve.", kicker="End")
    return s.render(os.path.join(OUT, "demoD_our_gun_in_target_exe.mp4"))


# =============================================================================
def _slide(sel, idxs, ms=650):
    def act(p):
        for i in idxs:
            p.evaluate("([s, v]) => { const e = document.querySelector(s); e.value = v; "
                       "e.dispatchEvent(new Event('input')); }", [sel, i])
            p.wait_for_timeout(ms)
    return act


def _scroll(sel):
    def act(p):
        p.evaluate("(s) => document.querySelector(s).scrollIntoView({block: 'start'})", sel)
        p.wait_for_timeout(300)
    return act


def _only_new_sim(p):
    """Show only the new sections of the Simulation tab (the older studies below are not
    part of these demos)."""
    p.add_style_tag(content="#live-card ~ * { display: none !important; }")


def _sweep_steps(name, key):
    """(run_dir, label, focus, aim, info line) for every ok row of one sweep."""
    data = json.load(open(os.path.join(CLONE, "reports", "simulation", "sweeps", f"{name}.json")))
    out = []
    for r in data[key]:
        if not r["ok"]:
            continue
        d = os.path.join(RUNS, "sweeps", name, r["run"])
        hit = f"{r['lost_in_gun']} electrons hit the gun" if r["lost_in_gun"] else "no electron hits the gun"
        foc = f"focus {r['throw_mm']:.1f} mm, waist {r['waist_mm']:.2f} mm" if r["throw_mm"] else "no focus"
        if key == "anode":
            lab = f"anode shift {r['anode_shift_mm']:+g} mm   (spacing d = {r['cathode_anode_mm']:.2f} mm)"
            line = (f"microperveance {r['sim_microperveance']:.3f}  (1/d^2 predicts "
                    f"{r['P_child_1_over_d2']:.3f})   {foc}   {hit}")
            aim = None
        else:
            lab = f"cathode Rc = {r['RcT_mm']:.1f} mm   (beam angle {r['thetaT_deg']:.1f} deg)"
            line = f"aim point {r['aim_point_mm']:.1f} mm   {foc}   {hit}"
            aim = r["aim_point_mm"]
        out.append((d, lab, r["throw_mm"], aim, line))
    return data, out


def video8():
    """C1: the complete website (our gun, Tiwary-Basu electrode, simulations, data, results)."""
    s = Scenes(WORK, "v8")
    s.card("Optimisation of a convergent electron gun using an ANN: the complete website",
           "Neural network for the beam angle -> Pierce gun synthesis -> Tiwary & Basu focus "
           "electrode -> simulation in target.exe. EEE F266: Shivansh Shekher Ojha, Satwik "
           "Shrivastava; supervisor Dr. Lalit Kumar.",
           "This is the complete project, on our website. A neural network gives the beam angle, "
           "the Pierce gun is synthesised around it, the beam focusing electrode comes from Tiwary "
           "and Basu, and every gun is checked in sir's simulator, target dot exe.",
           kicker="Demo C1")
    s.terminal(rec("v1_train"),
               "The network is trained from scratch: the paper's three, three, two, one network, "
               "Levenberg Marquardt, on twenty three real guns. On the seven held out guns the error "
               "is seven point nine three percent.", title="python -m src.ann.train --model m1")

    def gun1(p):
        slow_fill(p, "#in-P", "2", 40)
        slow_fill(p, "#in-rw", "1.5", 40)
        slow_fill(p, "#in-C", "40", 40)
        highlight(p, ".readout")

    def gun2(p):
        slow_fill(p, "#in-P", "0.3", 40)
        slow_fill(p, "#in-rw", "0.62", 40)
        slow_fill(p, "#in-C", "207.6", 40)
        highlight(p, "#bfe-card")

    s.browser(URL + "#predict", [
        ("Predict tab. Perveance two, waist one point five millimetres, convergence forty: the "
         "network gives a half beam cone angle of forty point nine degrees.", gun1),
        ("Our Pierce gun, drawn from that angle: the cathode, the anode and the beam converging to "
         "its waist.", lambda p: highlight(p, "#gun-svg")),
        ("The geometry that follows: cathode radius, radius of curvature, the anode and the "
         "spacing.", lambda p: highlight(p, "#geom-rows")),
        ("The beam focusing electrode by Tiwary and Basu: the zero volt equipotential of the ideal "
         "flow, drawn for this gun.", lambda p: highlight(p, "#bfe-svg")),
        ("It leaves the cathode at sixty seven point five degrees to the beam edge, here twenty six "
         "point six degrees to the axis.", lambda p: highlight(p, "#bfe-rows")),
        ("Change the inputs and everything updates: a high convergence gun gives twenty eight point "
         "four degrees.", gun2),
    ])
    s.image(os.path.join(REC, "paper1_overlay.png"),
            "The Tiwary and Basu construction is checked on the paper's own guns: our curves, in "
            "red, over the paper's figure two.", title="Tiwary & Basu (1988) Fig. 2, our curves in red")

    def rc_mode(p):
        p.select_option("#sw-gun", "case15")
        p.click("#sw-mode-rc")
        _scroll("#sw-gun")(p)
        _slide("#sw-step", [8, 6, 4, 2, 0], 800)(p)

    s.browser(URL + "#sim", [
        ("The Simulation tab. Each position of this slider is a real target dot exe run of our gun "
         "on sir's klystron gun.", _scroll("#sw-gun")),
        ("Moving the anode towards the cathode draws more current; moving it away draws less, and "
         "the focus moves down the tube.", _slide("#sw-step", [0, 2, 4, 6, 8, 9], 700)),
        ("The charts: target dot exe's current follows the one over d squared law near the design "
         "point.", _scroll("#sw-chart1")),
        ("Now the cathode's radius of curvature, on a smaller gun from the paper. A flatter cathode "
         "focuses further away, and at the ends electrons hit the gun, shown in orange.", rc_mode),
    ], setup=_only_new_sim)
    s.browser(URL + "#dataset", [
        ("The Dataset tab: the real guns from the paper's table two, which the network is trained "
         "and tested on.", None),
    ])
    s.browser(URL + "#method", [
        ("The Method tab explains the network, its training by Levenberg Marquardt, and the "
         "physics behind the synthesis.", None),
    ])
    s.card("Summary",
           "Network angle -> Pierce gun -> Tiwary & Basu focus electrode, in the browser; every gun "
           "checked in target.exe, with anode shift and cathode curvature studies.",
           "So the website gives the whole chain: the network's angle, the Pierce gun, the Tiwary "
           "and Basu beam focusing electrode, and the target dot exe simulations with the anode and "
           "cathode studies.", kicker="End")
    return s.render(os.path.join(OUT, "demoC1_complete_website.mp4"))


def video9():
    """D1: our gun in target.exe, plus the anode-shift and cathode-Rc studies and the live run."""
    import subprocess
    import time
    import urllib.request
    from anim import sweep_flipbook, trajectory_animation
    from src.sim import lktron as lk
    og = os.path.join(CLONE, "simulation", "runs", "ourgun")
    summ = json.load(open(os.path.join(CLONE, "reports", "simulation", "ourgun", "summary.json")))
    tv = {t["case"]: t["variant"] for t in summ["tuned"]}

    def cur(n):
        return lk.parse_log(os.path.join(og, n, "pierce.log"))

    s = Scenes(WORK, "v9")
    s.card("Our gun in sir's PIERCE (target.exe): simulation, anode shift, cathode curvature",
           "Our network's angle + Vaughan's synthesis + Tiwary & Basu's focus electrode, run in "
           "target.exe; then what happens when the anode moves and when the cathode's radius of "
           "curvature changes; then a live run from the website.",
           "This demo runs our gun in sir's simulator, target dot exe. Then it shows what happens "
           "when the anode moves, and when the cathode's radius of curvature changes. Finally, a "
           "live run from the website on this computer.", kicker="Demo D1")
    s.terminal(rec("v5_reference"),
               "First, a check: sir's own deck, run through our code, reproduces his result exactly, "
               "one hundred twenty one point seven four amps.", title="python demo/run_reference.py")
    s.terminal(rec("v7_ourgun"),
               "Our gun on sir's klystron gun and on the paper's seven held out guns. Untuned, it "
               "draws three to fifty six percent more current than designed. With the anode tuned by "
               "target dot exe in the loop, five guns match the design within about one percent.",
               title="python -m src.sim.ourgun --tune")
    sg = cur("sir_gun")
    s.frames(trajectory_animation(os.path.join(og, "sir_gun"), "target.exe: our gun on sir's klystron gun",
                                  [f"current {sg['cathode_current_A']:.1f} A (design 120 A)",
                                   "focus electrode: Tiwary & Basu curve"]),
             f"Our gun on sir's klystron gun: {sg['cathode_current_A']:.1f} amps against one hundred "
             "twenty designed, and no electron hits the gun.", min_dur=8)
    t15 = cur(tv["case15"])
    s.frames(trajectory_animation(os.path.join(og, tv["case15"]),
                                  "target.exe: paper case 15 after tuning the anode (+1.10 mm)",
                                  [f"current {t15['cathode_current_A'] * 1000:.1f} mA, design "
                                   f"{1.01e-6 * 1e4 ** 1.5 * 1000:.1f} mA"]),
             "Paper case fifteen, after moving the anode one point one millimetres: the current "
             "matches the design within zero point two percent.", min_dur=7)

    # --- the new studies ---------------------------------------------------------
    s.card("What happens when the anode moves, and when the cathode's curvature changes",
           "Anode shift: the anode is moved along the axis, -4 to +6 mm; the cathode-anode spacing "
           "d changes, and the current should follow 1/d^2 (and Langmuir-Blodgett). Cathode Rc: the "
           "radius of curvature is changed at the same cathode disc and perveance; the anode is "
           "re-placed by the Langmuir-Blodgett equation. 19 target.exe runs per gun.",
           "Next, sir's questions. What happens when the anode moves? The spacing d changes, so "
           "the current should follow one over d squared. And what happens when the cathode's "
           "radius of curvature changes? The beam aims at a different point, so where it focuses, "
           "and where electrons may hit, changes.", kicker="Parametric study")
    s.terminal(rec("v8_sweeps"),
               "Nineteen target dot exe runs of our gun on sir's klystron gun. Moving the anode four "
               "millimetres closer raises the perveance from two point one three to three point one "
               "one; six millimetres away lowers it to one point three one. Near the design point "
               "this follows one over d squared. Changing the cathode's radius of curvature moves "
               "the waist, and the flattest cathode makes the beam hit the gun.",
               title="python -m src.sim.sweeps")
    _, st = _sweep_steps("sir_gun", "anode")
    s.frames(sweep_flipbook(st, "target.exe: our gun on sir's klystron gun, anode moved"),
             "Here are those runs, one after another. Watch the anode move: closer draws more "
             "current, further away draws less and the focus moves down the tube.", min_dur=12)
    s.image(os.path.join(CLONE, "reports", "figures", "sweep_anode_sir_gun.png"),
            "The current against the anode shift: target dot exe in blue, one over d squared in "
            "grey, Langmuir Blodgett in orange. Then where the beam focuses, and its waist.",
            title="Anode shift: current, focus point, waist")
    _, st = _sweep_steps("case15", "rc")
    s.frames(sweep_flipbook(list(reversed(st)),
                            "target.exe: paper case 15, cathode radius of curvature changed"),
             "Now the cathode's radius of curvature, on paper case fifteen. As the cathode gets "
             "flatter, its aim point and the real focus both move away, from thirty three to fifty "
             "two millimetres. At both ends electrons hit the gun, ringed in orange.", min_dur=12)
    s.image(os.path.join(CLONE, "reports", "figures", "sweep_rc_case15.png"),
            "Summarised: the focus in target dot exe lies well beyond the aim point, because of the "
            "beam's own space charge, and it moves with the radius of curvature.",
            title="Cathode radius of curvature: focus, waist, current")
    s.terminal(rec("v8_sweeps15"), "The same study on case fifteen, in the terminal.",
               title="python -m src.sim.sweeps --gun case15")

    # --- live run from the website, served on this computer ------------------------
    port = 8795
    log = os.path.join(WORK, "serve_log.txt")
    srv = subprocess.Popen([sys.executable, "serve_local.py", "--no-browser", "--port", str(port)],
                           cwd=CLONE, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONWARNINGS="ignore"))
    try:
        for _ in range(60):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=1)
                break
            except OSError:
                time.sleep(0.5)
        s.image(text_image(log, os.path.join(WORK, "serve_banner.png"), "START_HERE.bat"),
                "For sir's computer: double click start here dot bat. It starts a local server, "
                "finds target dot exe, and opens this link in the browser.",
                title="START_HERE.bat")

        def live(p):
            for k, v in (("#lv-P", "1.2"), ("#lv-rw", "0.76"), ("#lv-C", "111"), ("#lv-V", "10"),
                         ("#lv-dz", "1.942"), ("#lv-dth", "0")):
                p.fill(k, v)
                p.wait_for_timeout(150)
            _scroll("#live-card")(p)
            p.click("#lv-run")
            p.wait_for_function("document.getElementById('lv-status').textContent.startsWith('Done')"
                                " || document.getElementById('lv-status').textContent.includes('failed')",
                                timeout=180000)
            _scroll("#lv-svg")(p)

        s.browser(f"http://127.0.0.1:{port}/#sim", [
            ("Served this way, the Simulation tab gets one more panel: run our gun in target dot exe, "
             "now.", _scroll("#live-card")),
            ("Paper case twelve, with the anode moved one point nine four millimetres. Press run: the "
             "website builds our gun and runs target dot exe on this computer.", live),
            ("Done in seconds. The trajectories, the focus and the perveance, within half a percent "
             "of the design. All of target dot exe's files are kept, for sir's own plotters.",
             lambda p: p.wait_for_timeout(500)),
        ], setup=_only_new_sim)
    finally:
        srv.terminate()
    s.card("Summary",
           "Our gun runs in target.exe: untuned +3 % to +56 % current; anode tuned in the loop, 5 "
           "guns within 1.3 %. Moving the anode: current follows 1/d^2 near the design; focus and "
           "waist move with it. Cathode Rc: the focus follows the curvature, well beyond the aim "
           "point; too flat or too curved and electrons hit the gun. Low-convergence guns fail.",
           "In short: our gun runs in sir's simulator, and the anode tuned in the loop brings five "
           "guns within about one percent. Moving the anode changes the current as one over d "
           "squared, and moves the focus. The cathode's curvature moves the focus too, and at the "
           "extremes electrons hit the gun. And sir can run all of this himself with one double "
           "click.", kicker="End")
    return s.render(os.path.join(OUT, "demoD1_our_gun_target_exe_sweeps.mp4"))


# =============================================================================
def log_table_image(log, out, title):
    """pierce.log's iteration table, with the last major iteration boxed and its CURRENT and
    MICROPERV. values marked."""
    from src.sim.compare_log import last_major_iteration
    it = last_major_iteration(log)
    lines = it["table"]
    img = Image.new("RGB", (1240, 70 + 24 * len(lines) + 70), (12, 13, 17))
    d = ImageDraw.Draw(img)
    d.text((16, 10), title, font=F(SANS_B, 22), fill=INK)
    f = F(MONO, 18)
    cw = d.textlength("0", font=f)
    for i, ln in enumerate(lines):
        y = 50 + 24 * i
        last = i == len(lines) - 1
        if last:
            d.rectangle([10, y - 3, 1230, y + 22], outline=(255, 196, 0), width=3)
        d.text((16, y), ln[:110], font=f, fill=(255, 220, 120) if last else (200, 210, 225))
    hdr = lines[0]
    y = 50 + 24 * len(lines) + 8
    for key, txt in (("CURRENT (A)", "current"), ("MICROPERV.", "microperveance")):
        x = 16 + hdr.index(key) * cw
        d.text((x, y), "^ " + txt, font=F(SANS_B, 18), fill=(255, 196, 0))
    d.text((16, y + 28), f"last major iteration {it['iteration']}:  current {it['current_A']:g} A,  "
           f"microperveance {it['microperveance']:g}", font=F(SANS_B, 20), fill=(255, 220, 120))
    img.save(out)
    return out


def video10():
    """E: predicted current / perveance vs pierce.log, last major iteration (sir's task)."""
    s = Scenes(WORK, "v10")
    s.card("Predicted current and perveance vs target.exe",
           "Sir's task: compare the predicted current / perveance with target.exe's value in "
           "pierce.log, at the last major iteration.",
           "Sir's task: compare the current and perveance our code predicts with the value target "
           "dot exe reports in pierce dot log, at the last major iteration.", kicker="Demo E")
    s.card("Where the value is",
           "pierce.log has a table of PIERCE's major iterations: CURRENT (A) and MICROPERV. for each. "
           "The last row is the converged answer. Predicted: perveance = the design P our gun is "
           "built for; current = P x 1e-6 x V^1.5.",
           "Target dot exe iterates the field and the space charge. pierce dot log prints a row for "
           "every major iteration, with the current and the micro perveance. The last row is its "
           "final answer. Our prediction is the design perveance our gun was built for, and the "
           "current that follows from it.", kicker="Method")
    s.image(log_table_image(os.path.join(CLONE, "reports", "simulation", "ourgun", "sir_gun", "pierce.log"),
                            os.path.join(WORK, "logtable_sir_gun.png"),
                            "pierce.log of our gun on sir's klystron gun (150 kV): iteration table"),
            "Here is pierce dot log for our gun on sir's klystron gun. The current starts at three "
            "hundred thirty three amps and settles. The last major iteration, number twenty, gives "
            "one hundred twenty three point six amps and micro perveance two point one two eight, "
            "against the predicted one hundred twenty amps and two point zero six six.",
            title="pierce.log: last major iteration")
    s.terminal(rec("v10_compare"),
               "The script reads that last row from every run's pierce dot log, and compares. Sir's "
               "own deck gives plus one point five percent. Our gun as built gives plus three "
               "percent on sir's gun and eight to fifty six percent on the paper's guns; two low "
               "convergence guns did not converge. After tuning the anode with target dot exe in the "
               "loop, the five tuned guns are within one point three percent of the prediction.",
               title="python -m src.sim.compare_log --show sir_gun")
    s.image(os.path.join(CLONE, "reports", "figures", "compare_log.png"),
            "The same comparison as a chart: the perveance error for each gun, as built in blue, "
            "and with the anode tuned in orange.",
            title="Perveance: pierce.log vs predicted")
    s.card("Result",
           "pierce.log (last major iteration) vs predicted: sir's deck +1.5 %; our gun as built +3 % "
           "(sir's gun) to +56 %; two low-convergence guns fail. Anode tuned with target.exe in the "
           "loop: sir's gun -1.3 %, case 6 +0.1 %, case 12 -0.4 %, case 15 +0.2 %, case 17 +0.7 %. "
           "Table: reports/simulation/compare_log.md.",
           "So target dot exe's current and perveance, at the last major iteration, match our "
           "prediction within one point three percent once the anode is tuned. The full table is "
           "in reports, simulation, compare log dot m d.", kicker="End")
    return s.render(os.path.join(OUT, "demoE_predicted_vs_pierce_log.mp4"))


if __name__ == "__main__":
    for k in WHICH:
        print("built", {"1": video1, "2": video2, "3": video3, "4": video4, "5": video5, "6": video6, "7": video7, "8": video8, "9": video9, "10": video10}[k]())
