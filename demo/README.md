# Demo videos

Three narrated videos, built from real runs (no output typed by hand):

1. `demo1_our_code.mp4`: training M1, the Stage-2 evaluation, the website (Predict tab
   with the old gun drawing, Dataset, Method, the Design tab with the new cathode and
   focus-electrode options), and the full test suite.
2. `demo2_our_code_plus_vedant.mp4`: Vedant's solver run on its own, our emulator of it,
   Gate G1, his deck writer, and the Design tab.
3. `demo3_target_exe_simulation.mp4`: our decks run in sir's PIERCE (`target.exe`),
   animated from its own output files, the tuned gun, sir's plotter, and the
   Simulation tab.

How they are made:

```bash
git clone --branch integration/vaughan-moo . <clone>          # demo runs never touch the repo
cp simulation/lktron/target.exe <clone>/simulation/lktron/
# record each run: python demo/capture.py <clone> <clone>/demo_rec/<name>.json -- <command>
python demo/make_videos.py <clone> <work_dir> <out_dir> 1 2 3
```

`capture.py` records every output line with its time; `videokit.py` replays those
recordings (sped up, labelled with the factor), drives `web/index.html` in Edge with
Playwright, narrates with the Windows "Zira" voice (captions carry the same text), and
joins scenes with ffmpeg (`pip install imageio-ffmpeg playwright`). `anim.py` animates a
PIERCE run from its own `pierce.plt` and `pierce.log`.
