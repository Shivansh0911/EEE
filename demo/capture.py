"""
Run a shell command for real and record every output line with its time.

    python demo/capture.py <workdir> <out.json> -- <command ...>

The demo videos replay these recordings (sped up, and labelled as such); no
output in a video is typed by hand.
"""

import json
import os
import subprocess
import sys
import time


def main():
    workdir, out = sys.argv[1], sys.argv[2]
    cmd = sys.argv[sys.argv.index("--") + 1:]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONWARNINGS="ignore",
               MPLBACKEND="Agg")
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         env=env, text=True, encoding="utf-8", errors="replace")
    lines = []
    for line in p.stdout:
        lines.append([round(time.time() - t0, 3), line.rstrip("\n")])
    p.wait()
    rec = {"command": " ".join(cmd), "cwd": workdir, "exit": p.returncode,
           "seconds": round(time.time() - t0, 2), "lines": lines}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    print(f"{out}: {len(lines)} lines, {rec['seconds']} s, exit {p.returncode}")


if __name__ == "__main__":
    main()
