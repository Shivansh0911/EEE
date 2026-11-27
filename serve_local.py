"""
Run the project website on this computer, with live target.exe simulations.

    double-click START_HERE.bat            (Windows; does the Python setup too)
    python serve_local.py                  (if Python and the packages are ready)
    python serve_local.py --exe "D:\\LKTRON\\target.exe" --port 8765

Opens http://127.0.0.1:8765 in the browser. The website is the same as the
hosted one; in addition, the Simulation tab gets a "Run our gun in target.exe
now" panel: it builds our gun (network angle + Vaughan synthesis + Tiwary-Basu
focus electrode) for the inputs typed, runs target.exe here, and draws the
result. Every run's files (pierce.in, pierce.log, pierce.plt, ...) are kept in
simulation/runs/local/<run>/ so sir's pierce_plotter.py and new3d.py can be
run on them.

target.exe is looked for in this order: --exe, the PIERCE_EXE environment
variable, simulation/lktron/target.exe, this folder, the folder above it.
Only this computer can reach the server (it listens on 127.0.0.1).
"""

import argparse
import http.server
import json
import os
import socket
import sys
import threading
import time
import traceback
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")


def find_exe(cli=None):
    for p in (cli, os.environ.get("PIERCE_EXE"),
              os.path.join(ROOT, "simulation", "lktron", "target.exe"),
              os.path.join(ROOT, "target.exe"),
              os.path.join(os.path.dirname(ROOT), "target.exe")):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    return None


class Handler(http.server.SimpleHTTPRequestHandler):
    exe = None
    lock = threading.Lock()          # one target.exe run at a time
    counter = [0]

    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def log_message(self, fmt, *args):          # keep the terminal readable
        if "/api/" in (args[0] if args else ""):
            sys.stdout.write(time.strftime("%H:%M:%S ") + (fmt % args) + "\n")

    def _json(self, obj, code=200):
        b = json.dumps(obj, default=float).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self._json({"ok": True, "target_exe": self.exe})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/simulate"):
            return self._json({"ok": False, "error": "unknown endpoint"}, 404)
        try:
            n = int(self.headers.get("Content-Length", 0))
            q = json.loads(self.rfile.read(n) or b"{}")
            P, rw, C = float(q["P"]), float(q["rw"]), float(q["C"])
            V = float(q.get("V", 10e3))
            dz, dth = float(q.get("anode_shift", 0)), float(q.get("theta_offset", 0))
            if not (0 < P < 10 and 0 < rw < 100 and 1 < C < 1000 and 1e3 <= V <= 1e6):
                return self._json({"ok": False, "error": "inputs out of range"})
            from src.sim import ourgun
            with self.lock:
                self.counter[0] += 1
                name = time.strftime("%Y%m%d_%H%M%S") + f"_P{P:g}_rw{rw:g}_C{C:g}_dz{dz:+g}_dth{dth:+g}"
                print(f"\nrun {self.counter[0]}: P {P:g}, rw {rw:g} mm, C {C:g}, {V / 1e3:g} kV, "
                      f"anode {dz:+g} mm, angle {dth:+g} deg -> target.exe ...", flush=True)
                theta = ourgun.predict_theta(P, rw, C) + dth
                runs = os.path.join(ROOT, "simulation", "runs", "local")
                out = os.path.join(ROOT, "reports", "simulation", "local")
                res = ourgun.simulate(name, P, rw, C, V, anode_shift=dz, theta=theta, runs=runs, out=out)
            if not res["ok"]:
                print("  failed:", res["error"][:200])
                return self._json({"ok": False, "error": res["error"]})
            with open(os.path.join(out, name, "plot.json"), encoding="utf-8") as f:
                res["plot"] = json.load(f)
            res["folder"] = os.path.join(runs, name)
            c = res["compare"]
            print(f"  microperveance {c['sim_microperveance']:.3f} ({c['perveance_err_pct']:+.1f} %), "
                  f"lost in gun {c['lost_in_gun']}/{c['n_trajectories']}, files in {res['folder']}")
            clean = lambda v: None if isinstance(v, float) and v != v else v
            res["compare"] = {k: clean(v) for k, v in c.items()}
            return self._json(res)
        except Exception as e:                                   # report, keep serving
            traceback.print_exc()
            return self._json({"ok": False, "error": f"{type(e).__name__}: {e}"})


def free_port(start):
    for p in range(start, start + 20):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port")


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--exe", help="path to target.exe")
    a.add_argument("--port", type=int, default=8765)
    a.add_argument("--no-browser", action="store_true")
    args = a.parse_args()
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    exe = find_exe(args.exe)
    if exe:
        os.environ["PIERCE_EXE"] = exe
        from src.sim import lktron
        lktron.EXE = exe
    Handler.exe = exe
    port = free_port(args.port)
    url = f"http://127.0.0.1:{port}/#sim"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print("=" * 70)
    print(" Optimisation of a convergent electron gun using an ANN")
    print(" EEE F266 - Shivansh Shekher Ojha, Satwik  |  supervisor Dr. Lalit Kumar")
    print("=" * 70)
    print(f" target.exe : {exe or 'NOT FOUND - copy it into simulation/lktron/ and restart'}")
    print(f" website    : {url}")
    print(" (open the link above if the browser did not open; Ctrl+C here stops it)")
    print("=" * 70, flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("stopped")


if __name__ == "__main__":
    main()
