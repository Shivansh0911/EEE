"""
Driver for Dr. Lalit Kumar's LKTRON PIERCE simulator (`target.exe`).

PIERCE solves Poisson's equation on a 2-D axisymmetric grid, launches Child-law
trajectories from the cathode and iterates field and space charge to
self-consistency -- an EGUN-class gun code. It reads `pierce.in` from its
working directory and writes `pierce.log` (summary), `pierce.plt` (boundary,
trajectories, equipotentials), `pierce.3D` (potential and field maps) and more.
Its input format is the point deck Vedant's synthesis writes, which is why his
code and this simulator fit together.

This module:
  * writes a deck PIERCE accepts (`deck_text`). Two things differ from the
    deck Vedant's `print_gun_points` writes, both found by running PIERCE:
      - Vedant's deck never closes the &CATHODE namelist with "/", and PIERCE
        stops with "Fortran runtime error: End of file";
      - with Vedant's &CONTROL/&PRINTS/&PLOT values PIERCE crashes re-reading
        its own pierce.plt ("Bad value during integer read"); the values in
        sir's reference deck (simulation/lktron/reference_ceeri/pierce.in)
        work, so they are used here.
  * runs target.exe in a private directory (`run`);
  * parses pierce.log and pierce.plt (`parse_log`, `parse_plt`).

target.exe is sir's copyrighted binary and is git-ignored; it must be present at
simulation/lktron/target.exe to run anything here.
"""

import math
import os
import re
import shutil
import subprocess
import time

from ..ann import dataset as ds

# Where target.exe is: PIERCE_EXE overrides (serve_local.py sets it when sir's copy
# is elsewhere), otherwise simulation/lktron/target.exe.
EXE = os.environ.get("PIERCE_EXE") or os.path.join(ds.repo_root(), "simulation", "lktron", "target.exe")
REFERENCE_DECK = os.path.join(ds.repo_root(), "simulation", "lktron", "reference_ceeri",
                              "pierce.in")

# &CONTROL, &PRINTS and &PLOT exactly as in sir's reference deck.
SIR_CONTROL = (
    " &CONTROL\n"
    " IOPT = 2, ICLOS = 0, IOS = 0, IORDR=1, IPONT = 5, NMAX = 2000, EL =   0.00000000050, \n"
    " maxtr={maxtr}, lmax=20,LC=1,eqstep=0.1,1.,10., DELIN = 0.2 , BETA = 0.6,\n"
    " /\n"
    " &PRINTS\n"
    " IPOT = 2, IFLDZ = 2, IFLDR = 2, ISPC = 2, IRAY = 2,\n"
    " /\n"
    " &PLOT\n"
    " ITRPL = 2, IPOTPL = 2, ICDPL = 2, ISCDPL = 2,imesh= 0\n"
    " / \n"
)

NICE_H = (0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0)


def available():
    return os.path.exists(EXE)


MAX_INTERVALS = 500      # PIERCE: "MAXIMUM ALLOWED MESH INTERVALS = 500" (found by running it)


def choose_grid(rc_mm, zmax_mm=None, cells_across_cathode=50):
    """
    Grid step H: the largest 'nice' step with ~50 cells across the cathode
    radius (sir's CEERI deck: rc = 27.65 mm on H = 0.5 mm, 55 cells), coarsened
    if needed so the domain stays under PIERCE's 500-interval limit (with a
    4 % margin).
    """
    target = rc_mm / cells_across_cathode
    ok = [h for h in NICE_H if h <= target]
    h = ok[-1] if ok else NICE_H[0]
    if zmax_mm is not None:
        need = zmax_mm / (0.96 * MAX_INTERVALS)
        h = max(h, min(x for x in NICE_H if x >= need))
    return h


def _snap(v, h):
    return round(round(v / h) * h, 10)


def _row(code, z, r, volt, zc=0.0, rcen=0.0, rad=0.0, h=0.5, exact_z=False):
    """One boundary row in Vedant's column layout (print_gun_points)."""
    z_out = f"{z:.8f}" if exact_z else f"{_snap(z, h):.2f}"
    return (f"{code:>2},  {z_out:<16}  {_snap(r, h):<8.2f}  {_snap(volt, h):<12.4f}"
            f"  {_snap(zc, h):<12.6f}  {_snap(rcen, h):<10.5f}  {_snap(rad, h):.4f}")


def deck_text(rows, *, title, h, d, crad, rc, dz=None, dx=None, maxtr=None):
    """
    A complete pierce.in.

    rows: list of dicts with keys code, z, r, volt and optionally zc, rcen, rad,
          exact_z -- one per boundary point, in boundary order. `code` describes
          the segment from this point to the next: 1 straight, 0 circular arc
          (centre zc, rcen; radius rad), -1 gap (insulating break).
    Consecutive points that land on the same grid node are merged.
    """
    dz = 4.0 * h if dz is None else dz           # sir: DZ 2.0 at H 0.5
    dx = 1.2 * h if dx is None else dx           # sir: DX 0.6 at H 0.5
    if maxtr is None:
        maxtr = max(200, int(math.ceil(crad / h / 50.0)) * 50)
    lines, last = [], None
    for p in rows:
        key = (_snap(p["z"], h), _snap(p["r"], h))
        if key == last and not p.get("exact_z"):
            continue
        last = key
        lines.append(_row(p["code"], p["z"], p["r"], p["volt"], p.get("zc", 0.0),
                          p.get("rcen", 0.0), p.get("rad", 0.0), h, p.get("exact_z", False)))
    n = len(lines)
    head = (f" '{title[:60]}'\n &BOUND\n"
            f" H =   {h},{h}, MBP = {n}, NT = {n}, NGRD = 0, jMAG = 0, NMAG = 0, MAGORD = 2,\n"
            " /\n" + SIR_CONTROL.format(maxtr=maxtr))
    tail = (" &GRID\n /\n"
            f" &CATHODE CSHAPE= \"SPHERE\", EMISSION=\"CHILD\", NSEG = 1, OCAT = {d:.2f},\n"
            f" CRAD = {crad:.2f}, RMAX = {rc:.2f} ,DZ = {dz:g} ,DX =  {dx:g}\n /\n")
    return head + "\n".join(lines) + "\n" + tail


def rows_from_vedant(pts, meta, V, h):
    """Vedant's 15 points (compute_gun_points_mm) as deck rows, coded exactly as
    his print_gun_points codes them: pt 1 starts the cathode arc, pts 3 and 7
    start the two insulating gaps, pts 8+ are at anode potential."""
    rows = []
    for pid, _label, z, r in pts:
        if pid == 0:
            continue
        row = {"code": 1, "z": z, "r": r, "volt": 0.0 if pid <= 7 else V,
               "exact_z": pid == 2}
        if pid == 1:
            row.update(code=0, zc=meta["arc_centre_z"], rad=meta["Rc_mm"])
        elif pid in (3, 7):
            row["code"] = -1
        rows.append(row)
    return rows


class SimulationFailed(RuntimeError):
    pass


def run(deck, workdir, timeout=3600):
    """Write `deck` to workdir/pierce.in, run target.exe there, parse the result."""
    if not available():
        raise FileNotFoundError(f"PIERCE executable not found at {EXE}")
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "pierce.in"), "w", encoding="ascii", newline="\n") as f:
        f.write(deck)
    for name in ("pierce.log", "pierce.plt"):
        p = os.path.join(workdir, name)
        if os.path.exists(p):
            os.remove(p)
    t0 = time.time()
    p = subprocess.run([EXE], cwd=workdir, stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, timeout=timeout)
    wall = time.time() - t0
    with open(os.path.join(workdir, "stdout.txt"), "w", encoding="utf-8") as f:
        f.write(p.stdout + p.stderr)
    log = os.path.join(workdir, "pierce.log")
    res = parse_log(log) if os.path.exists(log) else {"completed": False}
    res["wall_seconds"] = wall
    res["exit_code"] = p.returncode
    if not res.get("completed"):
        text = p.stdout + p.stderr
        if os.path.exists(log):
            with open(log, encoding="latin-1") as f:
                text += f.read()
        err = [" ".join(ln.split()) for ln in text.splitlines() if "error" in ln.lower()
               or "line" in ln.lower()]
        raise SimulationFailed(f"PIERCE did not complete (exit {p.returncode}): "
                               + " | ".join(err[-3:]))
    return res


_ITER = re.compile(r"^\s+(\d+)\s+(\d+)\s+([\d.E+-]+)\s+([\d.]+)\s+(-?1)\s+(-?1)\s+"
                   r"([\d.E+-]+)\s+([\d.E+-]+)\s+([\d.]+)\s+([\d.]+)\s*$")


def parse_log(path):
    """Everything PIERCE reports about the beam, from pierce.log."""
    with open(path, encoding="latin-1") as f:
        text = f.read()
    out = {"completed": "PROBLEM RUNENDS" in text}
    its = [m.groups() for m in (_ITER.match(l) for l in text.splitlines()) if m]
    if its:
        last = its[-1]
        out.update(iterations=int(last[0]), potential_error=float(last[2]),
                   converged=last[5] == "1", current_A=float(last[6]),
                   microperveance=float(last[7]))
        out["current_history_A"] = [float(i[6]) for i in its]

    def num(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    out["throw_mm"] = num(r"BEAM THROW FROM CATHODE:\s*([\d.E+-]+)")
    out["waist_mm"] = num(r"BEAM WAIST RADIUS\s*:\s*([\d.E+-]+)")
    out["cathode_current_A"] = num(r"Cathode Current\s*=\s*([\d.E+-]+)")
    out["beam_current_A"] = num(r"Beam Current\s*=\s*([\d.E+-]+)")
    out["n_trajectories"] = num(r"NUMBER OF TRAJECTORIES\s*=\s*(\d+)", int)
    out["cathode_area_m2"] = num(r"TOTAL AREA OF CATHODE\s*=\s*([\d.E+-]+)")
    inter = re.findall(r"^\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s*$",
                       text.split("Intercepted Current")[-1].split("EQUIPOTENTIAL")[0], re.M)
    out["electrodes"] = [{"electrode": int(a), "potential_V": float(b),
                          "trajectories_hit": int(c), "intercepted_A": float(d)}
                         for a, b, c, d in inter]
    out["intercepted_A"] = sum(e["intercepted_A"] for e in out["electrodes"])
    env = re.search(r"BEAM EDGE PROFILE DATA:\s*\n\s*Z\s+R\s*\n(.*?)\n\s*BEAM THROW", text, re.S)
    out["envelope"] = ([[float(x) for x in ln.split()] for ln in env.group(1).strip().splitlines()
                        if len(ln.split()) == 2] if env else [])
    return out


def parse_plt(path, keep=(1, 2, 3)):
    """pierce.plt segments as {code: [[(z, r), ...], ...]} for the first plot
    (the gun cross-section). Codes: 1 boundary, 2 trajectory, 3 equipotential
    (sir's pierce_plotter.py: 4 magnetic field, 5 current density, 6 space charge).
    A code-0 header separates plots; only the first plot is returned."""
    with open(path, encoding="latin-1") as f:
        tokens = f.read().split()
    out = {k: [] for k in keep}
    i, plots = 0, 0
    while i + 3 < len(tokens):
        try:
            code, n = int(tokens[i]), int(tokens[i + 1])
        except ValueError:
            i += 1
            continue
        if code == 0:
            plots += 1
            if plots > 1 and any(out.values()):
                break
            i += 4
            continue
        if n == 1:
            seg = [(float(tokens[i + 2]), float(tokens[i + 3]))]
            i += 4
        else:
            vals = [float(t) for t in tokens[i + 4:i + 4 + 2 * n]]
            seg = list(zip(vals[0::2], vals[1::2]))
            i += 4 + 2 * n
        if code in out:
            out[code].append(seg)
    return out


def copy_outputs(workdir, dest, names=("pierce.in", "pierce.log", "pierce.plt", "stdout.txt")):
    os.makedirs(dest, exist_ok=True)
    for n in names:
        p = os.path.join(workdir, n)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dest, n))
