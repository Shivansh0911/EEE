"""
Animate a real PIERCE run from its own output files: electrodes, then the
equipotentials, then the electron trajectories growing from the cathode, with
the current's convergence over PIERCE's iterations alongside. Colours as in
sir's pierce_plotter.py.
"""

import math

from PIL import Image, ImageDraw

from src.sim import lktron as lk
from videokit import BG, CAP_H, F, FPS, H, INK, INK2, MONO, SANS, SANS_B, W, ACC, GOOD, BAD

PLOT_BG = (247, 248, 250)


def _clip(seg, frac):
    """First `frac` of a polyline by arc length."""
    if frac >= 1:
        return seg
    L = [0.0]
    for (a, b), (c, d) in zip(seg, seg[1:]):
        L.append(L[-1] + math.hypot(c - a, d - b))
    target = frac * L[-1]
    out = [seg[0]]
    for i in range(1, len(seg)):
        if L[i] <= target:
            out.append(seg[i])
        else:
            a, b = seg[i - 1]
            c, d = seg[i]
            t = (target - L[i - 1]) / max(L[i] - L[i - 1], 1e-12)
            out.append((a + t * (c - a), b + t * (d - b)))
            break
    return out


def trajectory_animation(run_dir, title, info_lines, design_A=120.0):
    plot = lk.parse_plt(f"{run_dir}/pierce.plt")
    log = lk.parse_log(f"{run_dir}/pierce.log")
    hist = log.get("current_history_A", [])
    bnd, trj, eqp = plot[1], plot[2], plot[3]
    zs = [p[0] for s in bnd for p in s]
    rs = [p[1] for s in bnd for p in s]
    zmax, rmax = max(zs), max(rs)
    area = (24, 56, W - 24, H - CAP_H - 150)                      # gun panel
    k = min((area[2] - area[0]) / zmax, (area[3] - area[1]) / rmax)
    ox = area[0] + ((area[2] - area[0]) - zmax * k) / 2
    oy = area[1] + rmax * k
    P = lambda z, r: (ox + z * k, oy - r * k)

    def factory(dur):
        n = int(dur * FPS) + 2
        for i in range(n):
            t = i / max(1, n - 1)
            img = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(img)
            d.text((24, 14), title, font=F(SANS_B, 24), fill=INK)
            d.rectangle([ox - 6, area[1] - 6, ox + zmax * k + 6, oy + 6], fill=PLOT_BG)
            fb = min(1.0, t / 0.15)
            fe = min(1.0, max(0.0, (t - 0.15) / 0.2))
            ft = min(1.0, max(0.0, (t - 0.35) / 0.5))
            if fe > 0:
                g = int(255 - fe * (255 - 44)), int(255 - fe * (255 - 160)), int(255 - fe * (255 - 44))
                for s in eqp:
                    if len(s) > 1:
                        d.line([P(*p) for p in s], fill=g, width=1)
            if ft > 0:
                for s in trj:
                    c = _clip(s, ft)
                    if len(c) > 1:
                        d.line([P(*p) for p in c], fill=(214, 39, 40), width=1)
            for s in bnd:
                c = _clip(s, fb)
                if len(c) > 1:
                    d.line([P(*p) for p in c], fill=(0, 0, 0), width=3)
            # convergence inset: PIERCE's current per iteration
            y0, y1 = H - CAP_H - 128, H - CAP_H - 14
            x0, x1 = 60, 520
            d.text((x0, y0 - 6), "PIERCE iterations: cathode current", font=F(SANS, 16), fill=INK2)
            if hist:
                lo, hi = min(hist + [design_A]) * 0.95, max(hist + [design_A]) * 1.05
                X = lambda j: x0 + j / max(1, len(hist) - 1) * (x1 - x0)
                Y = lambda v: y1 - (v - lo) / (hi - lo) * (y1 - y0 - 24)
                d.line([X(0), Y(design_A), X(len(hist) - 1), Y(design_A)], fill=ACC, width=1)
                d.text((x1 + 8, Y(design_A) - 9), f"design {design_A:g} A", font=F(SANS, 14), fill=ACC)
                m = max(1, int(len(hist) * min(1.0, t / 0.85)))
                d.line([(X(j), Y(v)) for j, v in enumerate(hist[:m])], fill=(239, 112, 112), width=3)
                d.text((X(m - 1) + 6, Y(hist[m - 1]) - 22), f"{hist[m - 1]:.1f} A",
                       font=F(MONO, 15), fill=INK)
            yy = y0 - 4
            for ln in info_lines if t > 0.85 else []:
                d.text((640, yy), ln, font=F(SANS, 18), fill=INK)
                yy += 26
            yield img

    return factory
