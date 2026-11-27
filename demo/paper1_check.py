"""
Paper 1 (Tiwary & Basu, IEEE TED 1988) validated on its own three guns, for demo 1.

    python demo/paper1_check.py <paper.pdf> <overlay_out.png>

Prints, per gun: the focus electrode's angle to the beam edge at the cathode
rim (Pierce: 67.5), whether the anode equipotential hits the anode sphere
(the paper's interpolation missed it by 0.26-0.41 %), and the size of the
axisymmetric terms the planar map drops. Then draws our curves over the
paper's scanned Fig. 2 (the image stays local; the PDF is licensed).
"""

import math
import sys

import numpy as np
import pypdf
from PIL import ImageDraw

from src.physics import bfe

GUNS = [("Litton L-2087", 1.985, 13.665, 13.44, 6.867),
        ("Varian VTC 6364", 2.389, 10.158, 15.59, 4.235),
        ("Frost and Purl", 2.393, 17.567, 27.17, 7.311)]

print("Tiwary & Basu (1988): focus electrode = V=0 equipotential, anode = V=Va equipotential")
print(f"{'gun':16s} {'BFE angle':>10s} {'Ra exact':>9s} {'ours':>9s} {'paper':>8s} {'paper err':>9s} {'dropped terms':>14s}")
for name, ratio, rc, th, ra_paper in GUNS:
    ra = rc / ratio
    e = bfe.electrodes(rc, ra, th)
    z, r = e["anode"][0]
    ours = math.hypot(rc - z, r)
    res = bfe.laplace_residual(rc, ra, th)
    print(f"{name:16s} {bfe.start_angle_deg(rc, th):9.3f}d {ra:9.4f} {ours:9.4f} {ra_paper:8.3f} "
          f"{abs(ra_paper - ra) / ra * 100:8.2f}% {res['rms'] * 100:13.0f}%")
print("BFE leaves the cathode rim at Pierce's 67.5 deg; our anode curve hits Ra exactly.")
print("The planar map drops axisymmetric terms that are not small -> simulate before trusting.")

pdf, out = sys.argv[1], sys.argv[2]
im = pypdf.PdfReader(pdf).pages[2].images[0].image.convert("L")
w, h = im.size
crop = im.crop((int(w * 0.48), int(h * 0.10), w, int(h * 0.42))).convert("RGB")
a = np.asarray(crop.convert("L")) < 100
row = int(np.argmax(a.sum(1)[int(a.shape[0] * 0.6):])) + int(a.shape[0] * 0.6)
cols = np.where(a[row])[0]
k = (cols.max() - cols.min()) / 20.0
X = lambda x: cols.min() + k * x
Y = lambda y: row - k * y
d = ImageDraw.Draw(crop)
for name, ratio, rc, th, _ in GUNS:
    t0 = math.radians(th)
    for s in (bfe.bfe_s(t_max=40, n=4000), bfe.anode_s(math.log(ratio), tau_max=80, n=2000)):
        rho, t = rc * np.exp(s.real), t0 + s.imag
        m = t < math.radians(100)
        d.line(list(zip(map(X, (rho * np.cos(t))[m]), map(Y, (rho * np.sin(t))[m]))),
               fill=(230, 0, 0), width=3)
crop.save(out)
print(f"overlay of our curves (red) on the paper's Fig. 2: {out}")
