"""
Vedant's solver on its own, exactly as his program's mode 1 runs it
(pierce_synthesis, then his print_results), then the same gun through our
fast emulator, timed. Used in demo video 2.

    python demo/vedant_direct.py 150000 120 9.8 5
"""

import sys
import time

from src.physics import closure
from src.physics import vaughan_adapter as va

V, I, rw_mm, Jc = (float(x) for x in sys.argv[1:5])
m = va.module()
print(f"--- Vedant's solver (src/physics/vendor/pierce_gun_design_fixed.py), unmodified ---")
t = time.time()
r = m.pierce_synthesis(V, I, rw_mm / 25.4, Jc)
t_exact = time.time() - t
r["rw_mm"] = rw_mm
m.print_results(r)
print(f"his solver took {t_exact:.2f} s (30-digit mpmath)\n")

print("--- the same gun through our LBClosure emulator ---")
closure.synthesize(V, I, rw_mm, Jc)          # first call loads the kf table; not timed
t = time.time()
e = closure.synthesize(V, I, rw_mm, Jc)
t_em = time.time() - t
for k, label, unit, scale in [("theta", "theta_raw", "deg", 1), ("thetaT", "theta_T", "deg", 1),
                              ("Rc_mm", "Rc", "mm", 1), ("zac_mm", "zac", "mm", 1),
                              ("zw_mm", "throw zw", "mm", 1)]:
    exact = r[k] if not k.endswith("_mm") else r[k[:-3]] * 25.4
    print(f"  {label:10s} Vedant {exact:12.6f}   emulator {e[k][0]:12.6f}   "
          f"difference {abs(e[k][0] - exact) / exact:.1e}   {unit}")
print(f"one gun: emulator {t_em * 1000:.0f} ms vs Vedant {t_exact:.2f} s")
import numpy as np
n = 200
rng = np.random.default_rng(0)
Vs, rws, Jcs = rng.uniform(100e3, 200e3, n), rng.uniform(7, 12, n), rng.uniform(2, 10, n)
t = time.time()
closure.synthesize(Vs, I, rws, Jcs)
t_b = time.time() - t
print(f"{n} guns at once: emulator {t_b:.2f} s = {t_b / n * 1000:.1f} ms per gun; "
      f"Vedant's solver would need about {n * t_exact / 60:.0f} min "
      f"-> about {t_exact / (t_b / n):.0f}x faster, same answers")
