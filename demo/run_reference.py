"""
Run sir's own reference deck through our PIERCE driver and print what we read
back, next to sir's own recorded run. Used in the demo videos.

    python demo/run_reference.py
"""

import tempfile

from src.sim import lktron as lk

print("sir's deck:", lk.REFERENCE_DECK)
print("running target.exe (LKTRON PIERCE) on it through src/sim/lktron.py ...")
with open(lk.REFERENCE_DECK, encoding="ascii") as f:
    deck = f.read()
with tempfile.TemporaryDirectory() as d:
    r = lk.run(deck, d)
print(f"  iterations          {r['iterations']}  (converged: {r['converged']})")
print(f"  cathode current     {r['cathode_current_A']:.4f} A")
print(f"  microperveance      {r['microperveance']:.3f}")
print(f"  beam throw          {r['throw_mm']} mm")
print(f"  beam waist radius   {r['waist_mm']} mm")
print(f"  electrons lost      {r['intercepted_A']} A on the electrodes")
print(f"  run time            {r['wall_seconds']:.0f} s")
print("sir's own run of this deck: 121.7416 A, throw 85.5 mm, waist 7.4336 mm")
same = (abs(r["cathode_current_A"] - 121.7416) < 1e-4 and r["throw_mm"] == 85.5
        and abs(r["waist_mm"] - 7.4336) < 1e-4)
print("-> identical: our driver runs sir's program correctly" if same else "-> DIFFERENT")
