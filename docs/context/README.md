# Context files — where they came from

These files arrived loose in the repo root on 2026-10-29 with **rotated names**:
each file's name described a different file's content. They were placed here by
what they actually contain, verified by reading them, not by their names.

| Arrived as | Actually contains | Now at |
|---|---|---|
| `VEDANT_WORK.md` | Vedant's solver, `pierce_gun_design.py` (1561 lines, Python) | `src/physics/vendor/pierce_gun_design_fixed.py` |
| `pierce_gun_design_fixed.py` | Our midsem report (Markdown) | `docs/context/midsem_report.md` |
| `midsem_report.md` | Vedant's full Claude chat log (146 KB) | `docs/context/vedant_chat_full.txt` — **git-ignored**, see below |
| `satwik_colab.ipynb` | The "What Vedant built" summary (Markdown) | `docs/context/VEDANT_WORK.md` |
| `already_built_integration_branch.zip` | Not a zip: a 1.4 KB Markdown project note | `docs/context/PROJECT_OURS.md` |

Consequences, stated plainly:

- **The "already built integration branch" (94 tests, lb.py, kf.py, …) was not
  in any of these files.** Nothing could be unzipped, reviewed or re-run. The
  integration on branch `integration/vaughan-moo` was built from scratch against
  the brief; every number in it was computed in this repo.
- **Satwik's Colab notebook** was not among the loose files either, but the repo
  already held it as `notebooks/ANN_Testing.ipynb`. It was renamed (content
  untouched) to `notebooks/satwik_colab_reference.ipynb`.
- **Vedant's chat log** is a third party's private session. It is kept on disk
  as evidence for where his validation numbers come from, but git-ignored so it
  is not published with the repo. Ask him before committing it.
