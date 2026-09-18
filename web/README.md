# web/ — the static site

Open `index.html`. That is the whole procedure. There is no build step, no npm,
no bundler, and nothing to install.

It works two ways, and both are requirements:

* **double-clicked off the filesystem**, with no server and no network;
* **dropped onto Netlify** (or any static host) as-is.

## Why plain `<script>` tags rather than ES modules

The brief asked for `<script type="module">`. That does not survive the first
requirement: browsers apply CORS to module imports, and every `file://` document
is its own opaque origin, so `import` from a sibling file is blocked in Chrome
and Firefox. The page would load and then do nothing. `fetch()` of a local JSON
file is blocked for the same reason.

So the site uses classic scripts, which `file://` does allow, with each file
attaching to a single `EGUN` namespace, and ships its data as generated `.js`
files that assign onto `window.EGUN_DATA`. The canonical `.json` files sit
beside them and are the ones to read or reuse — the `.js` mirrors exist only to
get past the `file://` restriction.

Everything is self-contained. No CDN, no Google Fonts, no analytics, no runtime
network call of any kind. The font stack is the system stack; the favicon is an
inline data URI; the figures are local PNGs.

## Layout

```
web/
├── index.html         markup for all five tabs
├── css/styles.css     dark theme, one accent, system fonts
├── js/
│   ├── ann.js         the forward pass — 23 numbers, three matrix products
│   ├── physics.js     gun geometry; eqs (2) and (4) plus Langmuir–Blodgett
│   ├── gun.js         the live SVG cross-section
│   └── app.js         tabs, controls, tables, wiring
├── data/              GENERATED — see src/export/site_data.py
├── figures/           GENERATED — copied from reports/figures/
└── netlify.toml
```

## Regenerating

`data/` and `figures/` are build products. After retraining:

```bash
python -m src.export.site_data
```

`js/ann.js` is checked against the Python implementation to < 1e-9 by
`tests/test_js_python_parity.py`, which runs it under Node against the same
model file the browser loads. Run `pytest` before deploying.
