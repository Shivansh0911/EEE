# Deployment

The site is a folder of static files. There is no build step, no npm install, no
bundler and no backend. Deploying it is copying a directory.

---

## The fastest route: drag and drop

1. Open **<https://app.netlify.com/drop>** in a browser. No account is needed to
   publish; you will be offered one afterwards to keep the site.
2. Drag the **`web/`** folder from this repository onto the drop zone. The whole
   folder, not its contents, and not the repository root.
3. Wait a few seconds. Netlify returns a URL like
   `https://relaxed-something-a1b2c3.netlify.app`.
4. Open it and check the Predict tab draws a gun.

That URL is live and public immediately. To keep it beyond the session, sign in
when prompted and claim the site; to give it a readable name, use
**Site settings → Change site name**.

**Drag-and-drop ignores `web/netlify.toml`.** It publishes exactly what it is
given, so the security headers in that file do not apply on this route. Nothing
breaks without them — the page makes no network requests — but if you want them,
use the git route below.

---

## The durable route: connect the repository

Useful if the site should update when you push.

1. **Add new site → Import an existing project**, and pick the GitHub repo.
2. Netlify reads `web/netlify.toml`, which already says: base directory `web`,
   publish `.`, no build command. Accept what it offers.
3. Deploy.

If Netlify's UI asks for the settings directly rather than reading the file:

| Field | Value |
|---|---|
| Base directory | `web` |
| Build command | *(leave empty)* |
| Publish directory | `web` |

---

## Before you deploy

```bash
pytest                                  # includes the JS/Python parity test
python -m src.export.site_data          # only if you retrained since last time
```

The parity test is the one that matters here: it runs `web/js/ann.js` under Node
against the same model file the browser loads and asserts the two agree to
< 1e-9. If the browser and the paper are going to disagree, that is where you
find out, rather than in front of your supervisor.

Then open `web/index.html` directly in a browser — double-click it, no server —
and confirm the Predict tab still works. The site is built to run from
`file://`, and that check takes five seconds.

---

## What is deployed

| | |
|---|---|
| Backend | none |
| Runtime network requests | none |
| External CDNs, fonts, analytics | none |
| Total size | ~1.5 MB, mostly the six PNG figures |
| Works offline | yes, including from a USB stick |

The prediction path is 23 numbers and three matrix multiplications running in
the page. That is a deliberate architectural choice: a free-tier backend sleeps
after 15 minutes and takes about 50 seconds to wake, which means a spinner in
front of an audience. With the model in the browser there is nothing to wake.

---

## Regenerating the site's data

`web/data/` and `web/figures/` are build products, committed so the folder is
self-contained. After retraining:

```bash
python -m src.ann.train --model m1       --export models/model_m1.json
python -m src.ann.train --model m1-multi --export models/model_m1_multi.json
python -m src.ann.experiments
python -m src.ann.figures
python -m src.export.site_data
pytest
```

---

## If something is wrong on the live site

**Blank page, or "Data files did not load".** The `data/` folder did not come
with `index.html`. Re-drag the whole `web/` folder.

**Figures missing on the Results tab.** `web/figures/` is empty — run
`python -m src.export.site_data`.

**A prediction looks wrong.** Run `pytest tests/test_js_python_parity.py`. If it
passes, the browser agrees with the trained model to 1e-9 and the model itself
is what you are looking at — median test error is about 8 %, and the page says
so in three places.
