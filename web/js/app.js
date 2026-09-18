/*
 * Wiring for the page. Plain DOM, no framework, no build step.
 *
 * The framing rule this file exists to enforce: a prediction is never shown
 * without its uncertainty, and never without whatever caveat applies to it.
 * The error band is rendered in the same element as the number so there is no
 * layout in which one appears without the other.
 */
(function () {
  "use strict";

  var D = window.EGUN_DATA || {};
  var ann = window.EGUN.ann;
  var physics = window.EGUN.physics;
  var gun = window.EGUN.gun;

  var MODEL = D.model;
  var DATASET = D.dataset;
  var BENCH = D.benchmark || {};
  var LB = D.lbAlpha;

  var $ = function (id) { return document.getElementById(id); };

  function fmt(v, dp) {
    if (v == null || !isFinite(v)) return "—";
    return Number(v).toFixed(dp == null ? 2 : dp);
  }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }

  /* ------------------------------------------------------------------ tabs */

  function initTabs() {
    var tabs = Array.prototype.slice.call(document.querySelectorAll('[role="tab"]'));
    function show(id) {
      tabs.forEach(function (t) {
        var on = t.id === id;
        t.setAttribute("aria-selected", on ? "true" : "false");
        $(t.getAttribute("aria-controls")).hidden = !on;
      });
      // Deep links are a nicety; opening index.html straight off disk is a
      // requirement. replaceState throws a SecurityError on file:// URLs in
      // some browsers, and an exception here would take the whole page down
      // before anything rendered -- so it is allowed to fail quietly.
      try {
        if (history.replaceState) {
          history.replaceState(null, "", "#" + id.replace("tab-", ""));
        }
      } catch (e) { /* file:// — no history to update, and nothing lost */ }
      window.scrollTo(0, 0);
    }
    tabs.forEach(function (t) {
      t.addEventListener("click", function () { show(t.id); });
      t.addEventListener("keydown", function (e) {
        var i = tabs.indexOf(t);
        if (e.key === "ArrowRight") { tabs[(i + 1) % tabs.length].focus(); tabs[(i + 1) % tabs.length].click(); }
        if (e.key === "ArrowLeft") { var j = (i - 1 + tabs.length) % tabs.length; tabs[j].focus(); tabs[j].click(); }
      });
    });
    var want = "tab-" + (location.hash || "").replace("#", "");
    show(tabs.some(function (t) { return t.id === want; }) ? want : "tab-predict");
  }

  /* --------------------------------------------------------------- predict */

  /*
   * Sliders are integer 0..1000 under the hood and mapped onto the physical
   * range, because C spans two decades and needs a logarithmic mapping to be
   * usable at all -- a linear slider would put 5 to 50 in the first sixth of
   * the travel and spend the rest on the two guns above C = 200.
   */
  var SPEC = {
    P:  { key: "perveance_uperv", log: false, dp: 2 },
    rw: { key: "rw_mm",           log: false, dp: 2 },
    C:  { key: "C",               log: true,  dp: 2 },
  };

  var state = {};

  function bounds(name) {
    var env = MODEL.training_envelope[SPEC[name].key];
    // A little travel past the envelope, so the out-of-range warning is
    // reachable rather than theoretical.
    var lo = env[0], hi = env[1];
    if (SPEC[name].log) return { lo: lo / 1.6, hi: hi * 1.6, envLo: lo, envHi: hi };
    var pad = (hi - lo) * 0.18;
    return { lo: Math.max(lo - pad, 0.01), hi: hi + pad, envLo: lo, envHi: hi };
  }

  function toSlider(name, v) {
    var b = bounds(name);
    var f = SPEC[name].log
      ? (Math.log(v) - Math.log(b.lo)) / (Math.log(b.hi) - Math.log(b.lo))
      : (v - b.lo) / (b.hi - b.lo);
    return Math.round(Math.min(1, Math.max(0, f)) * 1000);
  }
  function fromSlider(name, s) {
    var b = bounds(name), f = s / 1000;
    return SPEC[name].log
      ? Math.exp(Math.log(b.lo) + f * (Math.log(b.hi) - Math.log(b.lo)))
      : b.lo + f * (b.hi - b.lo);
  }

  function setValue(name, v, from) {
    var b = bounds(name);
    v = Math.min(b.hi, Math.max(b.lo, Number(v)));
    if (!isFinite(v)) return;
    state[name] = v;
    if (from !== "number") $("in-" + name).value = v.toFixed(SPEC[name].dp);
    if (from !== "slider") $("sl-" + name).value = toSlider(name, v);
    update();
  }

  function initControls() {
    ["P", "rw", "C"].forEach(function (name) {
      var b = bounds(name);
      $("sc-" + name + "-lo").textContent = fmt(b.lo, SPEC[name].dp);
      $("sc-" + name + "-hi").textContent = fmt(b.hi, SPEC[name].dp);
      $("in-" + name).min = b.lo.toFixed(SPEC[name].dp);
      $("in-" + name).max = b.hi.toFixed(SPEC[name].dp);
      $("sl-" + name).addEventListener("input", function (e) {
        setValue(name, fromSlider(name, +e.target.value), "slider");
      });
      $("in-" + name).addEventListener("input", function (e) {
        var v = parseFloat(e.target.value);
        if (isFinite(v)) setValue(name, v, "number");
      });
    });

    /* Presets straight from the dataset, so a viewer can land on a real gun and
       see the published experimental value beside the prediction. */
    var picks = [3, 12, 15, 29].map(function (id) {
      return DATASET.rows.filter(function (r) { return r.source_case_id === id; })[0];
    }).filter(Boolean);
    var row = $("preset-row");
    row.appendChild(Object.assign(document.createElement("span"), {
      className: "muted", style: "font-size:.76rem;align-self:center;margin-right:2px",
      textContent: "Jump to a real gun:",
    }));
    picks.forEach(function (r) {
      var b = document.createElement("button");
      b.className = "ghost";
      b.type = "button";
      b.textContent = "case " + r.source_case_id +
        (r.paper_split === "test" ? " (held out)" : "");
      b.addEventListener("click", function () {
        state.P = r.perveance_uperv; state.rw = r.rw_mm; state.C = r.C;
        ["P", "rw", "C"].forEach(function (n) {
          $("in-" + n).value = state[n].toFixed(SPEC[n].dp);
          $("sl-" + n).value = toSlider(n, state[n]);
        });
        update();
      });
      row.appendChild(b);
    });
  }

  /* Does the current input match a gun in the dataset? Tolerance is tight: the
     published values carry two decimals, so this only fires on a real match. */
  function matchKnown() {
    return DATASET.rows.filter(function (r) {
      return Math.abs(r.perveance_uperv - state.P) < 5e-3 &&
             Math.abs(r.rw_mm - state.rw) < 5e-3 &&
             Math.abs(r.C - state.C) < 5e-2;
    })[0];
  }

  function bandPct() {
    var d = MODEL.metrics.restart_distribution || {};
    var sel = MODEL.metrics.selected || {};
    var med = d.test_mre_pct_median;
    if (med == null && sel.test_mre_pct) med = sel.test_mre_pct[MODEL.targets[0]];
    return med;
  }

  function msg(kind, tag, html) {
    return '<div class="msg ' + kind + '"><span class="tag">' + tag +
           "</span><div>" + html + "</div></div>";
  }

  function update() {
    var x = [state.P, state.rw, state.C];
    var res = ann.predict(MODEL, x);
    var theta = res.values[MODEL.targets[0]];

    /* ---- headline number, always with its band ---- */
    var pct = bandPct();
    var band = (pct != null && isFinite(theta)) ? (theta * pct) / 100 : null;
    $("out-theta").textContent = fmt(theta, 1) + "°";
    $("out-band").textContent = band != null ? "  ± " + fmt(band, 1) + "°" : "";
    $("out-band-note").innerHTML = band == null ? "" :
      "Band is ±" + fmt(pct, 1) + " %, the model's <strong>median</strong> error " +
      "across 30 restarts on the seven held-out guns. It is an aggregate figure " +
      "applied uniformly, not a confidence interval for this particular gun — " +
      "the true error here may be larger.";

    /* ---- caveats ---- */
    var out = [];
    res.outOfEnvelope.forEach(function (o) {
      out.push(msg("warn", "outside training range",
        "<strong>" + esc(o.input) + " = " + fmt(o.value, 2) + "</strong> is " +
        fmt(o.by, 2) + " " + o.side + " the trained range of " +
        fmt(MODEL.training_envelope[o.input][0], 2) + " to " +
        fmt(MODEL.training_envelope[o.input][1], 2) +
        ". The network has seen no gun like this and the number above is an " +
        "extrapolation from a model that structurally cannot extrapolate."));
    });
    res.saturated.forEach(function (s) {
      out.push(msg("bad", "saturated",
        "The normalised output for <strong>" + esc(s.target) + "</strong> has run " +
        "to " + fmt(s.normalised, 3) + ", past the ±" +
        (MODEL.saturation_limit || 0.98) + " limit. A tanh output over min-max " +
        "scaled targets asymptotes at ±1, which maps back to the edge of the " +
        "training range — so this is the network against its ceiling, not a " +
        "confident prediction. The paper's own worst test case is this failure."));
    });

    var known = matchKnown();
    if (known) {
      var err = theta - known.theta_cone_deg;
      out.push(msg("info", "known gun",
        "These inputs are <strong>case " + known.source_case_id + "</strong> from " +
        "Table 2, a " + (known.paper_split === "test" ? "<strong>held-out test</strong>"
        : "training") + " gun. Measured θ = <strong>" +
        fmt(known.theta_cone_deg, 2) + "°</strong>; this model is off by " +
        (err >= 0 ? "+" : "") + fmt(err, 2) + "°." +
        (known.paper_split === "train"
          ? " It was trained on this row, so the agreement is not evidence of " +
            "generalisation."
          : " It was never trained on this row.")));
    }
    $("messages").innerHTML = out.join("");

    /* ---- geometry + drawing ---- */
    var geo = physics.geometry({
      thetaDeg: theta, P: state.P, rw: state.rw, C: state.C, lbTable: LB,
    });
    gun.render($("gun-svg"), geo);
    $("gun-caption").innerHTML =
      "Meridional slice, drawn to scale, mirrored about the axis. " +
      "The cathode cap and r_c, R_c follow exactly from θ by eqs (2) and (4). " +
      "The anode is placed by the Langmuir–Blodgett relation, which is the part " +
      "of our physics engine that passed its check (~1e-9). " +
      "<strong>The anode-to-waist drift length is indicative only</strong> — it " +
      "uses the same space-charge closure that failed validation at MAE 1.66°. " +
      "A schematic, not a drawing to build from.";

    var rows = [
      ["Cathode disc radius r_c", fmt(geo.rc, 3) + " mm", "√C · r_w, eq (2)"],
      ["Cathode radius of curvature R_c", fmt(geo.Rc, 3) + " mm", "r_c / sin θ, eq (4)"],
      ["γ = ln(R_c/R_a)", fmt(geo.gamma, 4), "from Langmuir–Blodgett"],
      ["Anode aperture radius r_a", fmt(geo.ra, 3) + " mm", "R_a sin θ"],
      ["Cathode→anode along axis", fmt(geo.zAnode, 3) + " mm", "R_c − R_a"],
      ["Anode→waist drift", fmt(geo.drift ? geo.drift.length : NaN, 3) + " mm",
       "indicative — failed gate"],
    ];
    $("geom-rows").innerHTML = rows.map(function (r) {
      return '<div class="row"><span class="k">' + esc(r[0]) +
             ' <small class="muted">' + esc(r[2]) + '</small></span>' +
             '<span class="v">' + esc(r[1]) + "</span></div>";
    }).join("");

    /* ---- comparison with the classical methods ---- */
    var cmp = $("compare-body");
    if (known) {
      var items = [
        ["Measured (experiment)", known.theta_cone_deg, true],
        ["This model (M1)", theta, false],
        ["ANN — Panahi et al.", known.theta_ann_panahi_deg, false],
        ["Iterative", known.theta_iterative_deg, false],
        ["Non-iterative", known.theta_noniterative_deg, false],
        ["Modified non-iterative", known.theta_modified_noniter_deg, false],
      ];
      cmp.innerHTML = '<div class="rows">' + items.map(function (it) {
        return '<div class="row"><span class="k">' + esc(it[0]) + "</span>" +
               '<span class="v' + (it[2] ? " em" : "") + '">' +
               fmt(it[1], 2) + "°</span></div>";
      }).join("") + "</div>";
    } else {
      cmp.innerHTML = '<p class="note" style="margin:0">The three classical methods' +
        " are published as values for the 30 guns in Table 2, not as formulas we " +
        "hold — the iterative and non-iterative procedures are in papers that are " +
        "paywalled and were never retrieved, and our own implementation of the " +
        "closure failed its validation gate. So they can be shown for a real gun " +
        "but not computed for arbitrary inputs. Pick one of the cases above to " +
        "see all four side by side.</p>";
    }
  }

  /* --------------------------------------------------------------- dataset */

  // With 1030 rows the table is capped so the page stays responsive; the cap is
  // stated rather than silently applied, and filtering narrows below it.
  var DS_RENDER_LIMIT = 300;

  var DS_COLS = [
    { k: "tier", t: "Tier", tier: true },
    { k: "theta_source", t: "θ from" },
    { k: "source_case_id", t: "Case", num: true },
    { k: "split", t: "Split", pill: true },
    { k: "perveance_uperv", t: "P (µperv)", num: true, dp: 2 },
    { k: "rw_mm", t: "r_w (mm)", num: true, dp: 2 },
    { k: "C", t: "C", num: true, dp: 2 },
    { k: "theta_cone_deg", t: "θ measured (°)", num: true, dp: 2 },
    { k: "theta_ann_panahi_deg", t: "θ ANN paper (°)", num: true, dp: 2 },
    { k: "theta_iterative_deg", t: "θ iterative (°)", num: true, dp: 2 },
    { k: "theta_noniterative_deg", t: "θ non-iter (°)", num: true, dp: 2 },
    { k: "theta_modified_noniter_deg", t: "θ mod non-iter (°)", num: true, dp: 2 },
    { k: "rc_mm", t: "r_c (mm)", num: true, dp: 3 },
    { k: "Rc_mm", t: "R_c (mm)", num: true, dp: 3 },
    { k: "beam_type", t: "Beam" },
    { k: "generator_mre_pct", t: "Generator MRE %", num: true, dp: 2 },
    { k: "generator_envelope", t: "Generator envelope" },
    { k: "source_table", t: "Table" },
    { k: "origin_refs", t: "Upstream refs" },
    { k: "source_doi", t: "DOI", doi: true },
  ];

  var dsSort = { key: "source_case_id", dir: 1 };

  function dsRender() {
    var q = ($("ds-search").value || "").toLowerCase().trim();
    var split = $("ds-split").value;
    var tier = $("ds-tier").value;
    var rows = DATASET.rows.filter(function (r) {
      if (tier && r.tier !== tier) return false;
      if (split && r.split !== split) return false;
      if (!q) return true;
      return DS_COLS.some(function (c) {
        return String(r[c.k]).toLowerCase().indexOf(q) >= 0;
      }) || String(r.source_citation).toLowerCase().indexOf(q) >= 0
         || String(r.origin_citations).toLowerCase().indexOf(q) >= 0;
    });

    rows.sort(function (a, b) {
      var x = a[dsSort.key], y = b[dsSort.key];
      if (typeof x === "number" && typeof y === "number") return (x - y) * dsSort.dir;
      return String(x).localeCompare(String(y)) * dsSort.dir;
    });

    $("ds-head").innerHTML = DS_COLS.map(function (c) {
      var arrow = dsSort.key === c.k
        ? '<span class="arrow">' + (dsSort.dir > 0 ? "▲" : "▼") + "</span>" : "";
      return '<th data-k="' + c.k + '" scope="col">' + esc(c.t) + arrow + "</th>";
    }).join("");

    var shown = rows.slice(0, DS_RENDER_LIMIT);
    $("ds-body").innerHTML = shown.map(function (r) {
      return "<tr>" + DS_COLS.map(function (c) {
        var v = r[c.k];
        if (v === null || v === undefined || v === "") return '<td class="muted">—</td>';
        if (c.tier) {
          var real = v === "A_literature";
          return '<td><span class="pill ' + (real ? "real" : "synth") + '">' +
                 (real ? "real" : "generated") + "</span></td>";
        }
        if (c.doi) {
          return '<td><a href="https://doi.org/' + esc(v) + '" target="_blank" ' +
                 'rel="noopener">' + esc(v) + "</a></td>";
        }
        if (c.pill) {
          return '<td><span class="pill ' + esc(v) + '">' + esc(v) + "</span></td>";
        }
        if (c.num) {
          return '<td class="num">' +
                 (c.dp == null ? esc(v) : fmt(v, c.dp)) + "</td>";
        }
        return "<td>" + esc(v) + "</td>";
      }).join("") + "</tr>";
    }).join("");

    var nReal = rows.filter(function (r) { return r.tier === "A_literature"; }).length;
    $("ds-count").innerHTML =
      rows.length + " of " + DATASET.rows.length + " rows — <strong>" + nReal +
      " real</strong>, " + (rows.length - nReal) + " generated" +
      (rows.length > DS_RENDER_LIMIT
        ? ' <span class="muted">(showing the first ' + DS_RENDER_LIMIT +
          "; filter to narrow)</span>"
        : "");

    Array.prototype.forEach.call($("ds-head").children, function (th) {
      th.addEventListener("click", function () {
        var k = th.getAttribute("data-k");
        dsSort = { key: k, dir: dsSort.key === k ? -dsSort.dir : 1 };
        dsRender();
      });
    });
  }

  function initDataset() {
    var p = DATASET.provenance;
    var g = p.generator || {};
    var rv = p.real_vs_generated || {};

    $("ds-lede").innerHTML =
      "This dataset mixes two very different kinds of row, and the difference " +
      "matters more than the total. <strong>" + rv.real + " rows describe guns " +
      "that were actually built and measured.</strong> The other " +
      rv.generated + " are output from a physics model, computed at design " +
      "points nobody has published a gun for.";

    // The one statement a visitor must not be able to miss.
    $("ds-headline").innerHTML =
      '<span class="tag">read this</span><div><strong>' + esc(p.headline) +
      "</strong><br>" + esc(rv.plain_language) + "</div>";

    $("ds-stats").innerHTML = [
      [p.tier_a_rows, "real guns (Tier A)", true],
      [p.tier_b_rows, "physics-generated (Tier B)", false],
      [g.mre_pct_vs_experiment != null ? g.mre_pct_vs_experiment + " %" : "—",
       "generator error vs measurement", false],
      [g.envelope || "—", "generator envelope", false],
      [p.split.test, "test rows — all real", true],
      [p.tier_c_rows, "simulation rows (Tier C)", false],
    ].map(function (s) {
      return '<div class="stat"><div class="n"' +
             (s[2] ? ' style="color:var(--accent)"' : "") + ">" + esc(s[0]) +
             '</div><div class="l">' + esc(s[1]) + "</div></div>";
    }).join("");

    $("ds-provenance").innerHTML =
      "<h3 style='margin-top:0'>Tier A — the " + p.tier_a_rows + " real guns</h3>" +
      '<p class="note">All ' + p.tier_a_rows + " come from Table 2 of Panahi et " +
      "al. (2025), extracted programmatically from the PDF rather than typed — " +
      "transcribing a 30×9 numeric table by hand is exactly where silent " +
      "corruption enters. The seven asterisked test cases independently " +
      "reproduce the paper's own stated 23 / 7 split, which is the check that " +
      "the extraction was read correctly.</p>" +
      '<p class="note">The measurements originate further upstream, and each row ' +
      "records which: cases 1, 2, 9, 12, 25 and 29 from Frost, Purl &amp; Johnson " +
      "(1962), Tiwary &amp; Basu (1987) and Yang, Jia &amp; Zhu (2006); the rest " +
      "from Yang et al. Those three papers are paywalled and were <em>not</em> " +
      "independently retrieved, so the rows are counted once, against Panahi et " +
      "al., rather than being triple-counted as four sources.</p>" +
      msg("warn", "correction", "<strong>We found an error in our own working " +
        "sheet.</strong> " + esc(p.correction_note)) +

      "<h3>Tier B — the " + p.tier_b_rows + " generated rows</h3>" +
      '<p class="note"><strong>These are not measurements.</strong> They are the ' +
      "classical Pierce synthesis equations, solved at " + p.tier_b_rows +
      " new design points. Evaluating a validated model at new points is a " +
      "standard way to make training data — but no gun described by these rows " +
      "has ever been built, and nothing here should be read as though one had " +
      "been.</p>" +
      '<p class="note">' + esc(g.plain_language || "") + "</p>" +
      msg("bad", "what that 2.80 % does not mean",
        "The 2.80 % above is how closely the model reproduces the classical " +
        "<em>iterative method</em> — not measurement. That method is itself about " +
        "<strong>9.7 % away from the measured angles</strong>, so against what the " +
        "network actually has to predict the generator carries about " +
        "<strong>10.5 %</strong>. A model trained on these rows (M3) scored " +
        "<strong>9.77 %</strong> on real held-out guns, against <strong>7.93 %</strong> " +
        "for the same model trained on real data alone. Synthetic data made it " +
        "worse. See the Results tab.") +
      '<p class="note">Rules these rows obey: every one is <code>split = train</code>, ' +
      "so no synthetic row is ever scored against; none sits at the coordinates of " +
      "a held-out test gun; and each carries its generator's error and envelope in " +
      "its own columns, so you can read them off any row in the table below.</p>" +
      '<p class="note">Seven real guns have C &lt; 8 and fall outside the ' +
      "generator's validated envelope. <strong>They have no synthetic support at " +
      "all</strong> — that corner of the design space is simply missing, and the " +
      "coverage figure on the Results tab plots it as excluded rather than " +
      "cropping it out.</p>" +
      '<p class="note">' + esc(p.derived_note) + "</p>";

    $("ds-table-heading").textContent =
      "All " + DATASET.n_rows + " rows — " + p.tier_a_rows + " real, " +
      p.tier_b_rows + " generated";

    $("ds-footnote").innerHTML =
      "The <strong>Tier</strong> column says which kind of row you are looking " +
      "at, and <strong>θ from</strong> says where its angle came from — a " +
      "measurement or the generator. Click a column heading to sort. Every " +
      "column is read from the paper except r_c, R_c and the generated rows " +
      "themselves, all flagged in <code>derived_fields</code>. The underlying " +
      "file is <code>data/processed/dataset_master.csv</code>, mirrored here as " +
      "<code>data/dataset.json</code>.";

    $("ds-search").addEventListener("input", dsRender);
    $("ds-split").addEventListener("change", dsRender);
    $("ds-tier").addEventListener("change", dsRender);
    dsRender();
  }

  /* --------------------------------------------------------------- results */

  function initResults() {
    var dists = BENCH.distributions || {};
    var m1 = dists["M1 (theta only)"] || {};
    var paper = BENCH.paper || {};

    $("headline-card").innerHTML =
      "<h3 style='margin-top:0'>Headline</h3>" +
      '<p class="note" style="margin-top:0">Our median test error on the seven ' +
      "held-out guns is <strong>" + fmt(m1.test_mre_median, 2) + " %</strong> " +
      "(interquartile " + fmt(m1.test_mre_q1, 2) + "–" + fmt(m1.test_mre_q3, 2) +
      " %, full range " + fmt(m1.test_mre_min, 2) + "–" + fmt(m1.test_mre_max, 2) +
      " %), against the paper's reported <strong>" + fmt(paper.test_mre_pct, 2) +
      " %</strong>. <strong>We do not match it.</strong> " +
      (m1.frac_below_paper_3p74 != null
        ? Math.round(m1.frac_below_paper_3p74 * (m1.n_seeds || 30)) + " of " +
          (m1.n_seeds || 30) + " restarts landed below the paper's figure."
        : "") + "</p>";

    $("dist-body").innerHTML =
      '<p class="note">A single test-set number on seven points does not identify ' +
      "a model trained on 23 rows. Running the same code, the same data and the " +
      "same protocol 30 times over — changing nothing but the random " +
      "initialisation — gives errors spanning " + fmt(m1.test_mre_min, 2) + " % to " +
      fmt(m1.test_mre_max, 2) + " %. So everything on this page is reported as a " +
      "median and a range.</p>" +
      '<div class="table-scroll"><table class="plain"><thead><tr>' +
      "<th>Model</th><th>Median</th><th>IQR</th><th>Full range</th>" +
      "<th>Beat 3.74 %</th></tr></thead><tbody>" +
      Object.keys(dists).map(function (k) {
        var d = dists[k];
        return "<tr><td>" + esc(k) + '</td><td class="num">' +
          fmt(d.test_mre_median, 2) + ' %</td><td class="num">' +
          fmt(d.test_mre_q1, 2) + "–" + fmt(d.test_mre_q3, 2) +
          ' %</td><td class="num">' + fmt(d.test_mre_min, 2) + "–" +
          fmt(d.test_mre_max, 2) + ' %</td><td class="num">' +
          Math.round((d.frac_below_paper_3p74 || 0) * d.n_seeds) + " of " +
          d.n_seeds + "</td></tr>";
      }).join("") + "</tbody></table></div>";

    $("repro-body").innerHTML =
      '<p class="note"><strong>What we observe.</strong> Across 30 restarts of ' +
      "identical code, data and protocol, test error spans " +
      fmt(m1.test_mre_min, 2) + " % to " + fmt(m1.test_mre_max, 2) + " %. The " +
      "cross-validation score computed inside the 23 training rows ranks those " +
      "restarts only weakly (Spearman ≈ +0.50), so there is no reliable way to " +
      "pick a good initialisation in advance without looking at the test set — " +
      "which would destroy the thing being measured.</p>" +
      '<p class="note"><strong>What we conclude.</strong> The published 3.74 % ' +
      "is not robustly reproducible from the method as stated. A reader following " +
      "the paper's description should expect a result somewhere in a wide " +
      "distribution, with 3.74 % in its best few per cent rather than at its " +
      "centre.</p>" +
      '<p class="note"><strong>What we do not conclude.</strong> Nothing about ' +
      "how the paper's number was obtained. It reports a single value without a " +
      "seed, an initialisation scheme, a restart count, a selection rule or a " +
      "variance, so we cannot tell from outside whether it is a typical run of a " +
      "procedure differing from ours in some detail we cannot see, or one draw of " +
      "a lottery like the one we observe. Both are consistent with what is " +
      "published, and the gap may simply be a method detail the paper does not " +
      "record.</p>" +
      '<p class="note">The substantive point is not about this paper. On 23 ' +
      "training rows, a single test number on seven points does not identify a " +
      "model, whoever reports it. A median and a range cost nothing and would " +
      "have made the comparison decidable.</p>";

    /* benchmark table */
    var rowsFor = function (id) {
      var b = BENCH[id];
      if (!b) return "";
      var s = b.metrics.selected, t = b.targets[0];
      return "<tr><td>" + esc(id) + " <small class='muted'>(" +
        esc(b.targets.join(" + ")) + ", " + b.n_params + " params)</small></td>" +
        '<td class="num">' + fmt(s.train_mre_pct[t], 2) + ' %</td><td class="num">' +
        fmt(s.test_mre_pct[t], 2) + ' %</td><td class="num">' +
        fmt(s.test_rmse[t], 2) + '</td><td class="num">' +
        fmt(s.test_pearson_r[t], 4) + "</td></tr>";
    };
    $("bench-table").innerHTML =
      "<thead><tr><th>Model</th><th>Train MRE</th><th>Test MRE</th>" +
      "<th>Test RMSE</th><th>Test R</th></tr></thead><tbody>" +
      "<tr><td><strong>Panahi et al.</strong> <small class='muted'>as reported" +
      "</small></td>" +
      '<td class="num">' + fmt(paper.train_mre_pct, 2) + ' %</td><td class="num">' +
      fmt(paper.test_mre_pct, 2) + ' %</td><td class="num">' +
      fmt(paper.test_rmse_deg, 2) + '</td><td class="num">' +
      fmt(paper.test_pearson_r, 3) + "</td></tr>" +
      rowsFor("M1") + rowsFor("M1-multi") + "</tbody>";

    /* M3: did synthetic data help? */
    var m3 = BENCH["M3"], m1 = BENCH["M1"];
    if (m3 && m1) {
      var s1 = m1.metrics.selected, s3 = m3.metrics.selected;
      var t1 = m1.targets[0], t3 = m3.targets[0];
      var r1 = m1.restart_spread || {}, r3 = m3.restart_spread || {};
      $("m3-body").innerHTML =
        '<p class="note"><strong>No — it made the model worse, and the reason is ' +
        "worth more than the result.</strong> M3 was trained on the 23 real guns " +
        "plus " + (m3.n_synthetic_train || 1000) + " physics-generated rows, and " +
        "tested on the same seven real held-out guns. It was never tested on " +
        "synthetic rows.</p>" +
        '<div class="table-scroll"><table class="plain"><thead><tr>' +
        "<th>Model</th><th>Trained on</th><th>Train MRE</th>" +
        "<th>Test MRE (real guns)</th><th>Restart spread</th></tr></thead><tbody>" +
        "<tr><td><strong>M1</strong></td><td>23 real guns</td>" +
        '<td class="num">' + fmt(s1.train_mre_pct[t1], 2) + ' %</td>' +
        '<td class="num"><strong>' + fmt(s1.test_mre_pct[t1], 2) + " %</strong></td>" +
        '<td class="num">' + fmt(r1.test_mre_min, 2) + "–" + fmt(r1.test_mre_max, 2) +
        " %</td></tr>" +
        "<tr><td><strong>M3</strong></td><td>23 real + " +
        (m3.n_synthetic_train || 1000) + " synthetic</td>" +
        '<td class="num">' + fmt(s3.train_mre_pct[t3], 2) + ' %</td>' +
        '<td class="num"><strong>' + fmt(s3.test_mre_pct[t3], 2) + " %</strong></td>" +
        '<td class="num">' + fmt(r3.test_mre_min, 2) + "–" + fmt(r3.test_mre_max, 2) +
        " %</td></tr></tbody></table></div>" +
        '<p class="note">The extra data did what more data should do: M3 fits its ' +
        "training set almost perfectly (0.91 %) and its run-to-run spread " +
        "collapses. It converged confidently on the wrong answer.</p>" +
        '<p class="note"><strong>The generator was validated against the wrong ' +
        "target.</strong> Our physics reproduces the classical <em>iterative " +
        "method</em> to 2.80 % over its envelope — but that method is itself " +
        "9.71 % away from the measured angles. So against the quantity the " +
        "network has to predict, the generator carries <strong>10.48 %</strong>, " +
        "not 2.80 %. M3 inherited that bias almost exactly: its mean signed error " +
        "on the test guns is −2.92°, against −0.71° for M1, and the iterative " +
        "method's own bias on those guns is −2.58°.</p>" +
        '<p class="note">The synthetic rows are kept and remain useful as a fast ' +
        "stand-in <em>for the iterative method</em> — which is what they actually " +
        "are, and what a multi-objective optimizer needs when it must evaluate " +
        "the classical synthesis tens of thousands of times. They are not a route " +
        "to better prediction of measured angles. <strong>M1, trained on real " +
        "guns only, remains the model this site reports.</strong></p>";
    }

    /* per-case */
    var b = BENCH.M1;
    if (b) {
      $("percase-table").innerHTML =
        "<thead><tr><th>Case</th><th>θ measured</th><th>θ predicted</th>" +
        "<th>Error</th></tr></thead><tbody>" +
        b.test_case_ids.map(function (c, i) {
          var y = b.truth_test[i][0], p = b.predictions_test[i][0];
          return "<tr><td>" + c + '</td><td class="num">' + fmt(y, 2) +
            '°</td><td class="num">' + fmt(p, 2) + '°</td><td class="num">' +
            (p - y >= 0 ? "+" : "") + fmt(p - y, 2) + "°</td></tr>";
        }).join("") + "</tbody>";
      $("percase-note").innerHTML =
        "The shipped model, on the seven guns it never saw. Case 3 is the worst, " +
        "and it is the same gun the paper's own network under-predicts most " +
        "(45.03° against 50.28°) — it has the highest angle in the test set, and " +
        "a tanh output compresses hardest near the edge of its training range. " +
        "Case 12 is worth noting for the opposite reason: before C was " +
        "log-scaled it was wrong by +30.2°, and it is now wrong by +2.2°.";
    }
  }

  /* ---------------------------------------------------------------- method */

  function initMethod() {
    /* A small architecture diagram: every arrow is one weight. */
    var svg = $("arch-svg");
    var NS = "http://www.w3.org/2000/svg";
    var W = 680, H = 250;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label",
      "Network diagram: three inputs, a layer of three neurons, a layer of two " +
      "neurons, and one output, fully connected.");

    var layers = MODEL.architecture;
    var names = [["P", "r_w", "ln C"], null, null,
                 MODEL.targets.map(function (t) { return t === "theta_cone_deg" ? "θ" : t; })];
    var titles = ["input", "hidden 1 (tanh)", "hidden 2 (tanh)", "output (tanh)"];
    var xs = layers.map(function (_, i) {
      return 90 + (i * (W - 200)) / (layers.length - 1);
    });
    var pos = layers.map(function (n, i) {
      return Array.from({ length: n }, function (_, j) {
        return { x: xs[i], y: H / 2 + (j - (n - 1) / 2) * 52 };
      });
    });

    function mk(name, attrs, text) {
      var e = document.createElementNS(NS, name);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      if (text != null) e.textContent = text;
      return e;
    }

    for (var i = 0; i < pos.length - 1; i++) {
      pos[i].forEach(function (a) {
        pos[i + 1].forEach(function (b2) {
          svg.appendChild(mk("line", {
            x1: a.x, y1: a.y, x2: b2.x, y2: b2.y,
            stroke: "#2c313d", "stroke-width": 1,
          }));
        });
      });
    }
    pos.forEach(function (layer, i) {
      svg.appendChild(mk("text", {
        x: xs[i], y: 22, fill: "#6f7889", "font-size": 10,
        "text-anchor": "middle", "font-family": "system-ui, sans-serif",
      }, titles[i]));
      layer.forEach(function (p, j) {
        svg.appendChild(mk("circle", {
          cx: p.x, cy: p.y, r: 15,
          fill: i === 0 ? "#20242e" : "#1e3a5f",
          stroke: i === 0 ? "#2c313d" : "#5aa2ff", "stroke-width": 1.5,
        }));
        if (names[i]) {
          svg.appendChild(mk("text", {
            x: p.x - (i === 0 ? 26 : -26), y: p.y + 4,
            fill: "#a7b0c0", "font-size": 11,
            "text-anchor": i === 0 ? "end" : "start",
            "font-family": "ui-monospace, Menlo, Consolas, monospace",
          }, names[i][j]));
        }
      });
    });

    var nParams = 0;
    for (var L = 0; L < layers.length - 1; L++) {
      nParams += layers[L] * layers[L + 1] + layers[L + 1];
    }
    svg.appendChild(mk("text", {
      x: W / 2, y: H - 10, fill: "#6f7889", "font-size": 11,
      "text-anchor": "middle", "font-family": "system-ui, sans-serif",
    }, layers.join(" → ") + "  =  " + nParams + " parameters, trained on 23 guns"));

    $("method-arch").innerHTML =
      '<p class="note">' + layers.join(" → ") + ", tanh throughout, <strong>" +
      nParams + " parameters</strong>: " +
      layers.slice(0, -1).map(function (n, i) {
        return "(" + n + "×" + layers[i + 1] + " + " + layers[i + 1] + ")";
      }).join(" + ") + ". Trained on <strong>23</strong> guns — about as many " +
      "free parameters as data points, which is the central limitation of this " +
      "whole exercise and is discussed on the About tab.</p>";
  }

  /* ------------------------------------------------------------------ boot */

  function boot() {
    if (!MODEL || !DATASET) {
      document.querySelector("main").innerHTML =
        '<div class="msg bad"><span class="tag">error</span><div>Data files did ' +
        "not load. If you opened this from a zip, make sure the <code>data/</code> " +
        "folder sits next to <code>index.html</code>.</div></div>";
      return;
    }

    initTabs();
    initControls();

    var first = DATASET.rows.filter(function (r) { return r.source_case_id === 15; })[0]
                || DATASET.rows[0];
    state = { P: first.perveance_uperv, rw: first.rw_mm, C: first.C };
    ["P", "rw", "C"].forEach(function (n) {
      $("in-" + n).value = state[n].toFixed(SPEC[n].dp);
      $("sl-" + n).value = toSlider(n, state[n]);
    });

    initDataset();
    initResults();
    initMethod();
    update();

    var prov = MODEL.provenance || {};
    $("build-info").textContent =
      "Model " + (prov.model_id || "M1") + ", trained " +
      String(prov.trained_at || "").slice(0, 10) + ", seed " + prov.seed +
      " of " + prov.restarts + " restarts, " + prov.n_train + " training and " +
      prov.n_test + " test guns, " + prov.tier_b_rows + " synthetic rows.";
    $("footer-text").textContent =
      "Static page — no backend, no network calls, no analytics. " +
      "Prediction runs entirely in your browser.";
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
