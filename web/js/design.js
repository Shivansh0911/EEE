/*
 * The Design tab: Vedant's four inputs in, the full gun out.
 *
 * Synthesis runs here, in the page, through closure.js (the G1-verified
 * emulator of his exact solver). Only the Pareto search needs the API; when the
 * API is asleep or unreachable the precomputed fronts in data/pareto.js are
 * shown instead, and the page says which it is showing.
 *
 * Geometry is drawn from theta_raw (Rc, Ra, z_ac, throw); the cathode is drawn
 * as built, with the aberration-corrected theta_T and RcT -- as in Vedant's
 * compute_gun_points_mm, whose points the electrode outline uses directly.
 */
(function () {
  "use strict";

  var D = window.EGUN_DATA || {};
  var CL = window.EGUN.closure;
  var $ = function (id) { return document.getElementById(id); };
  var SVGNS = "http://www.w3.org/2000/svg";
  var current = null;

  function fmt(v, n) { return isFinite(v) ? v.toFixed(n) : "—"; }
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function msg(kind, html) { return '<div class="msg ' + kind + '">' + html + "</div>"; }

  // ---- the three numbers, equal prominence ----------------------------------
  function initStats() {
    var m = D.designMeta;
    var e = m.emulator_vs_exact, f = m.vaughan_floor;
    $("ds-stats").innerHTML =
      stat("max " + (e.theta_max_rel * 100).toExponential(1) + " %",
           "Emulator vs Vedant's exact solver (θ, " + e.n + " random designs, G1)") +
      stat("±" + m.calibration_band_pct.toFixed(1) + " %",
           "Calibrated θ vs a measured gun (90th-percentile leave-one-out error, 23 real guns)") +
      stat(f.exact_solver_mre_pct_30.toFixed(1) + " %",
           "Vaughan's floor: exact solver vs 30 measured guns (MRE; paper's own column " +
           f.paper_iterative_column_mre_pct_30.toFixed(1) + " %)");
  }
  function stat(n, l) {
    return '<div class="stat"><div class="n">' + n + '</div><div class="l">' + l + "</div></div>";
  }

  // ---- inputs ---------------------------------------------------------------
  var FIELDS = [["V", "ds-V", 1000], ["I", "ds-I", 1], ["rw_mm", "ds-rw", 1], ["Jc", "ds-Jc", 1]];
  function readInputs() {
    var x = {};
    FIELDS.forEach(function (f) { x[f[0]] = parseFloat($(f[1]).value) * f[2]; });
    return x;
  }
  function setInputs(d) {
    FIELDS.forEach(function (f) { $(f[1]).value = +(d[f[0]] / f[2]).toPrecision(6); });
    update();
  }

  // ---- one design -----------------------------------------------------------
  function update() {
    var x = readInputs(), out = $("ds-messages");
    if (![x.V, x.I, x.rw_mm, x.Jc].every(function (v) { return isFinite(v) && v > 0; })) {
      out.innerHTML = msg("bad", "All four inputs must be positive numbers.");
      return;
    }
    var r;
    try { r = CL.synthesize(x.V, x.I, x.rw_mm, x.Jc, D.kfTable); }
    catch (e) { out.innerHTML = msg("bad", esc(e.message)); return; }
    current = r;
    var m = [];
    if (!r.feasible) {
      $("ds-theta").textContent = "—";
      $("ds-theta-raw").textContent = "";
      $("ds-measured").textContent = "";
      $("ds-rows").innerHTML = "";
      clearSvg();
      setDeck({ refused: "design is infeasible: " + r.reason });
      out.innerHTML = msg("bad", "<strong>No Pierce gun exists for these inputs.</strong> " +
        esc(r.reason) + ". (μP = " + fmt(r.muP, 3) + ", C = " + fmt(r.C, 2) + ")");
      return;
    }
    $("ds-theta").textContent = fmt(r.thetaT, 2) + "°";
    $("ds-theta-raw").textContent = "θ_raw = " + fmt(r.theta, 3) + "° (analysis angle; all lengths below use it)";

    var cal = D.designMeta, t = CL.calibrate(cal.calibrator, r.thetaT, r.muP, r.rw_mm, r.C);
    var b = cal.calibration_band_pct / 100;
    var env = cal.calibration_envelope;
    var outside = (r.muP < env.perveance_uperv[0] || r.muP > env.perveance_uperv[1] ||
                   r.C < env.C[0] || r.C > env.C[1]);
    $("ds-measured").textContent = "If built and measured: " + fmt(t, 1) + "° (" +
      fmt(t * (1 - b), 1) + "–" + fmt(t * (1 + b), 1) + "°)" + (outside ? " — extrapolated" : "");

    if (!r.za_valid) m.push(msg("warn", "<strong>θ_raw > 56.44°.</strong> Vaughan's Eq.(20) has " +
      "no real root, so the anode plane z_a is undefined and no EGUN deck is written. " +
      "The Sharma (2001) aperture method that would fix this is not implemented."));
    if (!r.alpha_ok) m.push(msg("warn", "−α ≥ 7.9: Vedant's Eq.(5) inversion is past its " +
      "convergence limit here; treat the angle as unreliable."));
    if (r.thetaT > 66.5) m.push(msg("warn", "θ_T above 66.5°: beyond the optimiser's limit."));
    if (outside) m.push(msg("info", "μP or C is outside the 23 measured training guns' range " +
      "(μP " + env.perveance_uperv.join("–") + ", C " + env.C.join("–") +
      "): the measured-θ estimate is an extrapolation."));
    out.innerHTML = m.join("");

    var rows = [
      ["Microperveance μP", fmt(r.muP, 4)], ["Convergence C = (r_c/r_w)²", fmt(r.C, 3)],
      ["Relativistic diode factor k_f", fmt(r.rel_factor, 5)],
      ["Cathode disc radius r_c", fmt(r.rc_mm, 4) + " mm"],
      ["Cathode sphere R_c (from θ_raw)", fmt(r.Rc_mm, 4) + " mm"],
      ["Cathode as built R_cT = 0.905 R_c", fmt(r.RcT_mm, 4) + " mm"],
      ["Anode sphere R_a", fmt(r.Ra_mm, 4) + " mm"],
      ["Cathode–anode spacing z_ac", fmt(r.zac_mm, 4) + " mm"],
      ["Anode plane z_a", r.za_valid ? fmt(r.za_mm, 4) + " mm" : "undefined (θ_raw > 56.44°)"],
      ["Anode aperture r_a = 1.2 r_b(z_a)", fmt(r.ra_mm, 4) + " mm"],
      ["Throw, cathode vertex to waist z_w", fmt(r.zw_mm, 4) + " mm"],
      ["Compression R", fmt(r.R, 4)],
      ["Mean gap field V / z_ac", fmt(r.E_kV_per_mm, 2) + " kV/mm"],
    ];
    $("ds-rows").innerHTML = rows.map(function (p) {
      return '<div class="row"><span class="k">' + p[0] + '</span><span class="v">' + p[1] + "</span></div>";
    }).join("");

    var dk = CL.deck(r);
    setDeck(dk);
    draw(r, dk);
  }

  function setDeck(dk) {
    var btn = $("ds-deck"), note = $("ds-deck-note");
    btn.disabled = !dk.text;
    note.textContent = dk.text ? "Vedant's EGUN point deck (his construction, default " +
      "parameters: k1 0.02, d 0.2 mm, H 0.05 mm). Re-check with his exact solver: " +
      "python -m src.moo.deck --V … --I … --rw … --Jc …" : "No deck: " + dk.refused;
    btn.onclick = function () {
      var blob = new Blob([dk.text], { type: "text/plain" });
      var a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "pierce_gun_points.txt";
      a.click();
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    };
  }

  // ---- drawing --------------------------------------------------------------
  function clearSvg() { var s = $("ds-svg"); while (s.firstChild) s.removeChild(s.firstChild); }
  function el(name, attrs, parent) {
    var e = document.createElementNS(SVGNS, name);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    parent.appendChild(e);
    return e;
  }

  function draw(r, dk) {
    clearSvg();
    var svg = $("ds-svg"), W = 640, H = 300, pad = 24;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var d = 0.2, Dz = dk.params ? dk.params.Dz : 0;
    var zMax = dk.points ? dk.points[14][2] : d + 1.3 * r.zw_mm;
    var rMax = dk.points ? dk.points[7][3] * 1.08 : 1.8 * r.rc_mm;
    var s = Math.min((W - 2 * pad) / zMax, (H - 2 * pad) / rMax);
    var X = function (z) { return pad + z * s; }, Y = function (rr) { return H - pad - rr * s; };
    el("line", { x1: X(0), y1: Y(0), x2: X(zMax), y2: Y(0), class: "ds-axis" }, svg);

    // beam edge: from the cathode rim, converging on the cathode's centre of
    // curvature (theta_raw cone) to the anode, then out to the waist
    var thR = r.theta * Math.PI / 180, zv = d + Dz, beam = [];
    var zRim = d + (dk.points ? dk.points[2][2] - d : r.RcT_mm * (1 - Math.cos(r.thetaT * Math.PI / 180)));
    beam.push([zRim, r.rc_mm]);
    var zA = zv + (r.za_valid ? r.za_mm : r.zac_mm);
    beam.push([zA, r.rb_za_mm]);
    beam.push([d + r.zw_mm, r.rw_mm]);
    el("polyline", { points: beam.map(function (p) { return X(p[0]) + "," + Y(p[1]); }).join(" "),
                     class: "ds-beam" }, svg);

    if (dk.points) {
      var P = {}; dk.points.forEach(function (p) { P[p[0]] = [p[2], p[3]]; });
      var path = function (ids) {
        return ids.map(function (i) { return X(P[i][0]) + "," + Y(P[i][1]); }).join(" ");
      };
      // cathode arc 1 -> 2, built with RcT, theta_T
      var arc = [], RcT = r.RcT_mm, thT = r.thetaT * Math.PI / 180;
      for (var k = 0; k <= 40; k++) {
        var ph = thT * k / 40;
        arc.push(X(d + RcT - RcT * Math.cos(ph)) + "," + Y(RcT * Math.sin(ph)));
      }
      el("polyline", { points: arc.join(" "), class: "ds-cathode" }, svg);
      el("polyline", { points: path([2, 3, 4, 5, 6, 7]), class: "ds-electrode" }, svg);
      el("polyline", { points: path([8, 9, 10, 11, 12, 13, 14]), class: "ds-anode" }, svg);
    }
    var lab = function (z, rr, t) {
      var e = el("text", { x: X(z), y: Y(rr) - 6, class: "ds-label" }, svg);
      e.textContent = t;
    };
    lab(d + r.zw_mm, r.rw_mm, "waist");
    $("ds-caption").textContent = "Half cross-section, to scale (" + fmt(zMax, 1) +
      " mm long). Electrodes are Vedant's 15-point construction; the beam edge is a " +
      "schematic of the θ_raw cone and the drift to the waist, not a trajectory solve.";
  }

  // ---- Pareto via API, with a fallback -------------------------------------
  function apiUrl() { return (D.designMeta.api_url || "").replace(/\/$/, ""); }

  function loadFront() {
    var study = $("ds-study").value, box = $("ds-pareto-msg");
    var url = apiUrl();
    if (!url) { showFront(D.pareto[study], "bundled"); return; }
    box.innerHTML = msg("info", "Contacting the optimisation API… a free-tier server " +
      "sleeps when idle and can take 30–60 s to wake. The precomputed front is shown " +
      "meanwhile.");
    showFront(D.pareto[study], "bundled");
    var ctl = new AbortController();
    var timer = setTimeout(function () { ctl.abort(); }, 90000);
    fetch(url + "/pareto/" + study, { signal: ctl.signal })
      .then(function (res) { if (!res.ok) throw new Error("HTTP " + res.status); return res.json(); })
      .then(function (j) {
        clearTimeout(timer);
        showFront({ title: j.study.title, objectives: j.study.objectives, pareto: j.pareto,
                    knee: j.knee, evaluations: null,
                    verify: j.verify ? { n: j.verify.length,
                      max_rel_err: Math.max.apply(null, j.verify.map(function (v) { return v.max_err; })),
                      all_ok: j.verify.every(function (v) { return v.ok; }) } : null }, "live");
      })
      .catch(function (e) {
        clearTimeout(timer);
        box.innerHTML = msg("warn", "The API did not answer (" + esc(e.message) +
          "). Showing the precomputed front bundled with the page.");
      });
  }

  function showFront(f, source) {
    var box = $("ds-pareto-msg"), svg = $("ds-pareto-svg");
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    if (!f) { box.innerHTML = msg("bad", "No results for this study."); return; }
    if (source === "live") box.innerHTML = msg("info", "Live from the API.");
    else if (!apiUrl()) box.innerHTML = msg("info", "No API configured: showing the precomputed " +
      "front bundled with the page.");
    var o = f.objectives, xs = f.pareto.map(function (p) { return p["obj_" + o[0]]; });
    var ys = f.pareto.map(function (p) { return p["obj_" + o[1]]; });
    var W = 640, H = 280, pad = 44;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var X = function (v) { return pad + (v - x0) / ((x1 - x0) || 1) * (W - 2 * pad); };
    var Y = function (v) { return H - pad - (v - y0) / ((y1 - y0) || 1) * (H - 2 * pad); };
    el("line", { x1: pad, y1: H - pad, x2: W - pad, y2: H - pad, class: "ds-axis" }, svg);
    el("line", { x1: pad, y1: pad, x2: pad, y2: H - pad, class: "ds-axis" }, svg);
    var tx = el("text", { x: W / 2, y: H - 10, class: "ds-label", "text-anchor": "middle" }, svg);
    tx.textContent = o[0] + " (" + fmt(x0, 3) + " – " + fmt(x1, 3) + ")";
    var ty = el("text", { x: 12, y: H / 2, class: "ds-label",
                          transform: "rotate(-90 12 " + H / 2 + ")", "text-anchor": "middle" }, svg);
    ty.textContent = o[1] + " (" + fmt(y0, 3) + " – " + fmt(y1, 3) + ")";
    var knee = f.knee.index_in_pareto_csv;
    f.pareto.forEach(function (p, i) {
      var c = el("circle", { cx: X(xs[i]), cy: Y(ys[i]), r: i === knee ? 7 : 4,
                             class: i === knee ? "ds-knee" : "ds-pt", tabindex: 0 }, svg);
      var t = document.createElementNS(SVGNS, "title");
      t.textContent = "V " + fmt(p.V / 1000, 2) + " kV, rw " + fmt(p.rw_mm, 4) + " mm, Jc " +
        fmt(p.Jc, 3) + " A/cm², θ_T " + fmt(p.thetaT, 2) + "°" + (i === knee ? " (knee)" : "") +
        " — click to load";
      c.appendChild(t);
      var load = function () { setInputs({ V: p.V, I: p.I, rw_mm: p.rw_mm, Jc: p.Jc }); window.scrollTo(0, 0); };
      c.addEventListener("click", load);
      c.addEventListener("keydown", function (e) { if (e.key === "Enter") load(); });
    });
    var v = f.verify;
    $("ds-pareto-note").innerHTML = esc(f.title) + ". " + f.pareto.length + " Pareto designs" +
      (f.evaluations ? " from " + f.evaluations.toLocaleString() + " emulator evaluations" : "") +
      ". " + (v ? "Gate G3: " + v.n + " of them re-run through Vedant's exact solver, worst " +
      "relative error " + v.max_rel_err.toExponential(1) + (v.all_ok ? " — all pass." : " — FAILURES.")
      : "") + " The star is the knee (closest to the normalised ideal point). Click any point to load it above.";
  }

  function init() {
    if (!D.kfTable || !D.designMeta || !CL) return;
    initStats();
    var row = $("ds-presets");
    D.designMeta.presets.forEach(function (p) {
      var b = document.createElement("button");
      b.className = "ghost"; b.type = "button"; b.textContent = p.name;
      b.addEventListener("click", function () { setInputs(p); });
      row.appendChild(b);
    });
    FIELDS.forEach(function (f) { $(f[1]).addEventListener("input", update); });
    var sel = $("ds-study");
    Object.keys(D.pareto || {}).forEach(function (k) {
      var o = document.createElement("option"); o.value = k; o.textContent = D.pareto[k].title;
      sel.appendChild(o);
    });
    $("ds-load").addEventListener("click", loadFront);
    setInputs(D.designMeta.presets[0]);
    if (sel.options.length) showFront(D.pareto[sel.value], "bundled");
    sel.addEventListener("change", function () { showFront(D.pareto[sel.value], "bundled"); });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
