/*
 * Simulation tab, top section: our gun in target.exe while one quantity changes.
 *   - anode shift: the anode moved along the axis (spacing d), vs 1/d^2 and Langmuir-Blodgett
 *   - cathode R_c: the cathode's radius of curvature, anode re-placed by Langmuir-Blodgett
 * Data: data/sweeps.js, written by `python -m src.sim.sweeps` (each step is a real run).
 * Also the local live run: when the page is served by START_HERE.bat (serve_local.py),
 * /api/simulate runs target.exe on this computer.
 */
(function () {
  "use strict";

  var DATA = (window.EGUN_DATA || {}).sweeps;
  var $ = function (id) { return document.getElementById(id); };
  var NS = "http://www.w3.org/2000/svg";
  var state = { gun: null, mode: "anode", i: 0 };

  function el(name, attrs, parent, text) {
    var e = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  }
  function clear(svg) { while (svg.firstChild) svg.removeChild(svg.firstChild); }
  function f(v, n) { return v === null || v === undefined ? "—" : v.toFixed(n); }
  function sgn(v, n) { return (v >= 0 ? "+" : "") + v.toFixed(n); }

  /* trajectories, electrodes, equipotentials of one run, with the focus, the cathode's
     aim point and where electrons hit the gun marked */
  function drawRun(svg, plot, marks) {
    clear(svg);
    var zs = [], rs = [];
    plot.boundary.forEach(function (s) { s.forEach(function (p) { zs.push(p[0]); rs.push(p[1]); }); });
    var zMax = Math.max.apply(null, zs), rMax = Math.max.apply(null, rs);
    var W = 760, pad = 20, k = (W - 2 * pad) / zMax, H = Math.round(rMax * k) + 2 * pad + 14;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var X = function (z) { return pad + z * k; }, Y = function (r) { return H - pad - r * k; };
    var line = function (seg, cls) {
      el("polyline", { points: seg.map(function (p) { return X(p[0]) + "," + Y(p[1]); }).join(" "), class: cls }, svg);
    };
    plot.equipotentials.forEach(function (s) { line(s, "sim-eq"); });
    plot.trajectories.forEach(function (s) { line(s, "sim-traj"); });
    plot.boundary.forEach(function (s) { line(s, "sim-bnd"); });
    el("line", { x1: X(0), y1: Y(0), x2: X(zMax), y2: Y(0), class: "ds-axis" }, svg);
    // cathode vertex: the boundary point nearest the axis with the smallest z
    var zc = Infinity;
    plot.boundary.forEach(function (s) { s.forEach(function (p) { if (p[1] < 0.02 * rMax && p[0] < zc) zc = p[0]; }); });
    if (!isFinite(zc)) zc = 0;
    var tag = function (z, cls, text, row) {
      el("line", { x1: X(z), y1: Y(0) + 6, x2: X(z), y2: Y(rMax * 0.55), class: cls }, svg);
      el("text", { x: X(z) + 4, y: Y(rMax * 0.55) - 4 - 13 * row, class: "sw-tag", fill: cls === "sw-aim" ? "#1f77b4" : "#9467bd" }, svg, text);
    };
    if (marks.aim !== undefined && zc + marks.aim < zMax) tag(zc + marks.aim, "sw-aim", "aim point (R c)", 1);
    if (marks.focus !== null && marks.focus !== undefined) tag(zc + marks.focus, "sw-focus", "focus " + marks.focus.toFixed(1) + " mm", 0);
    // electrons that stop before the end and before the focus hit the gun
    var zEnd = 0;
    plot.trajectories.forEach(function (s) { zEnd = Math.max(zEnd, s[s.length - 1][0]); });
    var cut = marks.focus === null || marks.focus === undefined ? Infinity : zc + marks.focus;
    var hits = 0;
    plot.trajectories.forEach(function (s) {
      var p = s[s.length - 1];
      if (p[0] < zEnd - 1 && p[0] < cut) {
        el("circle", { cx: X(p[0]), cy: Y(p[1]), r: 5, class: "sw-hit" }, svg);
        hits++;
      }
    });
    if (hits) el("text", { x: X(zMax) - 4, y: pad + 4, "text-anchor": "end", class: "sw-tag", fill: "#ff7f0e" }, svg,
                 "orange circles: electrons hitting the gun");
    return zMax;
  }

  /* one sweep chart: x against y for the ok runs, optional dashed predictions, the current step ringed */
  function chart(svg, rows, xk, series, now, xlabel, ylabel, target) {
    clear(svg);
    var W = 420, H = 250, L = 52, R = 14, T = 14, B = 40;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var xs = rows.map(function (r) { return r[xk]; });
    var ys = [];
    series.forEach(function (s) { rows.forEach(function (r) { if (r[s.key] !== null && r[s.key] !== undefined) ys.push(r[s.key]); }); });
    if (target !== undefined) ys.push(target);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);
    var y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys), py = (y1 - y0) * 0.08 || 1;
    y0 -= py; y1 += py;
    var X = function (v) { return L + (v - x0) / (x1 - x0 || 1) * (W - L - R); };
    var Y = function (v) { return H - B - (v - y0) / (y1 - y0) * (H - T - B); };
    for (var j = 0; j <= 4; j++) {
      var yv = y0 + (y1 - y0) * j / 4;
      el("line", { x1: L, y1: Y(yv), x2: W - R, y2: Y(yv), class: "sim-grid" }, svg);
      el("text", { x: L - 6, y: Y(yv) + 4, "text-anchor": "end", class: "ds-label" }, svg, yv.toPrecision(3));
    }
    xs.forEach(function (v, i) {
      if (rows.length <= 10 || i % 2 === 0)
        el("text", { x: X(v), y: H - B + 15, "text-anchor": "middle", class: "ds-label" }, svg, v.toFixed(rows.length > 6 && xk !== "anode_shift_mm" ? 0 : 0));
    });
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "ds-label" }, svg, xlabel);
    el("text", { x: 12, y: (T + H - B) / 2, "text-anchor": "middle", class: "ds-label", transform: "rotate(-90 12 " + (T + H - B) / 2 + ")" }, svg, ylabel);
    if (target !== undefined) el("line", { x1: L, y1: Y(target), x2: W - R, y2: Y(target), class: "sw-target" }, svg);
    series.forEach(function (s) {
      var pts = rows.filter(function (r) { return r[s.key] !== null && r[s.key] !== undefined; })
        .map(function (r) { return X(r[xk]) + "," + Y(r[s.key]); }).join(" ");
      el("polyline", { points: pts, class: s.cls }, svg);
      if (s.cls === "sw-line") rows.forEach(function (r) {
        if (r[s.key] !== null && r[s.key] !== undefined) el("circle", { cx: X(r[xk]), cy: Y(r[s.key]), r: 3.5, class: "sw-dot" }, svg);
      });
    });
    var main = series[0].key;
    if (now && now[main] !== null && now[main] !== undefined)
      el("circle", { cx: X(now[xk]), cy: Y(now[main]), r: 8, class: "sw-now" }, svg);
  }

  function rows() {
    return DATA[state.gun][state.mode];
  }

  function render() {
    var g = DATA[state.gun], all = rows(), r = all[state.i], ok = all.filter(function (x) { return x.ok; });
    var d = g.design;
    var label = state.mode === "anode" ? "anode " + sgn(r.step, 0) + " mm"
      : "R c " + (r.ok ? r.RcT_mm.toFixed(1) + " mm" : "") + " (angle " + sgn(r.step, 0) + "°)";
    $("sw-step-label").textContent = label;
    var st = function (n, l) { return '<div class="stat"><div class="n">' + n + '</div><div class="l">' + l + "</div></div>"; };
    if (!r.ok) {
      $("sw-stats").innerHTML = st("failed", "target.exe could not run this gun");
      clear($("sw-svg"));
      $("sw-caption").textContent = r.error;
      return;
    }
    $("sw-stats").innerHTML =
      st(r.sim_microperveance.toFixed(3), "microperveance (design " + d.P + ", " + sgn(r.perveance_err_pct, 1) + " %)") +
      st(r.throw_mm === null ? "none" : r.throw_mm.toFixed(1) + " mm", "focus point from the cathode") +
      st(r.waist_mm === null ? "none" : r.waist_mm.toFixed(2) + " mm", "waist radius (design " + d.rw_mm + " mm)") +
      st(r.lost_in_gun + " / " + r.n_trajectories, "electrons hitting the gun");
    drawRun($("sw-svg"), r.plot, { focus: r.throw_mm, aim: state.mode === "rc" ? r.aim_point_mm : undefined });
    $("sw-caption").textContent = "target.exe run: " + label + ". Cathode–anode spacing d = " +
      r.cathode_anode_mm.toFixed(2) + " mm, cathode R c = " + r.RcT_mm.toFixed(1) + " mm, beam angle " +
      r.thetaT_deg.toFixed(1) + "°. " + (r.lost_in_gun ? r.lost_in_gun + " electrons hit the gun (orange)." :
      "No electron hits the gun.");
    if (state.mode === "anode") {
      chart($("sw-chart1"), ok, "anode_shift_mm", [{ key: "sim_microperveance", cls: "sw-line" },
        { key: "P_child_1_over_d2", cls: "sw-pred" }, { key: "P_langmuir_blodgett", cls: "sw-pred2" }],
        r, "anode shift (mm)", "microperveance", d.P);
      $("sw-chart1-cap").textContent = "Current drawn: target.exe (blue) against 1/d² scaling (grey dashes) and " +
        "Langmuir–Blodgett (orange dots). Moving the anode closer draws more current. Line: design.";
      chart($("sw-chart2"), ok, "anode_shift_mm", [{ key: "throw_mm", cls: "sw-line" }], r,
        "anode shift (mm)", "focus point (mm)");
      $("sw-chart2-cap").textContent = "Where the beam focuses: moving the anode away pushes the focus further down the tube.";
    } else {
      chart($("sw-chart1"), ok, "RcT_mm", [{ key: "throw_mm", cls: "sw-line" }, { key: "aim_point_mm", cls: "sw-pred" }], r,
        "cathode radius of curvature R c (mm)", "distance from cathode (mm)");
      $("sw-chart1-cap").textContent = "Focus point in target.exe (blue) and the cathode's aim point, its centre of curvature " +
        "(grey dashes). Space charge pushes the real focus well beyond the aim point.";
      chart($("sw-chart2"), ok, "RcT_mm", [{ key: "waist_mm", cls: "sw-line" }], r,
        "R c (mm)", "waist radius (mm)", d.rw_mm);
      $("sw-chart2-cap").textContent = "Waist radius: a flatter cathode (larger R c) converges less, so the waist grows; " +
        "too flat and the beam hits the gun. Line: design waist.";
    }
  }

  function setMode(m) {
    state.mode = m;
    $("sw-mode-anode").setAttribute("aria-pressed", m === "anode");
    $("sw-mode-rc").setAttribute("aria-pressed", m === "rc");
    var all = rows(), s = $("sw-step");
    s.max = all.length - 1;
    var zero = 0;
    all.forEach(function (r, i) { if (r.step === 0) zero = i; });
    state.i = zero;
    s.value = zero;
    render();
  }

  function initSweeps() {
    if (!DATA || !$("sw-gun")) return;
    var names = Object.keys(DATA), pick = $("sw-gun");
    var nice = { sir_gun: "Sir's klystron gun (150 kV)" };
    names.forEach(function (n) {
      var o = document.createElement("option");
      o.value = n; o.textContent = nice[n] || ("Paper " + n.replace("case", "case ") + " (10 kV)");
      pick.appendChild(o);
    });
    state.gun = names.indexOf("sir_gun") >= 0 ? "sir_gun" : names[0];
    pick.value = state.gun;
    pick.addEventListener("change", function () { state.gun = pick.value; setMode(state.mode); });
    $("sw-mode-anode").addEventListener("click", function () { setMode("anode"); });
    $("sw-mode-rc").addEventListener("click", function () { setMode("rc"); });
    $("sw-step").addEventListener("input", function () { state.i = +this.value; render(); });
    setMode("anode");
  }

  /* ---- local live run (only when served by serve_local.py) ---- */
  function initLive() {
    if (!$("live-card") || location.protocol === "file:") return;
    fetch("/api/status").then(function (r) { return r.ok ? r.json() : null; }).then(function (s) {
      if (!s) return;
      $("live-card").hidden = false;
      $("lv-status").textContent = s.target_exe ? "target.exe found: " + s.target_exe
        : "target.exe not found. Put it in simulation/lktron/ and restart START_HERE.bat.";
      $("lv-run").disabled = !s.target_exe;
    }).catch(function () { /* static hosting: no live runs */ });
    $("lv-run").addEventListener("click", function () {
      var body = { P: +$("lv-P").value, rw: +$("lv-rw").value, C: +$("lv-C").value,
                   V: +$("lv-V").value * 1e3, anode_shift: +$("lv-dz").value, theta_offset: +$("lv-dth").value };
      $("lv-status").textContent = "Running target.exe … (a minute or less)";
      $("lv-run").disabled = true;
      var t0 = Date.now();
      fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
        .then(function (r) { return r.json(); })
        .then(function (res) {
          $("lv-run").disabled = false;
          if (!res.ok) { $("lv-status").textContent = "target.exe run failed: " + res.error; return; }
          var c = res.compare, i = res.info;
          $("lv-status").textContent = "Done in " + ((Date.now() - t0) / 1000).toFixed(0) + " s. Files: " + res.folder;
          drawRun($("lv-svg"), res.plot, { focus: c.waist_inside_domain ? c.sim_throw_mm : null });
          $("lv-caption").textContent = "Our network's angle " + i.theta_net_deg.toFixed(2) + "°, built at " +
            i.thetaT_built_deg.toFixed(2) + "°; R c " + i.RcT_mm.toFixed(1) + " mm; spacing d " + i.cathode_anode_mm.toFixed(2) +
            " mm. target.exe: microperveance " + c.sim_microperveance.toFixed(3) + " vs " + i.P + " (" +
            sgn(c.perveance_err_pct, 1) + " %), current " + c.sim_current_A.toPrecision(4) + " A, " +
            (c.waist_inside_domain ? "waist " + c.sim_waist_mm.toFixed(2) + " mm at " + c.sim_throw_mm.toFixed(1) + " mm" : "no waist in the domain") +
            ", " + c.lost_in_gun + "/" + c.n_trajectories + " electrons hit the gun.";
        })
        .catch(function (e) { $("lv-run").disabled = false; $("lv-status").textContent = "Error: " + e; });
    });
  }

  function init() { initSweeps(); initLive(); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
