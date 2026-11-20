/*
 * The Simulation tab: recorded PIERCE (target.exe) runs of synthesised guns.
 * Data: data/simulation.js, written by `python -m src.sim.report`.
 * Colours follow sir's pierce_plotter.py: black electrodes, red trajectories,
 * green equipotentials.
 */
(function () {
  "use strict";

  var S = (window.EGUN_DATA || {}).simulation;
  var $ = function (id) { return document.getElementById(id); };
  var NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent, text) {
    var e = document.createElementNS(NS, name);
    Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (text !== undefined) e.textContent = text;
    parent.appendChild(e);
    return e;
  }
  function f(v, n) { return v === null || v === undefined ? "—" : v.toFixed(n); }
  function sg(v, n) { return v === null || v === undefined ? "" : (v >= 0 ? "+" : "") + v.toFixed(n) + " %"; }

  function stats() {
    var byKey = {};
    S.variants.forEach(function (v) { byKey[v.label] = v; });
    var ved = S.variants[1].compare, t1 = S.variants[7].compare, t2 = S.variants[8].compare;
    var st = function (n, l) { return '<div class="stat"><div class="n">' + n + '</div><div class="l">' + l + "</div></div>"; };
    $("sim-stats").innerHTML =
      st(sg(ved.current_err_pct, 1), "Vedant's deck as built: current vs design, in PIERCE") +
      st(sg(t1.current_err_pct, 1), "after tuning the electrode angle (ψ " + f(S.variants[7].notes.psi_deg, 1) + "°)") +
      st(sg(t2.current_err_pct, 1) + " / " + sg(t2.waist_err_pct, 1),
         "current / waist after tuning angle and anode (ψ " + f(S.tuned_2d.psi_deg, 1) +
         "°, anode +" + f(S.tuned_2d.anode_shift_mm, 2) + " mm)");
  }

  function table() {
    var d = S.design;
    $("sim-design").innerHTML = "Gun: " + d.V / 1e3 + " kV, " + d.I + " A, waist " + d.rw_mm +
      " mm, " + d.Jc + " A/cm² (the CEERI 8 MW klystron gun, the one sir's own reference deck " +
      "simulates). Design from Vedant's solver: microperveance " + d.muP.toFixed(4) + ", throw " +
      d.zw_mm.toFixed(1) + " mm. Grid " + S.grid_mm + " mm.";
    var h = "<tr><th>Run</th><th>Current A</th><th>vs design</th><th>Lost in gun</th>" +
      "<th>Throw mm</th><th>Waist mm</th><th>vs design</th></tr>";
    S.variants.forEach(function (v) {
      var c = v.compare;
      h += "<tr><td>" + v.label + "</td><td>" + f(c.sim_current_A, 2) + "</td><td>" +
        sg(c.current_err_pct, 1) + "</td><td>" + c.lost_in_gun + " / " + c.n_trajectories +
        "</td><td>" + (c.waist_inside_domain ? f(c.sim_throw_mm_refined, 1) : "no waist") +
        "</td><td>" + (c.waist_inside_domain ? f(c.sim_waist_mm_refined, 2) : "no waist") +
        "</td><td>" + (c.waist_inside_domain ? sg(c.waist_err_pct, 1) : "") + "</td></tr>";
    });
    $("sim-table").innerHTML = h;
  }

  function draw(i) {
    var v = S.variants[i], svg = $("sim-svg"), P = v.plot;
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var zs = [], rs = [];
    P.boundary.forEach(function (s) { s.forEach(function (p) { zs.push(p[0]); rs.push(p[1]); }); });
    var zMax = Math.max.apply(null, zs), rMax = Math.max.apply(null, rs);
    var W = 760, H = Math.round(W * rMax / zMax) + 40, pad = 20;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var k = (W - 2 * pad) / zMax;
    var X = function (z) { return pad + z * k; }, Y = function (r) { return H - pad - r * k; };
    var line = function (seg, cls) {
      el("polyline", { points: seg.map(function (p) { return X(p[0]) + "," + Y(p[1]); }).join(" "),
                       class: cls }, svg);
    };
    P.equipotentials.forEach(function (s) { line(s, "sim-eq"); });
    P.trajectories.forEach(function (s) { line(s, "sim-traj"); });
    P.boundary.forEach(function (s) { line(s, "sim-bnd"); });
    el("line", { x1: X(0), y1: Y(0), x2: X(zMax), y2: Y(0), class: "ds-axis" }, svg);
    var c = v.compare;
    $("sim-caption").textContent = v.label + ": " + f(c.sim_current_A, 2) + " A (" +
      sg(c.current_err_pct, 1) + " vs the design 120 A), " + c.lost_in_gun + " of " +
      c.n_trajectories + " trajectories lost in the gun, " +
      (c.waist_inside_domain ? "waist " + f(c.sim_waist_mm_refined, 2) + " mm at " +
        f(c.sim_throw_mm_refined, 1) + " mm from the cathode." : "no waist inside the domain.") +
      " " + zMax.toFixed(0) + " mm shown; every second trajectory drawn.";
  }

  function sweep() {
    var svg = $("sim-sweep"), W = 640, H = 280, L = 52, R = 52, T = 16, B = 40;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var pts = S.sweep;
    var x0 = 55, x1 = 75, c0 = 100, c1 = 180, w0 = 4, w1 = 14;
    var X = function (v) { return L + (v - x0) / (x1 - x0) * (W - L - R); };
    var Yc = function (v) { return H - B - (v - c0) / (c1 - c0) * (H - T - B); };
    var Yw = function (v) { return H - B - (v - w0) / (w1 - w0) * (H - T - B); };
    for (var x = 55; x <= 75; x += 5) {
      el("line", { x1: X(x), y1: T, x2: X(x), y2: H - B, class: "sim-grid" }, svg);
      el("text", { x: X(x), y: H - B + 16, "text-anchor": "middle", class: "ds-label" }, svg, x + "°");
    }
    for (var c = 100; c <= 180; c += 20)
      el("text", { x: L - 6, y: Yc(c) + 4, "text-anchor": "end", class: "sim-lab-c" }, svg, c + " A");
    for (var w = 4; w <= 14; w += 2)
      el("text", { x: W - R + 6, y: Yw(w) + 4, class: "sim-lab-w" }, svg, w + " mm");
    el("line", { x1: L, y1: Yc(120), x2: W - R, y2: Yc(120), class: "sim-target-c" }, svg);
    el("line", { x1: L, y1: Yw(9.8), x2: W - R, y2: Yw(9.8), class: "sim-target-w" }, svg);
    var cp = pts.map(function (p) { return X(p[0]) + "," + Yc(p[1]); }).join(" ");
    el("polyline", { points: cp, class: "sim-curve-c" }, svg);
    var wp = pts.filter(function (p) { return p[2] !== null; })
      .map(function (p) { return X(p[0]) + "," + Yw(p[2]); }).join(" ");
    el("polyline", { points: wp, class: "sim-curve-w" }, svg);
    pts.forEach(function (p) {
      el("circle", { cx: X(p[0]), cy: Yc(p[1]), r: 4, class: "sim-dot-c" }, svg);
      if (p[2] !== null) el("rect", { x: X(p[0]) - 4, y: Yw(p[2]) - 4, width: 8, height: 8, class: "sim-dot-w" }, svg);
    });
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle", class: "ds-label" }, svg,
       "focus electrode angle to the beam edge ψ");
    el("text", { x: L + 8, y: Yc(120) - 6, class: "sim-lab-c" }, svg, "design current 120 A (circles)");
    el("text", { x: W - R - 8, y: Yw(9.8) - 6, "text-anchor": "end", class: "sim-lab-w" }, svg,
       "design waist 9.8 mm (squares)");
  }

  function init() {
    if (!S) return;
    stats();
    table();
    var pick = $("sim-pick");
    S.variants.forEach(function (v, i) {
      var o = document.createElement("option");
      o.value = i; o.textContent = v.label;
      pick.appendChild(o);
    });
    pick.value = 8;
    pick.addEventListener("change", function () { draw(+pick.value); });
    draw(8);
    sweep();
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
