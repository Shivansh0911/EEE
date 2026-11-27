/*
 * The Predict tab's gun: cathode, beam-focusing electrode and anode face from
 * Tiwary & Basu (1988) only (EGUN.pierce.tbFromInputs), for the angle the
 * network predicts. Upper half (the gun is axisymmetric), one scale so angles are
 * true. The electrode's angle where it leaves the cathode rim is marked.
 */
(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var DEG = Math.PI / 180;
  var COL = { axis: "#4a5162", cathode: "#e9b34a", bfe: "#c8d0de", anode: "#8b94a8",
              beam: "#5aa2ff", beamFill: "rgba(90, 162, 255, 0.14)", ink: "#a7b0c0",
              angle: "#ef7070" };

  function el(name, attrs, parent, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    parent.appendChild(n);
    return n;
  }

  function render(host, tb, theta0) {
    while (host.firstChild) host.removeChild(host.firstChild);
    var W = 680, H = 320, pad = 26;
    host.setAttribute("viewBox", "0 0 " + W + " " + H);
    if (!tb) {
      el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", fill: COL.ink,
                   "font-size": 13 }, host, "No electrodes for these inputs.");
      return;
    }
    var pts = tb.bfe.concat(tb.anode);
    var zMax = Math.max.apply(null, pts.map(function (p) { return p[0]; })) * 1.08;
    var rMax = Math.max.apply(null, pts.map(function (p) { return p[1]; })) * 1.12;
    var s = Math.min((W - 2 * pad) / zMax, (H - 2 * pad - 10) / rMax);
    var x0 = pad + ((W - 2 * pad) - zMax * s) / 2, yAx = H - pad;
    var X = function (z) { return x0 + z * s; };
    var Y = function (r, m) { return yAx - m * r * s; };
    var line = function (P, m, attrs) {
      el("polyline", Object.assign({ points: P.map(function (p) { return X(p[0]) + "," + Y(p[1], m); }).join(" "),
        fill: "none" }, attrs), host);
    };
    el("line", { x1: X(0) - 10, y1: yAx, x2: X(zMax), y2: yAx, stroke: COL.axis,
                 "stroke-dasharray": "5 4" }, host);

    var th = theta0 * DEG, Rc = tb.Rc, Ra = tb.Ra;
    var rim = [Rc * (1 - Math.cos(th)), tb.rc];
    var edgeAtAnode = [Rc - Ra * Math.cos(th), Ra * Math.sin(th)];
    [1].forEach(function (m) {         // upper half: the gun is axisymmetric
      // beam: the theta0 cone from the cathode to the anode sphere
      el("polygon", { points: [[0, 0], rim, edgeAtAnode, [Rc - Ra, 0]].map(function (p) {
        return X(p[0]) + "," + Y(p[1], m); }).join(" "), fill: COL.beamFill, stroke: "none" }, host);
      line([rim, edgeAtAnode], m, { stroke: COL.beam, "stroke-width": 1.4, "stroke-dasharray": "6 3" });
      var arc = [];
      for (var k = 0; k <= 40; k++) {
        var a = th * k / 40;
        arc.push([Rc - Rc * Math.cos(a), Rc * Math.sin(a)]);
      }
      line(arc, m, { stroke: COL.cathode, "stroke-width": 3 });
      line(tb.bfe, m, { stroke: COL.bfe, "stroke-width": 2.6 });
      line(tb.anode, m, { stroke: COL.anode, "stroke-width": 2.6 });
    });

    // the electrode's angle at the rim: arc from the beam edge to the electrode
    var R = Math.min(0.28 * tb.rc, 0.5 * Math.hypot(tb.bfeTip[0] - rim[0], tb.bfeTip[1] - rim[1]));
    var a0 = -th, a1 = -th + 67.5 * DEG, arcA = [];
    for (var j = 0; j <= 24; j++) {
      var b = a0 + (a1 - a0) * j / 24;
      arcA.push([rim[0] + R * Math.cos(b), rim[1] + R * Math.sin(b)]);
    }
    line(arcA, 1, { stroke: COL.angle, "stroke-width": 1.6 });
    var mid = (a0 + a1) / 2;
    el("text", { x: X(rim[0] + 1.25 * R * Math.cos(mid)) + 4, y: Y(rim[1] + 1.25 * R * Math.sin(mid), 1),
                 fill: COL.angle, "font-size": 12, "font-family": "system-ui, sans-serif" }, host,
       "67.5° to beam edge = " + (67.5 - theta0).toFixed(1) + "° to axis");

    var lab = function (p, t, dy) {
      el("text", { x: X(p[0]) + 6, y: Y(p[1], 1) + (dy || -6), fill: COL.ink, "font-size": 12,
                   "font-family": "system-ui, sans-serif" }, host, t);
    };
    lab(tb.bfeTip, "focus electrode, V = 0", -8);
    lab(tb.anodeTop, "anode face, V = Va", -8);
    lab([0, 0], "cathode", 16);
  }

  root.EGUN = root.EGUN || {};
  root.EGUN.tbgun = { render: render };
})(typeof globalThis !== "undefined" ? globalThis : this);
