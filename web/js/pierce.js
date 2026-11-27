/*
 * Decks for Dr. Lalit Kumar's PIERCE simulator (target.exe), and the
 * Tiwary & Basu (1988) focus-electrode / anode synthesis, in the browser.
 *
 * A line-for-line port of src/sim/lktron.py (deck_text, rows_from_vedant,
 * choose_grid), src/sim/experiments.py (tiwary_basu_rows, extend_tube) and
 * src/physics/bfe.py. tests/test_pierce_js_parity.py checks the deck text is
 * identical to the Python's, character for character.
 *
 * Two fixes over Vedant's own deck writer, both found by running PIERCE:
 * the &CATHODE namelist is closed with "/", and the &CONTROL/&PRINTS/&PLOT
 * values are those of sir's reference deck.
 */
(function (root) {
  "use strict";

  var CL = root.EGUN && root.EGUN.closure;
  if (!CL && typeof require !== "undefined") CL = require("./closure.js");
  var DEG2RAD = Math.PI / 180;
  var NICE_H = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0];
  var MAX_INTERVALS = 500;
  var TUBE_FACTOR = 2.0;

  // ---- complex arithmetic ---------------------------------------------------
  function C(re, im) { return { re: re, im: im || 0 }; }
  function add(a, b) { return C(a.re + b.re, a.im + b.im); }
  function sub(a, b) { return C(a.re - b.re, a.im - b.im); }
  function mul(a, b) { return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
  function div(a, b) {
    var d = b.re * b.re + b.im * b.im;
    return C((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
  }
  function abs(a) { return Math.hypot(a.re, a.im); }
  function cpow(a, p) {                       // CPython's complex ** float
    var len = Math.pow(Math.hypot(a.re, a.im), p), ph = Math.atan2(a.im, a.re) * p;
    return C(len * Math.cos(ph), len * Math.sin(ph));
  }

  // ---- Tiwary & Basu: the alpha series in the complex plane ----------------
  var A = [1.0, 0.3, 0.075, 0.01432, 0.00216, 0.00035];
  var PIERCE = 3 * Math.PI / 8, GAMMA_MAX = 2.9;

  function alpha(g) {
    var out = C(0, 0);
    for (var k = A.length - 1; k >= 0; k--) out = mul(add(out, C(A[k])), g);
    return out;
  }
  function dalpha(g) {
    var out = C(0, 0);
    for (var k = A.length - 1; k >= 0; k--) out = add(mul(out, g), C((k + 1) * A[k]));
    return out;
  }
  function alphaInv(a, g0) {
    var g = g0;
    for (var i = 0; i < 60; i++) {
      var step = div(sub(alpha(g), a), dalpha(g));
      g = sub(g, step);
      if (abs(step) < 1e-14 * Math.max(1, abs(g))) return g;
    }
    throw new Error("alpha inversion did not converge");
  }
  function trace(target, params, g0) {
    var out = [], g = g0;
    for (var i = 0; i < params.length; i++) {
      g = alphaInv(target(params[i]), g);
      if (abs(g) > GAMMA_MAX) break;
      out.push(C(-g.re, -g.im));
    }
    return out;
  }
  function bfeS(n) {
    n = n || 400;
    var ts = [];
    for (var i = 1; i <= n; i++) ts.push(i * (6.0 / n));
    var e = C(Math.cos(PIERCE), -Math.sin(PIERCE));
    var s = trace(function (t) { return C(t * e.re, t * e.im); }, ts, C(ts[0] * e.re, ts[0] * e.im));
    return [C(0, 0)].concat(s);
  }
  function anodeS(y, n) {
    n = n || 400;
    var K = cpow(alpha(C(y)), 4.0 / 3.0), taus = [];
    for (var i = 0; i <= n; i++) taus.push(-(i * (40.0 / n)));
    return trace(function (tau) { return cpow(C(K.re, K.im + tau), 0.75); }, taus, C(y));
  }
  function toGun(s, Rc, th0, d) {
    return s.map(function (p) {
      var rho = Rc * Math.exp(p.re), th = th0 + p.im;
      return [d + Rc - rho * Math.cos(th), rho * Math.sin(th)];
    });
  }
  function upToApex(P) {
    var k = 0;
    for (var i = 1; i < P.length; i++) if (P[i][1] > P[k][1]) k = i;
    return P.slice(0, k + 1);
  }

  /* Focus electrode (V = 0) and anode (V = Va) equipotentials, each cut at its
     highest point, for synthesis result r (lengths in mm), cathode vertex at d. */
  function tiwaryBasuCurves(r, d) {
    var th = r.theta * DEG2RAD;
    var B = upToApex(toGun(bfeS(), r.Rc_mm, th, d));
    var An = upToApex(toGun(anodeS(Math.log(r.Rc_mm / r.Ra_mm)), r.Rc_mm, th, d));
    An = An.filter(function (p) { return p[1] >= r.ra_mm; });
    return { bfe: B, anode: An };
  }

  // ---- deck rows --------------------------------------------------------------
  function tiwaryBasuRows(r, D) {
    var c = tiwaryBasuCurves(r, D), B = c.bfe, An = c.anode;
    if (!An.length) throw new Error("anode equipotential never clears the aperture radius");
    var Rc = r.Rc_mm, rc = r.rc_mm, V = r.V, th = r.theta * DEG2RAD, Rmax = 1.75 * rc;
    var zAp = An[0][0];
    var tanPhi2 = 0.17409 * Math.sqrt(r.muP * Math.abs(Math.log(r.R)));
    var z12 = D + Rc, r12 = r.ra_mm - (z12 - zAp) * tanPhi2, Zmax = D + TUBE_FACTOR * r.zw_mm;
    var rows = [{ code: 0, z: D, r: 0.0, volt: 0.0, zc: D + Rc, rad: Rc },
                { code: 1, z: D + Rc * (1 - Math.cos(th)), r: rc, volt: 0.0, exact: true }];
    B.slice(1).forEach(function (p) { rows.push({ code: 1, z: p[0], r: p[1], volt: 0.0 }); });
    var tip = B[B.length - 1], top = An[An.length - 1];
    rows.push({ code: -1, z: tip[0], r: Rmax, volt: 0.0 });
    rows.push({ code: 1, z: top[0], r: Rmax, volt: V });
    An.slice().reverse().forEach(function (p) { rows.push({ code: 1, z: p[0], r: p[1], volt: V }); });
    rows.push({ code: 1, z: z12, r: r12, volt: V }, { code: 1, z: Zmax, r: r12, volt: V },
              { code: 1, z: Zmax, r: 0.0, volt: V });
    return { rows: rows, crad: Rc, curves: c };
  }

  /* Vedant's 15-point construction at grid H, vertex offset d and focus-electrode
     angle psi, coded as his print_gun_points codes it, drift tube extended. */
  function vedantRows(r, H, D, psi) {
    var dp = CL.designParams(r, { H: H, d: D, psi: psi });
    var pts = CL.gunPoints(r, dp), RcT = r.RcT_mm, rows = [];
    pts.forEach(function (p) {
      var pid = p[0];
      if (pid === 0) return;
      var row = { code: 1, z: p[2], r: p[3], volt: pid <= 7 ? 0.0 : r.V, exact: pid === 2 };
      if (pid === 1) { row.code = 0; row.zc = D + RcT; row.rad = RcT; }
      else if (pid === 3 || pid === 7) row.code = -1;
      rows.push(row);
    });
    var zmax = D + TUBE_FACTOR * r.zw_mm;
    rows[rows.length - 2].z = zmax;
    rows[rows.length - 1].z = zmax;
    return { rows: rows, crad: RcT, points: pts, params: dp };
  }

  // ---- Python-compatible formatting ---------------------------------------
  function roundHalfEven(x) {
    var r = Math.round(x);
    if (Math.abs(x - Math.trunc(x)) === 0.5) r = 2 * Math.round(x / 2);
    return r;
  }
  function snap(v, h) { return Number((roundHalfEven(v / h) * h).toFixed(10)); }
  function pad(s, w) { s = String(s); while (s.length < w) s += " "; return s; }
  function padR(s, w) { s = String(s); while (s.length < w) s = " " + s; return s; }
  function pyFloat(x) { return Number.isInteger(x) ? x.toFixed(1) : String(x); }
  function pyG(x) { return String(Number(x.toPrecision(6))); }

  function chooseGrid(rc, zmax) {
    var target = rc / 50, h = NICE_H[0];
    NICE_H.forEach(function (x) { if (x <= target) h = x; });
    if (zmax !== undefined) {
      var need = zmax / (0.96 * MAX_INTERVALS);
      var up = NICE_H.filter(function (x) { return x >= need; })[0];
      h = Math.max(h, up);
    }
    return h;
  }

  /* lktron.deck_text */
  function deckText(rows, o) {
    var h = o.h, dz = o.dz === undefined ? 4.0 * h : o.dz, dx = o.dx === undefined ? 1.2 * h : o.dx;
    var maxtr = o.maxtr === undefined ? Math.max(200, Math.ceil(o.crad / h / 50.0) * 50) : o.maxtr;
    var lines = [], last = null;
    rows.forEach(function (p) {
      var key = snap(p.z, h) + "," + snap(p.r, h);
      if (key === last && !p.exact) return;
      last = key;
      var zOut = p.exact ? p.z.toFixed(8) : snap(p.z, h).toFixed(2);
      lines.push(padR(p.code, 2) + ",  " + pad(zOut, 16) + "  " + pad(snap(p.r, h).toFixed(2), 8) +
        "  " + pad(snap(p.volt, h).toFixed(4), 12) + "  " + pad(snap(p.zc || 0, h).toFixed(6), 12) +
        "  " + pad(snap(p.rcen || 0, h).toFixed(5), 10) + "  " + snap(p.rad || 0, h).toFixed(4));
    });
    var n = lines.length;
    var head = " '" + o.title.slice(0, 60) + "'\n &BOUND\n" +
      " H =   " + pyFloat(h) + "," + pyFloat(h) + ", MBP = " + n + ", NT = " + n +
      ", NGRD = 0, jMAG = 0, NMAG = 0, MAGORD = 2,\n /\n" +
      " &CONTROL\n" +
      " IOPT = 2, ICLOS = 0, IOS = 0, IORDR=1, IPONT = 5, NMAX = 2000, EL =   0.00000000050, \n" +
      " maxtr=" + maxtr + ", lmax=20,LC=1,eqstep=0.1,1.,10., DELIN = 0.2 , BETA = 0.6,\n" +
      " /\n &PRINTS\n IPOT = 2, IFLDZ = 2, IFLDR = 2, ISPC = 2, IRAY = 2,\n /\n" +
      " &PLOT\n ITRPL = 2, IPOTPL = 2, ICDPL = 2, ISCDPL = 2,imesh= 0\n / \n";
    var tail = " &GRID\n /\n &CATHODE CSHAPE= \"SPHERE\", EMISSION=\"CHILD\", NSEG = 1, OCAT = " +
      o.d.toFixed(2) + ",\n CRAD = " + o.crad.toFixed(2) + ", RMAX = " + o.rc.toFixed(2) +
      " ,DZ = " + pyG(dz) + " ,DX =  " + pyG(dx) + "\n /\n";
    return head + lines.join("\n") + "\n" + tail;
  }

  /* One call for the page: a PIERCE-ready deck for synthesis result r.
     opts: electrodes ("vedant" | "tiwary-basu"), psi, anodeShift, grid. */
  function pierceDeck(r, opts) {
    opts = opts || {};
    if (!r.feasible) return { refused: "design is infeasible: " + r.reason };
    if (!r.za_valid) return { refused: "theta_raw > 56.44 deg: Vaughan Eq.(20) has no anode plane" };
    var h = opts.grid || chooseGrid(r.rc_mm, TUBE_FACTOR * r.zw_mm + 1), D = h;
    try {
      var b = opts.electrodes === "tiwary-basu" ? tiwaryBasuRows(r, D)
        : vedantRows(r, h, D, opts.psi);
      var dz = opts.anodeShift || 0;
      if (dz) b.rows.forEach(function (row) { if (row.volt > 0) row.z += dz; });
      var title = "PIERCE GUN " + pyG(r.V / 1e3) + "KV " + pyG(r.I) + "A " +
        (opts.electrodes === "tiwary-basu" ? "TIWARY-BASU" : "VEDANT");
      return { text: deckText(b.rows, { title: title, h: h, d: D, crad: b.crad, rc: r.rc_mm }),
               rows: b.rows, grid: h, curves: b.curves || null };
    } catch (e) {
      return { refused: e.message };
    }
  }

  /* bfe.from_inputs: Tiwary & Basu's electrodes from the Predict tab's inputs
     (P, rw, C) and its angle theta0. Rc/Ra from the Langmuir-Blodgett cone
     relation P = 14.665 (1 - cos theta0) / alpha^2. Cathode vertex at z = 0. */
  var CONE = 14.664652527448363;          // lb.CONE_CONST: (16 pi/9) eps0 sqrt(2 e/m) x 1e6 / 2
  function tbFromInputs(P, rw, Cc, theta0, n) {
    n = n || 4000;
    var th = theta0 * DEG2RAD, rc = Math.sqrt(Cc) * rw, Rc = rc / Math.sin(th);
    var a = Math.sqrt(CONE * (1 - Math.cos(th)) / P), y = CL.yFromA(a), Ra = Rc * Math.exp(-y);
    var B = upToApex(toGun(bfeS(n), Rc, th, 0)), An = upToApex(toGun(anodeS(y, n), Rc, th, 0));
    return { rc: rc, Rc: Rc, Ra: Ra, y: y, alpha: a, bfe: B, anode: An,
             angleToEdge: PIERCE / DEG2RAD, angleToAxis: PIERCE / DEG2RAD - theta0,
             bfeTip: B[B.length - 1], anodeTop: An[An.length - 1], anodeOnAxisZ: Rc - Ra };
  }

  var api = { tiwaryBasuCurves: tiwaryBasuCurves, tbFromInputs: tbFromInputs, tiwaryBasuRows: tiwaryBasuRows,
              vedantRows: vedantRows, deckText: deckText, chooseGrid: chooseGrid,
              pierceDeck: pierceDeck, bfeS: bfeS, anodeS: anodeS };
  root.EGUN = root.EGUN || {};
  root.EGUN.pierce = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
