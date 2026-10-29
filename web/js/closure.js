/*
 * LBClosure in the browser: the emulator of Vedant's exact Pierce-gun synthesis,
 * his EGUN deck construction, and the Stage-2 calibration.
 *
 * A line-for-line port of src/physics/closure.py, src/physics/kf.py, and the
 * default path of get_design_params_mode1 / compute_gun_points_mm /
 * print_gun_points in src/physics/vendor/pierce_gun_design_fixed.py.
 * tests/test_design_js_parity.py runs this file under Node against the Python:
 * synthesis to < 1e-9 relative, deck text character for character.
 *
 * Two angles are kept distinct: theta (raw, the analysis angle, which every
 * length is computed from) and thetaT (aberration-corrected, the angle the
 * cathode is built with and Panahi Table 2 reports).
 */
(function (root) {
  "use strict";

  var IN_TO_MM = 25.4;
  var DEG2RAD = Math.PI / 180, RAD2DEG = 180 / Math.PI;   // CPython's radians/degrees
  var LO = 0.05 * DEG2RAD, HI = 89.0 * DEG2RAD;
  var N_HI = 400, N_SCAN = 300, N_BISECT = 64;
  var CONE = 14.67, A_MAX = 9.6, LENS = 0.417, ABER = 0.905, C174 = 0.17409;
  var ALPHA_NEWTON_OK = 7.9;

  // ---- Langmuir-Blodgett series (Vaughan 1981 Eq. 5, 6, 10) ----------------
  function aFromY(y) {
    return y + y * y * (0.3 + y * (0.075 + y * (0.01432 + y * (0.00216 + y * 0.00035))));
  }
  function daDy(y) {
    return 1 + y * (0.6 + y * (0.225 + y * (0.05728 + y * (0.0108 + y * 0.00162))));
  }
  function pPoly(y) {
    return 1 + y * (0.6 + y * (0.225 + y * (0.0573 + y * (0.0108 + y * 0.0021))));
  }
  function yFromA(a) {
    var y = a - 0.275 * (a * a) + 0.06 * Math.pow(a, 3) - 0.006 * Math.pow(a, 4);
    for (var i = 0; i < 20; i++) y = y - (aFromY(y) - a) / daDy(y);
    return y;
  }

  // ---- kf table, 4-point Lagrange in (ln V, y) -----------------------------
  function lagrange4(t) {
    return [-t * (t - 1) * (t - 2) / 6.0, (t + 1) * (t - 1) * (t - 2) / 2.0,
            -(t + 1) * t * (t - 2) / 2.0, (t + 1) * t * (t - 1) / 6.0];
  }
  function stencil(x, x0, x1, n) {
    var h = (x1 - x0) / (n - 1), s = (x - x0) / h;
    var i = Math.min(Math.max(Math.floor(s), 1), n - 3);
    return [i, s - i];
  }
  function kf(V, y, T) {
    if (V < 500 * (1 - 1e-12) || V > 600e3 * (1 + 1e-12)) {
      throw new Error("kf table covers V in [500, 600000] V");
    }
    y = Math.min(Math.max(y, T.y_min), T.y_max);
    var su = stencil(Math.log(V), T.u_min, T.u_max, T.nu);
    var sy = stencil(y, T.y_min, T.y_max, T.ny);
    var wu = lagrange4(su[1]), wy = lagrange4(sy[1]), out = 0;
    for (var a = 0; a < 4; a++)
      for (var b = 0; b < 4; b++)
        out = out + wu[a] * wy[b] * T.kf[su[0] - 1 + a][sy[0] - 1 + b];
    return out;
  }

  // ---- the theta solve ------------------------------------------------------
  function state(t, rc, rw, muP, k) {
    var a = Math.sqrt(CONE * k * (1.0 - Math.cos(t)) / muP);
    var y = yFromA(a);
    return [a, y, rc * Math.exp(-y) / rw];
  }
  function residual(t, rc, rw, muP, k) {
    var s = state(t, rc, rw, muP, k);
    var tp1 = Math.sin(t) * (1.0 - LENS * pPoly(s[1]) / s[0]);
    var tp2 = C174 * Math.sqrt(muP * Math.abs(Math.log(s[2])));
    return Math.abs(tp2) - Math.abs(tp1);
  }
  function solveTheta(rc, rw, muP, k) {
    var first = -1, i;
    for (i = 1; i <= N_HI; i++) {
      var t = LO + (HI - LO) * i / N_HI;
      if (Math.sqrt(CONE * k * (1.0 - Math.cos(t)) / muP) >= A_MAX) { first = i - 1; break; }
    }
    var thHi = first < 0 ? HI : (first === 0 ? LO : LO + (HI - LO) * first / N_HI);
    if (thHi <= LO) return { theta: NaN, why: 1 };
    var ts = [], fs = [];
    for (i = 0; i <= N_SCAN; i++) {
      ts.push(LO + (thHi - LO) * i / N_SCAN);
      fs.push(residual(ts[i], rc, rw, muP, k));
    }
    var roots = [];
    for (i = 0; i < N_SCAN; i++) {
      var fa = fs[i], fb = fs[i + 1];
      if (fa === 0) { roots.push(ts[i]); continue; }
      if (fa * fb < 0) {
        var a_ = ts[i], b_ = ts[i + 1];
        for (var j = 0; j < N_BISECT; j++) {
          var mid = 0.5 * (a_ + b_), fm = residual(mid, rc, rw, muP, k);
          if (fa * fm <= 0) b_ = mid; else { a_ = mid; fa = fm; }
        }
        roots.push(0.5 * (a_ + b_));
      }
    }
    if (!roots.length) return { theta: NaN, why: 2 };
    var best = -Infinity;
    roots.forEach(function (r) { if (state(r, rc, rw, muP, k)[2] > 1 && r > best) best = r; });
    return isFinite(best) ? { theta: best, why: 0 } : { theta: NaN, why: 3 };
  }

  var REASONS = {
    1: "empty theta window (series limit at the first step)",
    2: "no theta satisfies Vaughan Eq.(10) = Eq.(14): outside the Pierce regime",
    3: "every root leaves R <= 1: no downstream waist",
    4: "rc <= rw: the cathode is no larger than the waist (raise I or lower Jc or rw)",
  };

  /* One design. V volts, I amps, rw mm, Jc A/cm^2. Lengths out in mm. */
  function synthesize(V, I, rwMm, Jc, T, apertureFactor) {
    var af = apertureFactor === undefined ? 1.2 : apertureFactor;
    var rc = Math.sqrt(I / (Math.PI * Jc)) / 2.54, rw = rwMm / IN_TO_MM;
    var muP = 1e6 * I / Math.pow(V, 1.5);
    var out = { V: V, I: I, rw_mm: rwMm, Jc: Jc, muP: muP, C: (rc / rw) * (rc / rw),
                rc_mm: rc * IN_TO_MM, feasible: false };
    if (!(rc > rw)) { out.why = 4; out.reason = REASONS[4]; return out; }
    var k = 1, s;
    for (var it = 0; it < 10; it++) {
      s = solveTheta(rc, rw, muP, k);
      var good = s.why === 0;
      var y0 = state(good ? s.theta : LO, rc, rw, muP, k)[1];
      var kn = good ? kf(V, y0, T) : k;
      var done = Math.abs(kn - k) < 1e-12;
      k = kn;
      if (done) break;
    }
    s = solveTheta(rc, rw, muP, k);
    if (s.why !== 0) { out.why = s.why; out.reason = REASONS[s.why]; return out; }
    var t = s.theta, st = state(t, rc, rw, muP, k), a = st[0], y = st[1], R = st[2];
    var Rc = rc / Math.sin(t), Ra = Rc * Math.exp(-y), rbza = rc * Math.exp(-y);
    var ra = af * rbza, zac = Rc - Ra, disc = Ra * Ra - ra * ra;
    var zaValid = disc > 0, za = zaValid ? Rc - Math.sqrt(disc) : Rc;
    var Rm1 = R - 1, Rp = Rm1 > 0 ? Rm1 : 0;
    var Z = 11.0 * Math.sqrt(Rp) + 1.32 * Rp + 0.0615 * (Rp * Rp) - 0.00167 * Math.pow(Rp, 3);
    var zw = Rm1 > 0 ? zac + rw * Z / Math.sqrt(muP) : zac;
    var sc = Math.sin(t) / ABER, corr = sc <= 1;
    var thT = corr ? Math.asin(Math.min(sc, 1)) : t, RcT = corr ? ABER * Rc : Rc;
    var mm = function (v) { return v * IN_TO_MM; };
    Object.assign(out, {
      feasible: true, why: 0, reason: "",
      theta: t * RAD2DEG, thetaT: thT * RAD2DEG,
      Rc_mm: mm(Rc), Ra_mm: mm(Ra), zac_mm: mm(zac), za_mm: mm(za), rb_za_mm: mm(rbza),
      ra_mm: mm(ra), zw_mm: mm(zw), RcT_mm: mm(RcT), R: R, y: y, alpha_neg: a,
      rel_factor: k, za_valid: zaValid, correction_applied: corr,
      alpha_ok: a < ALPHA_NEWTON_OK, E_kV_per_mm: V / 1000 / mm(zac),
    });
    return out;
  }

  /* Stage-2 calibration, theta_meas ~ f * (1 + b0 + w.z(P, ln rw, ln C)). */
  function calibrate(p, f, P, rw, C) {
    if (p.type === "scalar") return p.k * f;
    var x = [P, Math.log(rw), Math.log(C)], s = p.coef[0];
    for (var i = 0; i < 3; i++) s += p.coef[i + 1] * ((x[i] - p.mu[i]) / p.sd[i]);
    return f * (1 + s);
  }

  // ---- Python-compatible number formatting for the deck ---------------------
  function roundHalfEven(x) {                 // Python round(x) -> int
    var r = Math.round(x), diff = Math.abs(x - Math.trunc(x));
    if (diff === 0.5) r = 2 * Math.round(x / 2);
    return r;
  }
  function roundTo(x, n) { return Number(x.toFixed(n)); }   // Python round(x, n)
  function snap(v, h) { return roundTo(roundHalfEven(v / h) * h, 10); }
  function fmt(v, n) { return v.toFixed(n === undefined ? 2 : n); }
  function padL(s, w) { s = String(s); while (s.length < w) s += " "; return s; }
  function padR(s, w) { s = String(s); while (s.length < w) s = " " + s; return s; }

  function cross(p1, p2, q1, q2) {
    function o(a, b, c) { return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]); }
    var d1 = o(q1, q2, p1), d2 = o(q1, q2, p2), d3 = o(p1, p2, q1), d4 = o(p1, p2, q2);
    return (d1 * d2 < 0) && (d3 * d4 < 0);
  }

  /* Vedant's get_design_params_mode1 on its default path (no prompts). r holds
     the synthesis result with lengths in mm. Throws an Error with his reason when
     a default is not admissible -- the same cases his code refuses. */
  function designParams(r) {
    var Rc = r.Rc_mm, Ra = r.Ra_mm, rc = r.rc_mm, ra = r.ra_mm, za = r.za_mm, zw = r.zw_mm;
    var zac = Rc - Ra, thR = r.theta * DEG2RAD, thTR = r.thetaT * DEG2RAD, RcT = r.RcT_mm;
    var Dz = RcT * (1 - Math.cos(thTR)) - Rc * (1 - Math.cos(thR));
    var Hz = 0.05, Hr = 0.05, d = 0.2, l = 0.0;
    var sinT = Math.sin(thTR), thDeg = thTR * RAD2DEG;
    var k1max = roundTo(0.99 - sinT, 4), k1 = 0.02;
    if (k1 > k1max) throw new Error("k1 = 0.02 exceeds its limit " + k1max);
    var sinD = sinT + k1, delta = Math.asin(sinD);
    var psiLo = thDeg + 1.0, psiHi = Math.min(89.5, 90.0 + thDeg - delta * RAD2DEG - 0.5);
    if (psiLo >= psiHi) throw new Error("k1 leaves no valid electrode angle");
    var psi = 67.5;
    if (!(psiLo <= psi && psi <= psiHi)) {
      throw new Error(psi < psiLo
        ? "theta_T = " + thDeg.toFixed(2) + " deg is too large for a 67.5 deg focus electrode"
        : "k1 starts the electrode too far round the cathode sphere");
    }
    var slope = Math.tan(psi * DEG2RAD - thTR);
    var z5 = RcT * (1 - Math.cos(delta)), r5 = RcT * sinD, H2 = RcT * sinT;
    var r6 = 1.5 * H2, Rmax = 1.75 * H2;
    if (r6 <= r5) throw new Error("the electrode would end below its own start");
    var z6 = z5 + (r6 - r5) / slope, zmaxRel = 1.3 * zw;
    var betaLo = ra < Ra ? Math.asin(ra / Ra) * RAD2DEG + 1.0 : 10.0, betaHi = 89.9;
    if (!(Ra * 0.98 < Rmax)) betaHi = Math.min(betaHi, Math.asin(0.98 * Rmax / Ra) * RAD2DEG);
    if (betaLo >= betaHi) throw new Error("the anode sphere reaches above the outer wall");
    var z9 = function (b) { return Dz + Rc - Ra * Math.cos(b * DEG2RAD); };
    var betaFor = function (g) {
      var c = (Dz + Rc - (z6 + g)) / Ra;
      return Math.acos(Math.max(-1.0, Math.min(1.0, c))) * RAD2DEG;
    };
    var electrode = [[[z5, r5], [z6, r6]], [[z6, r6], [z6, Rmax]]];
    var faceOk = function (g) {
      var b = betaFor(g) * DEG2RAD, P9 = [z6 + g, Ra * Math.sin(b)];
      return !electrode.some(function (s) { return cross(s[0], s[1], P9, [Dz + za, ra]); });
    };
    var gapLo = Math.max(z9(betaLo) - z6, 1e-3);
    var gapHi = Math.min(z9(betaHi) - z6, zmaxRel - z6 - 1e-3);
    if (gapHi <= gapLo) throw new Error("the electrode tip is too far downstream for pt 9");
    var step = (gapHi - gapLo) / 400;
    while (gapLo < gapHi && !faceOk(gapLo)) gapLo += step;
    if (gapLo >= gapHi) throw new Error("no pts 7-8 gap keeps the anode face clear");
    var alpha = roundTo(zac, 4);
    if (!(gapLo <= alpha && alpha <= gapHi)) alpha = roundTo(gapLo + 0.5 * (gapHi - gapLo), 4);
    if (alpha < gapLo || alpha > gapHi) throw new Error("default gap outside its range");
    var beta = betaFor(alpha), betaRad = beta * DEG2RAD, r9 = Ra * Math.sin(betaRad);
    var gapMax = Math.floor((r9 - ra) / Hr - 1e-9);
    if (gapMax < 1) throw new Error("pt 9 is less than one grid step above the aperture");
    var nH = Math.min(gapMax, Math.max(1, roundHalfEven(0.5 * (r9 - ra) / Hr)));
    var r10 = ra + nH * Hr, P10 = [Dz + za, r10], P11 = [Dz + za, ra], P9 = [z6 + alpha, r9];
    if (electrode.some(function (s) {
      return cross(s[0], s[1], P9, P10) || cross(s[0], s[1], P10, P11);
    })) throw new Error("the anode face through pt 10 cuts the focus electrode");
    var tanPhi2 = 0.17409 * Math.sqrt(r.muP * Math.abs(Math.log(r.R)));
    var r12 = ra - (Rc - za) * tanPhi2;
    if (r12 <= 0) throw new Error("the drift-tube wall reaches the axis before pt 12");
    var rOthers = Math.max(r6, rc + k1 * RcT, r9, r10);
    if (Rmax <= rOthers) throw new Error("Rmax is not above every other point");
    return { d: d, l: l, k1: k1, psi_deg: psi, nH_gap: nH, r6: r6, Dz: Dz, r10: r10,
             r12: r12, alpha: alpha, beta_deg: beta, beta_rad: betaRad, Rmax: Rmax,
             Zmax: d + 1.3 * zw, H_z: Hz, H_r: Hr };
  }

  /* compute_gun_points_mm: the 15 points [id, label, z, r]. */
  function gunPoints(r, dp) {
    var Rc = r.Rc_mm, Ra = r.Ra_mm, rc = r.rc_mm, ra = r.ra_mm, za = r.za_mm;
    var th = r.thetaT * DEG2RAD, RcT = r.RcT_mm, Dz = dp.Dz, d = dp.d, l = dp.l, k1 = dp.k1;
    var sT = Math.sin(th), cT = Math.cos(th), sD = sT + k1;
    if (sD >= 1.0) throw new Error("sin(theta_T) + k1 >= 1");
    var delta = Math.asin(sD), cD = Math.cos(delta);
    var sB = Math.sin(dp.beta_rad);
    var slope = Math.tan(dp.psi_deg * DEG2RAD - th);
    var z5 = d + RcT * (1 - cD), r5 = RcT * sD, z6 = z5 + (dp.r6 - r5) / slope;
    var z7 = z6, z8 = z7 + dp.alpha, z11 = d + Dz + za;
    return [
      [0, "True origin", 0.0, 0.0],
      [1, "Cathode vertex", d, 0.0],
      [2, "Cathode rim", d + RcT * (1 - cT), RcT * sT],
      [3, "Cathode body rear-inner", d - l, rc],
      [4, "Cathode body rear-outer", d - l, rc + k1 * RcT],
      [5, "Focus elec. base", z5, r5],
      [6, "Focus elec. tip", z6, dp.r6],
      [7, "Focus elec. outer tip", z7, dp.Rmax],
      [8, "Drift-tube front outer", z8, dp.Rmax],
      [9, "Anode sphere rim", z8, Ra * sB],
      [10, "Anode face above aperture", z11, ra + dp.nH_gap * dp.H_r],
      [11, "Anode aperture edge", z11, ra],
      [12, "Drift-tube inner start", d + Dz + Rc, dp.r12],
      [13, "Drift-tube inner end", dp.Zmax, dp.r12],
      [14, "Drift-tube back, on axis", dp.Zmax, 0.0],
    ];
  }

  /* print_gun_points' text file, character for character. */
  function deckText(r, dp, pts) {
    var Hz = dp.H_z, Hr = dp.H_r, RcT = r.RcT_mm, arcZ = dp.d + RcT;
    var N = pts.filter(function (p) { return p[0] !== 0; }).length;
    var maxtr = Math.max(Math.ceil(RcT / Hr / 50.0) * 50, 50);
    var L = [" 'PIERCE GUN TEST'", " &BOUND",
      " H =   " + Hz + "," + Hr + ", MBP = " + N + ", NT = " + N +
        ", NGRD = 0, jMAG = 0, NMAG = 0, MAGORD = 2,",
      " /", " &CONTROL",
      " IOPT = 2, ICLOS = 0, IOS = 0, IORDR=1, IPONT = 5, NMAX = 1000, EL =   0.0000050, ",
      " maxtr=" + maxtr + ", lmax=20,LC=1,eqstep=0.1,1.,10.,",
      " /", " &PRINTS", " IPOT = 0, IFLDZ = 2, IFLDR = 2, ISPC = 1, IRAY = 2,", " /",
      " &PLOT", " ITRPL = 2, IPOTPL = 2, ICDPL = 2, ISCDPL = 2,", " / "];
    pts.forEach(function (p) {
      var pid = p[0], z = p[2], rv = p[3];
      if (pid === 0) return;
      var code = 1, zc = 0.0, rc_ = 0.0, rad = 0.0;
      if (pid === 1) { code = 0; zc = arcZ; rad = RcT; }
      else if (pid === 3 || pid === 7) code = -1;
      var volt = pid <= 7 ? 0.0 : r.V;
      var zOut = pid === 2 ? z.toFixed(8) : fmt(snap(z, Hz));
      L.push(padR(code, 2) + ",  " + padL(zOut, 16) + "  " + padL(fmt(snap(rv, Hr)), 8) +
             "  " + padL(fmt(snap(volt, Hz), 4), 12) + "  " + padL(fmt(snap(zc, Hz), 6), 12) +
             "  " + padL(fmt(snap(rc_, Hr), 5), 10) + "  " + fmt(snap(rad, Hr), 4));
    });
    var DZ = roundTo(Hz * 3.0, 6), DX = roundTo(Hr * 2.0, 6);
    L.push(" &GRID", "  /",
      " &CATHODE CSHAPE= \"SPHERE\", EMISSION=\"CHILD\", NSEG = 1, OCAT = " +
        fmt(snap(dp.d, Hz)) + ",",
      " CRAD = " + fmt(snap(RcT, Hr)) + ", RMAX = " + fmt(snap(r.rc_mm, Hr)) +
        " ,DZ = " + DZ + ", DX =  " + DX);
    return L.join("\n") + "\n";
  }

  /* Deck for a synthesised design, or {refused: reason}. */
  function deck(r) {
    if (!r.feasible) return { refused: "design is infeasible: " + r.reason };
    if (!r.za_valid) {
      return { refused: "theta_raw = " + r.theta.toFixed(3) + " deg > 56.44 deg: Vaughan " +
        "Eq.(20) has no real root for the anode plane, so the deck's aperture point would " +
        "be invented. Sharma (2001) would fix this and is not implemented." };
    }
    try {
      var dp = designParams(r), pts = gunPoints(r, dp);
      return { text: deckText(r, dp, pts), points: pts, params: dp };
    } catch (e) {
      return { refused: e.message };
    }
  }

  var api = { synthesize: synthesize, kf: kf, yFromA: yFromA, calibrate: calibrate,
              designParams: designParams, gunPoints: gunPoints, deckText: deckText,
              deck: deck, REASONS: REASONS };
  root.EGUN = root.EGUN || {};
  root.EGUN.closure = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
