/*
 * The live cross-section of the gun, mirroring Fig. 1 of the paper.
 *
 * Axially symmetric, so the drawing is a meridional slice: the axis runs left
 * to right, and everything is mirrored about it. Drawn to scale in millimetres
 * within the viewport, with the scale bar stating what the scale is — the point
 * is that dragging a slider visibly changes the shape of the gun, which is hard
 * to get from a table of numbers.
 *
 * Labelled: theta (the cone half-angle at the cathode), r_c and R_c (eqs 2 and
 * 4), the anode aperture, and the beam waist r_w.
 *
 * Nothing here relies on colour to carry meaning: every element is also named
 * by a text label, and the beam is distinguished by fill as well as hue.
 */
(function (root) {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var DEG = Math.PI / 180;

  var COL = {
    axis: "#4a5162",
    metal: "#8b94a8",
    metalFill: "#2a2f3b",
    beam: "#5aa2ff",
    beamFill: "rgba(90, 162, 255, 0.16)",
    ink: "#a7b0c0",
    dim: "#6f7889",
    accent: "#5aa2ff",
  };

  function el(name, attrs, text) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) {
      if (Object.prototype.hasOwnProperty.call(attrs, k)) {
        n.setAttribute(k, attrs[k]);
      }
    }
    if (text != null) n.textContent = text;
    return n;
  }

  /*
   * Draw. `geo` is the object from physics.geometry(); `host` is the <svg>.
   *
   * The gun is laid out along z with the cathode pole at z = 0, so the drawing
   * spans z in [0, zWaist] and r in [-rc, +rc]. Both are fitted into the
   * viewBox with a single isotropic scale, so angles on screen are true angles
   * -- a theta of 70 degrees has to LOOK like 70 degrees or the drawing is
   * lying about the one quantity it exists to show.
   */
  function render(host, geo) {
    while (host.firstChild) host.removeChild(host.firstChild);

    var W = 680, H = 300;
    host.setAttribute("viewBox", "0 0 " + W + " " + H);
    host.setAttribute("role", "img");

    if (!geo || !geo.ok) {
      var why = (geo && geo.reason) || "these inputs do not define a gun.";
      host.setAttribute("aria-label", "Gun geometry undefined: " + why);
      host.appendChild(el("text", {
        x: W / 2, y: H / 2 - 12, fill: COL.ink, "font-size": 13,
        "text-anchor": "middle", "font-family": "system-ui, sans-serif",
      }, "No gun geometry for these inputs"));
      // Wrap the reason by hand; SVG has no text flow.
      wrapText(host, why, W / 2, H / 2 + 12, 78, 15);
      return;
    }

    var padL = 74, padR = 74, padT = 34, padB = 40;
    var zSpan = Math.max(geo.zWaist, 1e-6);
    var rSpan = Math.max(geo.rc, geo.ra, geo.rw) * 1.06;

    var sx = (W - padL - padR) / zSpan;
    var sy = (H - padT - padB) / (2 * rSpan);
    var s = Math.min(sx, sy);                 // isotropic: angles stay true

    var x0 = padL;
    var yMid = padT + (H - padT - padB) / 2;
    var X = function (z) { return x0 + z * s; };
    var Y = function (r) { return yMid - r * s; };

    host.setAttribute("aria-label",
      "Cross-section of a Pierce gun with cone half-angle " +
      geo.thetaDeg.toFixed(1) + " degrees, cathode radius " +
      geo.rc.toFixed(2) + " millimetres and beam waist " +
      geo.rw.toFixed(2) + " millimetres.");

    /* --- axis --- */
    host.appendChild(el("line", {
      x1: X(-0.03 * zSpan), y1: yMid, x2: X(1.05 * zSpan), y2: yMid,
      stroke: COL.axis, "stroke-width": 1, "stroke-dasharray": "5 4",
    }));

    /*
     * Cathode: a spherical cap of radius R_c whose centre of curvature is the
     * convergence point at z = R_c on the axis. Its rim sits at +/- r_c, and it
     * is dished back from the rim by the sagitta.
     */
    var cx = geo.Rc;                       // centre of curvature, on the axis
    var rimZ = geo.sagitta;                // rim plane
    var arcR = geo.Rc * s;
    var cathode = "M " + X(rimZ) + " " + Y(geo.rc) +
                  " A " + arcR + " " + arcR + " 0 0 1 " + X(rimZ) + " " + Y(-geo.rc);
    host.appendChild(el("path", {
      d: cathode, fill: "none", stroke: COL.metal, "stroke-width": 3.5,
      "stroke-linecap": "round",
    }));

    /* Focus electrode stubs, the Pierce part: short flanks off the cathode rim
       at the Pierce angle. Indicative only, so they are drawn thin. */
    [1, -1].forEach(function (sgn) {
      host.appendChild(el("line", {
        x1: X(rimZ), y1: Y(sgn * geo.rc),
        x2: X(rimZ - 0.06 * zSpan), y2: Y(sgn * geo.rc * 1.16),
        stroke: COL.metal, "stroke-width": 2, opacity: 0.55,
      }));
    });

    /* --- the converging cone edge, cathode rim to anode aperture --- */
    [1, -1].forEach(function (sgn) {
      host.appendChild(el("line", {
        x1: X(rimZ), y1: Y(sgn * geo.rc),
        x2: X(geo.zAnode), y2: Y(sgn * geo.ra),
        stroke: COL.beam, "stroke-width": 1.4, opacity: 0.75,
      }));
    });

    /*
     * Beam envelope. Two regions: the straight convergent cone from cathode to
     * anode, then the space-charge-limited drift from the anode to the waist,
     * integrated in physics.js. Filled as one polygon so it reads as a beam.
     */
    var top = ["M " + X(rimZ) + " " + Y(geo.rc)];
    top.push("L " + X(geo.zAnode) + " " + Y(geo.ra));
    geo.drift.points.slice().reverse().forEach(function (p) {
      top.push("L " + X(geo.zAnode + (geo.drift.length - p.z)) + " " + Y(p.r));
    });
    var bottomPts = geo.drift.points.map(function (p) {
      return "L " + X(geo.zAnode + (geo.drift.length - p.z)) + " " + Y(-p.r);
    });
    var d = top.join(" ") + " " + bottomPts.join(" ") +
            " L " + X(geo.zAnode) + " " + Y(-geo.ra) +
            " L " + X(rimZ) + " " + Y(-geo.rc) + " Z";
    host.appendChild(el("path", {
      d: d, fill: COL.beamFill, stroke: COL.beam, "stroke-width": 1.4,
      "stroke-opacity": 0.8,
    }));

    /* --- anode: an apertured plate at z_anode --- */
    var plateW = Math.max(4, 0.012 * zSpan * s);
    [1, -1].forEach(function (sgn) {
      var inner = Y(sgn * geo.ra);
      var outer = Y(sgn * rSpan * 1.02);
      host.appendChild(el("rect", {
        x: X(geo.zAnode) - plateW / 2,
        y: Math.min(inner, outer),
        width: plateW,
        height: Math.abs(outer - inner),
        fill: COL.metalFill, stroke: COL.metal, "stroke-width": 1.6,
      }));
    });

    /* --- the waist --- */
    host.appendChild(el("line", {
      x1: X(geo.zWaist), y1: Y(geo.rw), x2: X(geo.zWaist), y2: Y(-geo.rw),
      stroke: COL.accent, "stroke-width": 2,
    }));

    /* ---------------- labels ---------------- */
    function label(x, y, text, opts) {
      opts = opts || {};
      host.appendChild(el("text", {
        x: x, y: y,
        fill: opts.fill || COL.ink,
        "font-size": opts.size || 11,
        "font-family": "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        "text-anchor": opts.anchor || "middle",
      }, text));
    }

    /* theta, drawn as an actual arc at the cathode so the angle is visible */
    var apexX = X(cx), apexY = yMid;
    var armLen = Math.min(geo.Rc * 0.42, zSpan * 0.3) * s;
    var th = geo.thetaDeg * DEG;
    host.appendChild(el("path", {
      d: "M " + apexX + " " + apexY +
         " L " + (apexX - armLen * Math.cos(th)) + " " + (apexY - armLen * Math.sin(th)),
      stroke: COL.dim, "stroke-width": 1, "stroke-dasharray": "3 3", fill: "none",
    }));
    var arcRad = armLen * 0.42;
    host.appendChild(el("path", {
      d: "M " + (apexX - arcRad) + " " + apexY +
         " A " + arcRad + " " + arcRad + " 0 0 1 " +
         (apexX - arcRad * Math.cos(th)) + " " + (apexY - arcRad * Math.sin(th)),
      stroke: COL.accent, "stroke-width": 1.4, fill: "none",
    }));
    label(apexX - arcRad * 1.15 * Math.cos(th / 2) ,
          apexY - arcRad * 1.15 * Math.sin(th / 2) - 4,
          "θ = " + geo.thetaDeg.toFixed(1) + "°",
          { fill: COL.accent, anchor: "end", size: 12 });

    /* r_c with a dimension tick at the cathode rim */
    host.appendChild(el("line", {
      x1: X(rimZ) - 12, y1: Y(geo.rc), x2: X(rimZ) - 12, y2: yMid,
      stroke: COL.dim, "stroke-width": 1,
    }));
    label(X(rimZ) - 16, (Y(geo.rc) + yMid) / 2, "r_c " + geo.rc.toFixed(2),
          { anchor: "end" });

    label(X(rimZ) + 6, Y(-geo.rc) + 20, "cathode", { anchor: "start", fill: COL.dim });
    label(X(rimZ) - 16, Y(-geo.rc) + 20, "R_c " + geo.Rc.toFixed(2) + " mm",
          { anchor: "end", fill: COL.dim });

    label(X(geo.zAnode), Y(-rSpan) + 18, "anode", { fill: COL.dim });
    label(X(geo.zAnode), padT - 12, "r_a " + geo.ra.toFixed(2));

    /* r_w at the waist */
    host.appendChild(el("line", {
      x1: X(geo.zWaist) + 12, y1: Y(geo.rw), x2: X(geo.zWaist) + 12, y2: yMid,
      stroke: COL.dim, "stroke-width": 1,
    }));
    label(X(geo.zWaist) + 16, (Y(geo.rw) + yMid) / 2 + 4,
          "r_w " + geo.rw.toFixed(2), { anchor: "start", fill: COL.accent });
    label(X(geo.zWaist), Y(-rSpan) + 18, "waist", { fill: COL.accent });

    /* scale bar: the drawing claims to be to scale, so it has to say the scale */
    var barMm = niceStep(zSpan / 4);
    var barPx = barMm * s;
    var bx = W - padR - barPx, by = H - 14;
    host.appendChild(el("line", {
      x1: bx, y1: by, x2: bx + barPx, y2: by, stroke: COL.dim, "stroke-width": 1.5,
    }));
    [bx, bx + barPx].forEach(function (x) {
      host.appendChild(el("line", {
        x1: x, y1: by - 3, x2: x, y2: by + 3, stroke: COL.dim, "stroke-width": 1.5,
      }));
    });
    label(bx + barPx / 2, by - 6, barMm + " mm", { fill: COL.dim, size: 10 });
  }

  /* SVG text does not wrap, so lines are broken at a character budget. */
  function wrapText(host, text, x, y, chars, lineHeight) {
    var words = String(text).split(/\s+/), line = "", lines = [];
    words.forEach(function (w) {
      if ((line + " " + w).trim().length > chars) { lines.push(line.trim()); line = w; }
      else { line += " " + w; }
    });
    if (line.trim()) lines.push(line.trim());
    lines.slice(0, 5).forEach(function (l, i) {
      host.appendChild(el("text", {
        x: x, y: y + i * lineHeight, fill: COL.dim, "font-size": 11,
        "text-anchor": "middle", "font-family": "system-ui, sans-serif",
      }, l));
    });
  }

  /* 1, 2, 5, 10, 20, 50 ... so the bar is a number a person can read off */
  function niceStep(v) {
    var p = Math.pow(10, Math.floor(Math.log10(Math.max(v, 1e-6))));
    var n = v / p;
    var m = n >= 5 ? 5 : n >= 2 ? 2 : 1;
    return m * p;
  }

  var api = { render: render };
  root.EGUN = root.EGUN || {};
  root.EGUN.gun = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
