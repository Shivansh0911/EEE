/*
 * The forward pass, in the browser.
 *
 * This is the whole "model" at runtime: 23 numbers and three matrix
 * multiplications. It runs in microseconds, which is why the Predict tab never
 * calls a backend and works with the network unplugged.
 *
 * Everything here is driven by the exported model JSON -- architecture,
 * activations, the input transform, the scaler ranges. Nothing about the
 * network is hardcoded, because a browser that assumed the wrong shape would
 * produce a plausible-looking wrong angle rather than an error.
 *
 * tests/test_js_python_parity.py runs this file under Node and asserts it
 * matches the Python implementation to < 1e-9 on 100 random inputs. If that
 * test fails, this file and the paper disagree, and the site is wrong.
 *
 * Loaded as a classic script (see web/README.md for why, not ESM): it attaches
 * to window.EGUN.ann, and also exports for Node so the parity test can require
 * it without a build step.
 */
(function (root) {
  "use strict";

  var TRANSFORMS = {
    identity: function (v) { return v; },
    log: Math.log,
    log10: Math.log10,
  };

  var ACTIVATIONS = {
    tanh: Math.tanh,
    linear: function (z) { return z; },
  };

  /* Physical units -> transformed units. Must run BEFORE scaling: the scaler's
   * x_min/x_max are stored in transformed space. */
  function applyInputTransform(model, x) {
    var out = x.slice();
    var spec = model.input_transform || {};
    for (var name in spec) {
      if (!Object.prototype.hasOwnProperty.call(spec, name)) continue;
      var j = model.inputs.indexOf(name);
      if (j < 0) throw new Error("input_transform names unknown input " + name);
      var fn = TRANSFORMS[spec[name]];
      if (!fn) throw new Error("unknown transform " + spec[name]);
      out[j] = fn(out[j]);
    }
    return out;
  }

  /* Min-max to [-1, 1], using the ranges fitted on the training rows only. */
  function normaliseInputs(model, xTransformed) {
    var lo = model.scaler.x_min, hi = model.scaler.x_max;
    return xTransformed.map(function (v, j) {
      return (2 * (v - lo[j])) / (hi[j] - lo[j]) - 1;
    });
  }

  function denormaliseOutputs(model, yn) {
    var lo = model.scaler.y_min, hi = model.scaler.y_max;
    return yn.map(function (v, k) {
      return ((v + 1) / 2) * (hi[k] - lo[k]) + lo[k];
    });
  }

  /* One layer: a @ W + b, then the activation. W is (n_in, n_out). */
  function layer(a, W, b, act) {
    var nOut = b.length, out = new Array(nOut);
    for (var n = 0; n < nOut; n++) {
      var s = b[n];
      for (var m = 0; m < a.length; m++) s += a[m] * W[m][n];
      out[n] = act(s);
    }
    return out;
  }

  /* Normalised inputs -> normalised outputs. The parity test targets this. */
  function forwardNormalised(model, xn) {
    var act = ACTIVATIONS[model.activation];
    var outAct = ACTIVATIONS[model.output_activation];
    if (!act || !outAct) throw new Error("unknown activation in model");

    var nLayers = model.architecture.length - 1;
    var a = xn;
    for (var i = 1; i <= nLayers; i++) {
      a = layer(a, model.weights["W" + i], model.weights["b" + i],
                i === nLayers ? outAct : act);
    }
    return a;
  }

  /*
   * The full path a UI needs: physical inputs in, physical outputs out, plus
   * the two things the reader has to be told about.
   *
   * `outOfEnvelope` lists inputs outside the range the model was trained on.
   * `saturated` lists outputs whose NORMALISED value has run past the model's
   * saturation limit -- a tanh output over min-max scaled targets asymptotes at
   * +/-1, which maps back to the edge of the training range, so a prediction
   * there is the network against its ceiling rather than an extrapolation. The
   * paper's own worst test case is exactly this failure.
   */
  function predict(model, x) {
    var xn = normaliseInputs(model, applyInputTransform(model, x));
    var yn = forwardNormalised(model, xn);
    var y = denormaliseOutputs(model, yn);

    var limit = model.saturation_limit || 0.98;
    var saturated = [];
    yn.forEach(function (v, k) {
      if (Math.abs(v) > limit) {
        saturated.push({ target: model.targets[k], normalised: v });
      }
    });

    var outOfEnvelope = [];
    model.inputs.forEach(function (name, j) {
      var env = model.training_envelope[name];
      if (!env) return;
      if (x[j] < env[0]) {
        outOfEnvelope.push({ input: name, value: x[j], bound: env[0],
                             side: "below", by: env[0] - x[j] });
      } else if (x[j] > env[1]) {
        outOfEnvelope.push({ input: name, value: x[j], bound: env[1],
                             side: "above", by: x[j] - env[1] });
      }
    });

    var out = {};
    model.targets.forEach(function (name, k) { out[name] = y[k]; });
    return { values: out, vector: y, normalised: yn, saturated: saturated,
             outOfEnvelope: outOfEnvelope };
  }

  var api = {
    applyInputTransform: applyInputTransform,
    normaliseInputs: normaliseInputs,
    denormaliseOutputs: denormaliseOutputs,
    forwardNormalised: forwardNormalised,
    predict: predict,
  };

  root.EGUN = root.EGUN || {};
  root.EGUN.ann = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
