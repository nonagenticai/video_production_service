/*
 * _motion_lib.js — shared motion vocabulary for the branded motion-graphics
 * templates (VP-3, aspect A4 / C23 consumer). Injected into every template's
 * <head> by templates/__init__.py::get_template (after the embedded fonts,
 * before the template's own body script), so every template shares ONE easing
 * library, ONE entrance/exit grammar, and ONE timeline reader.
 *
 * Self-contained: no imports, no network, no wall-clock reads, no RNG — the
 * animation recorder frame-steps window.render(t) deterministically, so every
 * value here must be a pure function of (PARAMS, t).
 *
 * PARAMS contract (what the orchestrator actually sends — see
 * w2v_orchestrator/src/services/stages/shared/motion_slides.py
 * ::_motion_signature_params; ALL optional, string-typed, defaults below):
 *   easing       cubic-bezier name (easeOutCubic default)
 *   stagger_ms   inter-element delay, ms
 *   entrance     entrance direction: "up" | "down" | "left" | "right" | "scale" | "fade"
 *   exit         exit direction (same vocabulary)
 *   parallax     0..1 depth-travel scale
 *   glow_drift   0..1 ambient-glow drift scale
 *   entrance_ms  entrance window length, ms
 *   hold_ms      on-screen hold length, ms
 *   exit_ms      exit window length, ms  (0 => no exit, card holds to the end)
 *   marks        JSON array of absolute element entrance offsets, ms
 * Any key absent => the default below, so a params-less render is unchanged.
 */
(function () {
  "use strict";

  // --- named cubic-bezier easings ---------------------------------------
  function cubicBezier(x1, y1, x2, y2) {
    // Sample the bezier's x for a target t via a few Newton/bisection steps,
    // then return its y. Deterministic, allocation-free.
    function bez(a, b, u) {
      var v = 1 - u;
      return 3 * v * v * u * a + 3 * v * u * u * b + u * u * u;
    }
    return function (t) {
      if (t <= 0) return 0;
      if (t >= 1) return 1;
      var lo = 0, hi = 1, u = t;
      for (var i = 0; i < 20; i++) {
        var x = bez(x1, x2, u);
        if (Math.abs(x - t) < 1e-4) break;
        if (x < t) lo = u; else hi = u;
        u = (lo + hi) / 2;
      }
      return bez(y1, y2, u);
    };
  }

  var EASINGS = {
    linear: function (p) { return p; },
    easeOutCubic: function (p) { return 1 - Math.pow(1 - p, 3); },
    easeOutQuint: function (p) { return 1 - Math.pow(1 - p, 5); },
    easeOutExpo: function (p) { return p >= 1 ? 1 : 1 - Math.pow(2, -10 * p); },
    easeOutBack: cubicBezier(0.34, 1.56, 0.64, 1.0),
    easeInOutCubic: cubicBezier(0.65, 0.0, 0.35, 1.0),
    easeOutQuart: function (p) { return 1 - Math.pow(1 - p, 4); },
  };

  function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }

  // Eased progress in [0,1] for a raw progress p, by easing NAME (falls back to
  // easeOutCubic for an unknown/absent name).
  function ease(name, p) {
    var fn = EASINGS[name] || EASINGS.easeOutCubic;
    return fn(clamp01(p));
  }

  // --- timeline reader ---------------------------------------------------
  function num(v, dflt) {
    if (v === undefined || v === null || v === "") return dflt;
    var n = typeof v === "number" ? v : parseFloat(v);
    return isFinite(n) ? n : dflt;
  }

  // Presence test for an optional (string-typed) PARAM: true only when the
  // orchestrator actually sent this key. Templates gate a signature-derived
  // value on ITS OWN source key with this, so a PARTIAL signature (e.g. only
  // `easing`) never disturbs an un-set dimension (stagger/timeline) — and an
  // empty-string placeholder counts as absent.
  function has(P, k) {
    return !!(P && P[k] !== undefined && P[k] !== null && P[k] !== "");
  }

  // Resolve the numeric motion timeline from PARAMS, with defaults that
  // reproduce the legacy hand-tuned feel when no motion block is present.
  // `nElements`/`fallbackTotal` let a template size its own default timeline.
  function timing(P, nElements, fallbackTotal) {
    P = P || {};
    var n = nElements > 0 ? nElements : 1;
    var entrance = num(P.entrance_ms, 1500);
    // Cap the stagger so the LAST element's entrance start ((n-1)*stagger) never
    // exceeds the entrance window — otherwise a trailing element can be captured
    // before it finishes fading in (the recorder honours a short request-level
    // duration_ms over RENDER_DURATION_MS). Mirrors the orchestrator's own guard
    // plan_card_timing: stagger_ms = min(sig.stagger_ms, entrance_ms // n).
    var stagger = Math.min(num(P.stagger_ms, 420), n > 1 ? Math.floor(entrance / n) : entrance);
    var hold = num(P.hold_ms, Math.max(0, (fallbackTotal || 4200) - entrance - 900));
    var exit = num(P.exit_ms, 0);
    var marks;
    if (typeof P.marks === "string" && P.marks) {
      try { marks = JSON.parse(P.marks); } catch (e) { marks = null; }
    } else if (Array.isArray(P.marks)) {
      marks = P.marks;
    }
    if (!Array.isArray(marks) || marks.length === 0) {
      marks = [];
      for (var i = 0; i < n; i++) marks.push(i * stagger);
    }
    var holdEnd = entrance + hold;
    return {
      stagger: stagger,
      entrance: entrance,
      hold: hold,
      exit: exit,
      marks: marks,
      holdEnd: holdEnd,
      total: holdEnd + exit,
      parallax: num(P.parallax, 0),
      glowDrift: num(P.glow_drift, 1),
      easing: (P && P.easing) || "easeOutCubic",
      entranceDir: (P && P.entrance) || "up",
      exitDir: (P && P.exit) || "fade",
    };
  }

  // Total render duration (ms) for window.RENDER_DURATION_MS. Uses the resolved
  // timeline total when a motion block drove it, else the caller's fallback.
  function renderDurationMs(P, fallbackTotal) {
    var t = timing(P, 1, fallbackTotal);
    // If the caller passed a real per-card duration via hold, honour the total;
    // otherwise fall back so legacy (no-PARAMS) renders are unchanged.
    return Math.max(t.total, fallbackTotal || 0) || fallbackTotal || t.total;
  }

  // --- the one entrance+exit primitive ----------------------------------
  // Offset (px) for a direction at eased progress `e` (1 => settled at 0).
  function offsetFor(dir, e, travel) {
    var d = (1 - e) * travel;
    switch (dir) {
      case "down": return "translateY(" + -d + "px)";
      case "up": return "translateY(" + d + "px)";
      case "left": return "translateX(" + d + "px)";
      case "right": return "translateX(" + -d + "px)";
      case "scale": return "scale(" + (1 - (1 - e) * 0.06) + ")";
      case "fade": return "translateY(0px)";
      default: return "translateY(" + d + "px)";
    }
  }

  // Animate `el` for absolute time `t` (ms). opts:
  //   start   entrance start offset (ms) — usually T.marks[i] (+ a base)
  //   dur     entrance duration (ms)
  //   travel  entrance travel distance (px)
  //   T       the timing() object for this render
  //   exitDur exit duration (ms, default T.exit)
  // Entrance ramps opacity 0->1 + slides in; after T.holdEnd it ramps back out
  // in T.exitDir over exitDur (skipped when exit == 0).
  function layer(el, t, opts) {
    if (!el) return;
    var T = opts.T;
    var start = opts.start || 0;
    var dur = opts.dur || 600;
    var travel = opts.travel === undefined ? 20 : opts.travel;
    var eIn = ease(T.easing, (t - start) / dur);
    var opacity = eIn;
    var transform = offsetFor(T.entranceDir, eIn, travel);
    var exitDur = opts.exitDur === undefined ? T.exit : opts.exitDur;
    if (exitDur > 0 && t > T.holdEnd) {
      var eOut = ease("easeOutCubic", (t - T.holdEnd) / exitDur);
      opacity = 1 - eOut;
      transform = offsetFor(T.exitDir, 1 - eOut, travel);
    }
    el.style.opacity = opacity;
    el.style.transform = transform;
  }

  // Deterministic ambient drift for a decorative glow layer, scaled by
  // T.glowDrift (0 disables). Mirrors the legacy animated_slide drift feel.
  function glow(el, t, T) {
    if (!el) return;
    var k = T.glowDrift;
    if (k <= 0) { el.style.transform = "translate(0,0) scale(1)"; return; }
    var period = 12000;
    var phase = (t % period) / period; // 0..1, deterministic
    var s = Math.sin(phase * Math.PI * 2);
    el.style.transform =
      "translate(" + (s * 6 * k) + "vmin," + (-Math.abs(s) * 4 * k) + "vmin) scale(" +
      (1 + Math.abs(s) * 0.12 * k) + ")";
    el.style.opacity = 0.18 + Math.abs(s) * 0.1;
  }

  // Map the flat brand PARAMS onto :root CSS custom properties, exactly the
  // vocabulary the existing templates use (--bg/--bg2/--accent/--accent2/
  // --text/--muted + font families). Only sets a var when the param is present,
  // so a template's built-in defaults survive.
  function applyBrandVars(P) {
    P = P || {};
    var root = document.documentElement.style;
    var map = {
      bg: P.bg, bg2: P.bg2, accent: P.accent, accent2: P.accent2,
      text: P.text, muted: P.muted,
      "font-display": P.font_display, "font-sans": P.font_sans,
    };
    for (var k in map) {
      if (map[k]) root.setProperty("--" + k, map[k]);
    }
  }

  window.MotionLib = {
    cubicBezier: cubicBezier,
    EASINGS: EASINGS,
    ease: ease,
    clamp01: clamp01,
    has: has,
    timing: timing,
    renderDurationMs: renderDurationMs,
    layer: layer,
    glow: glow,
    applyBrandVars: applyBrandVars,
  };
})();
