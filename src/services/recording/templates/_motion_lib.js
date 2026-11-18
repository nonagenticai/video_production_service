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


  // =====================================================================
  // FILM KIT — shared look + clock for the explainer-film templates
  // (two_beat, levels, checklist, handoff, and the opt-in film layouts of
  // two_up / timeline / animated_title_card). Everything below is inert until a
  // template calls MotionLib.film(P, ...), so templates that never do render
  // exactly as before.
  //
  // PARAMS read by the kit (all optional):
  //   duration_ms  number  clip length (ms); pass the SAME value as the record
  //                        request so default marks spread over the real shot.
  //   marks        number[] | JSON string — absolute beat offsets (ms). Beat 0
  //                        always starts at 0 (never an empty first frame);
  //                        missing trailing beats continue at the last spacing.
  //   stagger_ms   number  beat spacing when `marks` is absent.
  //   easing       string  named easing (EASINGS) for entrances.
  //   film_t_ms    number  where this card starts on the FILM clock (ms). The
  //                        ambient layer is a pure function of film_t_ms + t and
  //                        of nothing template-specific, so consecutive cards
  //                        that pass their running start offset cut seamlessly.
  //   glow_drift   0..1    ambient drift amplitude (0 freezes it).
  // =====================================================================

  // One viewport unit for every film layout: 1u = 1% of a 16:9 frame's width
  // (19.2px at 1080p, 38.4px at 2160p), shrinking to fit on other aspects.
  var FILM_CSS =
    ":root{--u:min(1vw,1.7778vh)}" +
    "@media (max-aspect-ratio:1/1){:root{--u:min(1.9vw,1.05vh)}}" +
    "body.film{font-family:var(--font-sans);font-synthesis:none;color:var(--text);" +
    "-webkit-font-smoothing:antialiased;text-rendering:geometricPrecision}" +
    "body.film .stage{background:radial-gradient(120% 120% at 72% 8%,var(--bg2),var(--bg)) !important;" +
    "background-size:auto !important;animation:none !important}" +
    "body.film .stage>.glow{display:none !important}" +
    ".amb{position:absolute;inset:0;overflow:hidden;pointer-events:none;z-index:-1}" +
    ".amb .glow{position:absolute;border-radius:50%;filter:blur(calc(var(--u)*7));will-change:transform}" +
    ".amb .g1{width:calc(var(--u)*46);height:calc(var(--u)*46);right:calc(var(--u)*-10);" +
    "top:calc(var(--u)*-16);background:var(--accent);opacity:.24}" +
    ".amb .g2{width:calc(var(--u)*40);height:calc(var(--u)*40);left:calc(var(--u)*-12);" +
    "bottom:calc(var(--u)*-18);background:var(--accent2);opacity:.15}" +
    ".amb .grid{position:absolute;inset:calc(var(--u)*-4);opacity:.11;" +
    "background-image:radial-gradient(circle,var(--text) 0,var(--text) calc(var(--u)*.07),transparent calc(var(--u)*.1));" +
    "background-size:calc(var(--u)*3.2) calc(var(--u)*3.2);" +
    "-webkit-mask-image:radial-gradient(90% 90% at 78% 12%,#000,transparent 72%);" +
    "mask-image:radial-gradient(90% 90% at 78% 12%,#000,transparent 72%)}" +
    ".fk{font-size:calc(var(--u)*2.3);font-weight:700;letter-spacing:.16em;text-transform:uppercase;" +
    "color:var(--accent);line-height:1.2}" +
    ".ft{font-family:var(--font-display);font-weight:700;font-size:calc(var(--u)*4.3);line-height:1.08;" +
    "letter-spacing:-.022em;color:var(--text);text-wrap:balance}";

  // ---------------------------------------------------------------------
  // CAPTION BAND — where video_compiler burns the film's captions, so film
  // cards can keep their text out of it. COMPUTED from the compiler, not
  // guessed (video_compiler_service origin/main @ 608bfdd):
  //   src/config.py:671  caption_font_size_frac   0.045 x render_height
  //   src/config.py:676  caption_margin_v_frac    0.06  x render_height (MarginV)
  //   src/config.py:741  caption_box_padding_frac 0.25  x font size (box Outline)
  //   src/config.py:723  caption_box_enabled      True  (BorderStyle=3 box per line)
  //   src/services/captions_ass.py:53   mood "energetic" font_size_frac 0.050 (the
  //                                     largest; no mood changes margin_v_frac)
  //   src/services/captions_ass.py:133  _wrap: at most TWO lines per cue
  //   src/services/captions_ass.py:318-339  font_size = round(H*frac), margin_v =
  //       round(H*0.06), Outline = round(font_size*0.25)
  //   src/services/captions_ass.py:346  Alignment 2 (bottom centre), MarginV
  //   src/services/captions_ass.py:352  PlayResX/Y = render size, so these are
  //                                     pixels of the delivered frame
  // libass sizes a DejaVu Sans line to exactly Fontsize px, so the box top is
  //   H - margin_v - lines*font_size - padding.
  // Rendered with libass (ffmpeg ass filter, the same style line) and measured:
  //            1 line        2 lines       2 lines, energetic
  //   1080p    y=954 .8833   y=905 .8380   y=893 .8269
  //   2160p    y=1909 .8838  y=1812 .8389  y=1787 .8273
  // CAPTION_SAFE_FRAC (bottom 18% of the frame) clears the worst case (2 lines,
  // largest mood font) at both sizes; a film card with caption_safe on (the
  // default) lays its content out above it.
  var CAPTION = { font_frac: 0.045, font_frac_max: 0.05, margin_v_frac: 0.06, pad_frac: 0.25, max_lines: 2 };
  var CAPTION_SAFE_FRAC = 0.18;

  // y (px from the top) of the caption box's top edge for an H-px-high frame.
  function captionBoxTop(H, lines, fontFrac) {
    var fs = Math.round(H * (fontFrac || CAPTION.font_frac));
    var pad = Math.max(1, Math.round(fs * CAPTION.pad_frac));
    return H - Math.round(H * CAPTION.margin_v_frac) - (lines || CAPTION.max_lines) * fs - pad;
  }

  var FILM_LEAD_MS = 180; // beat 0 is already ~3/4 in at t=0: a cut never lands on an empty frame
  var TAU = Math.PI * 2;

  function filmCss() {
    // Switch on the film kit's embedded Inter faces (shipped inert — see
    // templates/__init__.py::_FILM_FONT_FACES).
    var ff = document.getElementById("film-fonts");
    if (ff) ff.media = "all";
    if (document.getElementById("film-css")) return;
    var st = document.createElement("style");
    st.id = "film-css";
    // caption_safe: the stage's bottom edge sits on the caption band's top, so a
    // card's flex layout (centred or stretched) lays out above the captions.
    st.textContent = FILM_CSS +
      "body.film.cap-safe{--cap-safe:calc(100vh*" + CAPTION_SAFE_FRAC + ")}" +
      "body.film.cap-safe .stage{padding-bottom:var(--cap-safe) !important}";
    document.head.appendChild(st);
  }

  function parseMarks(P) {
    var m = null;
    if (P && typeof P.marks === "string" && P.marks) {
      try { m = JSON.parse(P.marks); } catch (e) { m = null; }
    } else if (P && Array.isArray(P.marks)) {
      m = P.marks;
    }
    return Array.isArray(m) && m.length ? m : null;
  }

  // Resolve n beat offsets (ms) for a clip of `dur` ms. Caller marks win; absent
  // marks spread over the first half of the shot. Beat 0 is pinned to 0, the
  // list is made non-decreasing, and every beat is kept early enough to finish
  // its entrance inside the shot.
  function filmMarks(P, n, dur) {
    P = P || {};
    n = n > 0 ? n : 1;
    var given = parseMarks(P) || [];
    var step = has(P, "stagger_ms")
      ? Math.max(0, num(P.stagger_ms, 0))
      : (n > 1 ? Math.min(1800, (dur * 0.5) / (n - 1)) : 0);
    var latest = Math.max(0, dur - 1200);
    var out = [];
    for (var i = 0; i < n; i++) {
      var v;
      if (i === 0) v = 0;
      else if (i < given.length) v = num(given[i], out[i - 1]);
      else if (given.length > 1) v = out[i - 1] + Math.max(0, out[i - 1] - (i > 1 ? out[i - 2] : 0));
      else v = out[i - 1] + step;
      if (i > 0 && v < out[i - 1]) v = out[i - 1];
      out.push(Math.min(v, latest));
    }
    return out;
  }

  // The ambient layer's transforms for film time `ft` (ms): two soft glows on
  // slow, incommensurate sine paths plus a dot grid on a constant drift. Pure —
  // no element, no template input — which is what makes it identical per brand.
  function ambientState(ft, k) {
    k = k === undefined ? 1 : k;
    var gx = ((ft / 1000) * 0.32) % 3.2;
    var gy = ((ft / 1000) * 0.18) % 3.2;
    return {
      g1: { x: 6 * k * Math.sin((TAU * ft) / 23000), y: 4 * k * Math.sin((TAU * ft) / 31000 + 1.3),
            s: 1 + 0.1 * k * Math.sin((TAU * ft) / 17000 + 0.4) },
      g2: { x: 5 * k * Math.sin((TAU * ft) / 27000 + 2.1), y: 3.5 * k * Math.sin((TAU * ft) / 19000 + 0.7),
            s: 1 + 0.08 * k * Math.sin((TAU * ft) / 29000) },
      grid: { x: -gx * k, y: -gy * k },
    };
  }

  function uTranslate(x, y) {
    return "translate(calc(var(--u)*" + x.toFixed(4) + "),calc(var(--u)*" + y.toFixed(4) + "))";
  }

  // Draw an SVG stroke on: the first `p` (0..1) of the path, in path order. The
  // dash is set in the path's own MEASURED length (getTotalLength), so what is
  // visible is exactly the geometric prefix of that length — no reliance on a
  // pathLength normalisation.
  function draw(el, p) {
    if (!el) return;
    p = clamp01(p);
    var L = el.__len || (el.__len = el.getTotalLength());
    el.style.strokeDasharray = L.toFixed(3) + " " + (L + 1).toFixed(3);
    el.style.strokeDashoffset = (L * (1 - p)).toFixed(3);
    el.style.visibility = p <= 0.001 ? "hidden" : "visible";
  }

  // The verdict tick (viewBox 0 0 24 24), authored LONG leg first: tip -> corner
  // -> short end. A tick drawn short-leg-first passes through a frame where both
  // legs are equally long — a symmetric "V", i.e. a chevron (#188 review, L3 at
  // 50%). Drawn long-leg-first, every partial is either a single stroke or a
  // full long leg plus a short leg at most half its length: always a tick.
  var TICK_D = "M18 8L10.5 16.5L6.5 12.5";

  // Shrink `el`'s font-size until it fits `maxLines` lines and its own box
  // (layout metrics only, so entrance transforms do not disturb it).
  // Text that still does not fit at the floor is NOT clamped or hidden: it is
  // drawn as it wraps, and reported in window.FIT_OVERFLOW (element key ->
  // {lines, max_lines}) so the recorder can flag the card (render 01a1179d:
  // a caller cut its own text to fit, and nothing knew).
  function reportFit(el, lines, maxLines) {
    var key = el.id || el.className || "text";
    var book = (window.FIT_OVERFLOW = window.FIT_OVERFLOW || {});
    if (lines > maxLines || el.scrollWidth > el.clientWidth + 1) {
      book[key] = { lines: lines, max_lines: maxLines };
    } else {
      delete book[key];
    }
  }

  function fit(el, maxLines, minScale) {
    if (!el || !el.textContent) return;
    el.style.fontSize = "";
    var cs = getComputedStyle(el);
    var base = parseFloat(cs.fontSize);
    var lh = parseFloat(cs.lineHeight);
    if (!isFinite(base) || base <= 0) return;
    var ratio = isFinite(lh) && lh > 0 ? lh / base : 1.15;
    var floor = minScale || 0.6;
    var s = 1;
    var lines = 0;
    for (var i = 0; i < 30; i++) {
      lines = Math.round(el.offsetHeight / (base * s * ratio));
      if (lines <= maxLines && el.scrollWidth <= el.clientWidth + 1) break;
      s *= 0.95;
      if (s < floor) { s = floor; i = 30; }
      el.style.fontSize = (base * s) + "px";
    }
    reportFit(el, Math.round(el.offsetHeight / (base * s * ratio)), maxLines);
  }

  // The film's recurring motif: a dial — tick ring, a gauge arc that draws on and
  // then keeps being nudged, a counter-rotating inner arc. viewBox 200x200.
  // `k` scales every stroke (a dial drawn large passes k < 1 to keep its lines fine).
  function orbitSvg(k) {
    function w(x) { return (x * k).toFixed(2); }
    var ticks = "";
    for (var i = 0; i < 60; i++) {
      var a = (TAU * i) / 60;
      var r0 = i % 5 === 0 ? 86 : 90;
      ticks += "M" + (100 + r0 * Math.cos(a)).toFixed(2) + " " + (100 + r0 * Math.sin(a)).toFixed(2) +
        "L" + (100 + 95 * Math.cos(a)).toFixed(2) + " " + (100 + 95 * Math.sin(a)).toFixed(2);
    }
    return (
      '<svg viewBox="0 0 200 200" width="100%" height="100%" fill="none" stroke-linecap="round" aria-hidden="true">' +
      '<path class="ob-ticks" d="' + ticks + '" stroke="var(--text)" stroke-opacity=".28" stroke-width="' + w(1) + '"/>' +
      '<circle cx="100" cy="100" r="74" stroke="var(--text)" stroke-opacity=".14" stroke-width="' + w(1.2) + '"/>' +
      '<circle class="ob-dash" cx="100" cy="100" r="54" stroke="var(--text)" stroke-opacity=".3" stroke-width="' + w(1.2) + '" stroke-dasharray="1.5 7.5"/>' +
      '<circle class="ob-arc" cx="100" cy="100" r="74" stroke="var(--accent)" stroke-width="' + w(4.5) + '"/>' +
      '<circle class="ob-in" cx="100" cy="100" r="36" stroke="var(--accent2)" stroke-opacity=".75" stroke-width="' + w(2.4) + '" stroke-dasharray="52 400"/>' +
      '<circle class="ob-tip" r="' + w(6.5) + '" fill="var(--accent2)" stroke="var(--bg)" stroke-width="' + w(2.5) + '"/>' +
      "</svg>"
    );
  }

  // Paint the dial in `host` for clip time t: `p` 0..1 is the draw-on progress,
  // after which the arc tip keeps travelling (period ~5.6s) and the tick ring and
  // inner rings rotate — the continuous foreground motion of the cards that
  // carry it. opts: { inner: false drops the two inner rings (for a dial drawn
  // AROUND a text lockup, where they would cross the type); stroke: line-weight
  // scale, default 1 }.
  function orbit(host, t, p, opts) {
    if (!host) return;
    if (!host.firstChild) {
      host.innerHTML = orbitSvg(opts && opts.stroke > 0 ? opts.stroke : 1);
      if (opts && opts.inner === false) {
        host.querySelector(".ob-dash").style.display = "none";
        host.querySelector(".ob-in").style.display = "none";
      }
    }
    var tk = host.querySelector(".ob-ticks");
    tk.style.transformOrigin = "100px 100px";
    tk.style.transform = "rotate(" + ((t / 1000) * 1.5).toFixed(3) + "deg)";
    var C = TAU * 74;
    var START = -220, SWEEP = 260; // degrees: a gauge open at the bottom
    var frac = clamp01(p) * (0.8 + 0.14 * Math.sin((TAU * t) / 5600));
    var arc = host.querySelector(".ob-arc");
    arc.style.strokeDasharray = (C * (SWEEP / 360) * frac).toFixed(2) + " " + (C + 10).toFixed(2);
    arc.style.transform = "rotate(" + START + "deg)";
    arc.style.transformOrigin = "100px 100px";
    arc.style.visibility = frac <= 0.002 ? "hidden" : "visible";
    var ang = ((START + SWEEP * frac) * Math.PI) / 180;
    var tip = host.querySelector(".ob-tip");
    tip.setAttribute("cx", (100 + 74 * Math.cos(ang)).toFixed(2));
    tip.setAttribute("cy", (100 + 74 * Math.sin(ang)).toFixed(2));
    tip.style.opacity = clamp01(p * 4);
    var d = host.querySelector(".ob-dash");
    d.style.transformOrigin = "100px 100px";
    d.style.transform = "rotate(" + ((t / 1000) * 5).toFixed(3) + "deg)";
    var inn = host.querySelector(".ob-in");
    inn.style.transformOrigin = "100px 100px";
    inn.style.transform = "rotate(" + (-(t / 1000) * 14).toFixed(3) + "deg)";
    inn.style.opacity = clamp01(p * 2);
  }

  // Build the film clock for one card. opts: { beats, fallbackMs, stage }.
  function film(P, opts) {
    P = P || {};
    opts = opts || {};
    filmCss();
    document.body.classList.add("film");
    var n = opts.beats > 0 ? opts.beats : 1;
    var fallback = opts.fallbackMs || 6000;
    var dur = num(P.duration_ms, fallback);
    if (!(dur >= 500)) dur = fallback;
    var marks = filmMarks(P, n, dur);
    var easing = P.easing || "easeOutCubic";
    var filmT = num(P.film_t_ms, 0);
    var drift = num(P.glow_drift, 1);
    var capSafe = !has(P, "caption_safe") || !(P.caption_safe === false ||
      /^(false|0|no|off)$/i.test(String(P.caption_safe)));
    document.body.classList.toggle("cap-safe", capSafe);
    var amb = null;

    function start(i) { return i === 0 ? -FILM_LEAD_MS : marks[Math.min(i, marks.length - 1)]; }

    var F = {
      dur: dur,
      marks: marks,
      easing: easing,
      filmT: filmT,
      captionSafe: capSafe,
      // Eased 0..1 progress of beat i, `d` ms long, optionally `delay` ms later.
      p: function (t, i, d, delay) { return ease(easing, (t - start(i) - (delay || 0)) / (d || 600)); },
      // Linear 0..1 progress between two absolute times.
      span: function (t, a, b) { return b > a ? clamp01((t - a) / (b - a)) : (t >= b ? 1 : 0); },
      // -1..1 sine of the clip clock; 0..1 sawtooth of the clip clock.
      wave: function (t, period, phase) { return Math.sin((TAU * t) / period + (phase || 0)); },
      loop: function (t, period, phase) {
        var x = (t / period + (phase || 0)) % 1;
        return x < 0 ? x + 1 : x;
      },
      // Fade + rise an element in on beat i. o: { dur, travel (u), delay, dir }.
      enter: function (el, t, i, o) {
        if (!el) return 0;
        o = o || {};
        var e = F.p(t, i, o.dur || 620, o.delay);
        var d = (1 - e) * (o.travel === undefined ? 1.6 : o.travel);
        el.style.opacity = e;
        el.style.transform = o.dir === "left" ? uTranslate(d, 0) : o.dir === "right" ? uTranslate(-d, 0) : uTranslate(0, d);
        return e;
      },
      // Position the shared ambient layer for clip time t (built on first call).
      ambient: function (t) {
        if (!amb) {
          var stage = opts.stage || document.querySelector(".stage") || document.body;
          var el = document.createElement("div");
          el.className = "amb";
          el.innerHTML = '<div class="grid"></div><div class="glow g1"></div><div class="glow g2"></div>';
          stage.insertBefore(el, stage.firstChild);
          amb = { grid: el.children[0], g1: el.children[1], g2: el.children[2] };
        }
        var a = ambientState(filmT + t, drift);
        amb.g1.style.transform = uTranslate(a.g1.x, a.g1.y) + " scale(" + a.g1.s.toFixed(4) + ")";
        amb.g2.style.transform = uTranslate(a.g2.x, a.g2.y) + " scale(" + a.g2.s.toFixed(4) + ")";
        amb.grid.style.transform = uTranslate(a.grid.x, a.grid.y);
      },
      // Run a layout pass now and again once the embedded fonts have loaded
      // (the recorder awaits document.fonts.ready before the first frame).
      ready: function (fn) {
        fn();
        if (document.fonts && document.fonts.ready) {
          void document.body.offsetHeight;
          document.fonts.ready.then(function () { fn(); if (typeof window.render === "function") window.render(0); });
        }
      },
    };
    return F;
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
    film: film,
    filmMarks: filmMarks,
    ambientState: ambientState,
    draw: draw,
    TICK_D: TICK_D,
    fit: fit,
    orbit: orbit,
    FILM_LEAD_MS: FILM_LEAD_MS,
    CAPTION: CAPTION,
    CAPTION_SAFE_FRAC: CAPTION_SAFE_FRAC,
    captionBoxTop: captionBoxTop,
  };
})();
