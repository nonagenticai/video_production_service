/*
 * _icons.js — a small inline line-icon set for the branded motion templates
 * (VP-3, aspect A4). Injected into every template's <head> by
 * templates/__init__.py::get_template. Exposes window.ICONS (name -> 24x24 SVG
 * path `d` string) and window.iconSvg(name, opts) -> an <svg> string stroked in
 * the current accent (or "" for an unknown/absent name).
 *
 * Self-contained (no network, no data-URIs). A template reads P.icon (a glyph
 * name) and injects iconSvg(P.icon). NOTE: the orchestrator does not send `icon`
 * on the live deck-motion path today, so P.icon is normally absent and templates
 * fall back to no icon — this is a forward-looking primitive (presence-is-opt-in)
 * ready for a future orchestrator projection PR. Glyph names are kept generic so
 * they can align with motion_plan.pick_icon's vocabulary when that lands.
 *
 * All paths are drawn for a 24x24 viewBox, meant to be stroked (fill="none").
 */
(function () {
  "use strict";

  var ICONS = {
    // structure / product
    check: "M4 12.5l5 5 11-11",
    arrow_right: "M4 12h15M13 6l6 6-6 6",
    arrow_up_right: "M7 17L17 7M9 7h8v8",
    bolt: "M13 2L4 14h7l-1 8 9-12h-7l1-8z",
    star: "M12 3l2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 18.8 6.1 21.9l1.1-6.5L2.5 9.8l6.5-.9L12 3z",
    spark: "M12 3v6M12 15v6M3 12h6M15 12h6",
    target: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 8a4 4 0 100 8 4 4 0 000-8zM12 11a1 1 0 100 2 1 1 0 000-2z",
    rocket: "M5 15c-1 2-1 4-1 4s2 0 4-1M9 15l-3-3c3-8 9-9 12-9 0 3-1 9-9 12zM14 9a1.5 1.5 0 100 .01",
    layers: "M12 3l9 5-9 5-9-5 9-5zM3 13l9 5 9-5M3 16l9 5 9-5",
    grid: "M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z",
    chart: "M4 20V6M4 20h16M9 20v-6M14 20v-9M19 20V9",
    trend_up: "M4 15l5-5 4 4 7-7M15 7h5v5",
    gauge: "M4 18a8 8 0 1116 0M12 18l4-6",
    clock: "M12 3a9 9 0 100 18 9 9 0 000-18zM12 7v5l3 2",
    calendar: "M4 5h16v16H4zM4 9h16M8 3v4M16 3v4",
    shield: "M12 3l8 3v6c0 5-4 8-8 9-4-1-8-4-8-9V6l8-3z",
    lock: "M6 11h12v9H6zM8 11V8a4 4 0 018 0v3",
    globe: "M12 3a9 9 0 100 18 9 9 0 000-18zM3 12h18M12 3c3 3 3 15 0 18M12 3c-3 3-3 15 0 18",
    users: "M9 11a3 3 0 100-6 3 3 0 000 6zM3 20a6 6 0 0112 0M17 11a3 3 0 10-.5-6M16 20a6 6 0 00-3-5",
    chat: "M4 5h16v11H9l-5 4V5z",
    doc: "M6 3h9l3 3v15H6zM14 3v4h4",
    code: "M9 8l-4 4 4 4M15 8l4 4-4 4",
    play: "M8 5l11 7-11 7V5z",
    heart: "M12 20s-7-4.5-9-9a4.5 4.5 0 019-1 4.5 4.5 0 019 1c-2 4.5-9 9-9 9z",
    gear: "M12 9a3 3 0 100 6 3 3 0 000-6zM19 12l2-1-1-3-2 .5a7 7 0 00-1.5-1.5L18 4l-3-1-1 2a7 7 0 00-2 0l-1-2-3 1 .5 2A7 7 0 006 8L4 7.5 3 10l2 1a7 7 0 000 2l-2 1 1 3 2-.5A7 7 0 007 20l1 2 3-1v-2a7 7 0 002 0v2l3 1 1-2 -.5-2A7 7 0 0018 16l2 .5 1-3-2-1a7 7 0 000-2z",
    // film set (same 24x24 single-path stroke style)
    cross: "M6 6l12 12M18 6L6 18",
    bot: "M12 3v4M6 7h12a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V9a2 2 0 012-2zM9 12v2M15 12v2M2 12v3M22 12v3",
    user: "M12 12a4 4 0 100-8 4 4 0 000 8zM4 21a8 8 0 0116 0",
    headset: "M4 14v-2a8 8 0 0116 0v2M4 14a2 2 0 012-2h1v6H6a2 2 0 01-2-2v-2zM20 14a2 2 0 00-2-2h-1v6h1a2 2 0 002-2v-2zM18 18v1a3 3 0 01-3 3h-3",
    chat_dots: "M4 5h16v11H9l-5 4V5zM8.5 10.5v.01M12 10.5v.01M15.5 10.5v.01",
    braces: "M8 4c-2 0-3 1-3 3v2c0 1.5-1 3-2 3 1 0 2 1.5 2 3v2c0 2 1 3 3 3M16 4c2 0 3 1 3 3v2c0 1.5 1 3 2 3-1 0-2 1.5-2 3v2c0 2-1 3-3 3",
    eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 100 6 3 3 0 000-6z",
    mail: "M3 6h18v12H3zM3 7l9 6 9-6",
    route: "M6 19a2 2 0 100-4 2 2 0 000 4zM18 9a2 2 0 100-4 2 2 0 000 4zM8 17h6a3 3 0 003-3v-5",
  };

  function iconSvg(name, opts) {
    opts = opts || {};
    var d = ICONS[name];
    if (!d) return "";
    var color = opts.color || "var(--accent)";
    var size = opts.size || "1em";
    var sw = opts.strokeWidth || 2;
    return (
      '<svg viewBox="0 0 24 24" width="' + size + '" height="' + size + '" ' +
      'fill="none" stroke="' + color + '" stroke-width="' + sw + '" ' +
      'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
      '<path d="' + d + '"/></svg>'
    );
  }

  window.ICONS = ICONS;
  window.iconSvg = iconSvg;
})();
