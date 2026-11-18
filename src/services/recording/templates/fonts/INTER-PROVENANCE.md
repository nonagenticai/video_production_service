# Inter (film sans) — provenance

`inter-{400,600,700}-normal-latin.woff2` are the static Regular / SemiBold /
Bold faces of **Inter 4.1**, embedded as data URIs into every motion template
(`templates/__init__.py::_FONT_FACES`) so a brand that asks for Inter gets real
600/700 weights instead of browser-synthesised bold.

- Upstream: https://github.com/rsms/inter — SIL Open Font License 1.1 (`OFL-Inter.txt`).
- Fetched from the official release `Inter-4.1.zip`
  (sha256 `9883fdd4a49d4fb66bd8177ba6625ef9a64aa45899767dde3d36aa425756b11e`),
  files `web/Inter-{Regular,SemiBold,Bold}.woff2`.
- Subset to latin with fontTools (`pyftsubset --flavor=woff2`), unicodes
  `U+0000-00FF,U+0131,U+0152-0153,U+02BB-02BC,U+02C6,U+02DA,U+02DC,U+2000-206F,U+20AC,U+2122,U+2190-2193,U+2212,U+2215,U+FEFF,U+FFFD`,
  layout features `kern,liga,calt,tnum,case,ccmp,locl,mark,mkmk`.
- Size: ~17.5 KB per face (53 KB raw, ~71 KB as base64 in each page).
- No italic face is embedded: film-style templates set `font-synthesis:none`
  and never ask for italic.
