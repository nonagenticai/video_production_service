# Fraunces (brand display face) — provenance

`Fraunces-Regular.ttf` and `Fraunces-SemiBold.ttf` are static instances of the
canonical **Fraunces** variable font, used by the PIL/FFmpeg title-card path
(`src/services/compilation/title_card_generator.py`) so compilation title cards
render in the brand face instead of the generic DejaVu/Liberation Sans fallback.
This mirrors the HTML/Chromium motion cards, which already embed Fraunces
(`src/services/recording/templates/fonts/fraunces-400-normal-latin.woff2`).

## Source
- Upstream: Fraunces VF, SIL Open Font License 1.1 — https://github.com/undercasetype/Fraunces
- Fetched from Google Fonts: `ofl/fraunces/Fraunces[SOFT,WONK,opsz,wght].ttf`
- License text: `OFL-Fraunces.txt` (in this directory).

## How the static instances were produced
`fontTools.varLib.instancer.instantiateVariableFont` pinning:

| File                    | wght | opsz | SOFT | WONK | used for                              |
|-------------------------|------|------|------|------|---------------------------------------|
| `Fraunces-Regular.ttf`  | 400  | 28   | 0    | 0    | subtitle / bullets / channel (18–22px)|
| `Fraunces-SemiBold.ttf` | 600  | 72   | 0    | 0    | headline title (52px, `bold=True`)    |

`opsz` is matched to the render size (text vs display); `SOFT=0`/`WONK=0` give
the clean, non-quirky brand letterforms. Pillow cannot load `.woff2`, which is
why real `.ttf` faces are vendored here. The `Dockerfile` `COPY assets/fonts/*.ttf`
glob installs them into the image automatically.
