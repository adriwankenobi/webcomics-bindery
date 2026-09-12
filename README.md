# webcomics-bindery — webcomics → print books

Pipeline to turn webcomics (e.g. fan translations) into print-ready PDFs,
including an optional **re-lettering** stage that re-typesets every speech
bubble bigger and sharper for small trim sizes (proven on a full 333-page
book, ~×1.3–1.5 over the original lettering). See `ARCHITECTURE.md` for how
everything works inside.

**Language support**: nothing in the pipeline is tied to one language — all
text lives in per-comic transcript data, and detection/fitting work on
pixels and space-separated words. Any left-to-right, space-separated,
Latin-script-like language works out of the box (accents included, provided
your chosen font has the glyphs). CJK (no word spaces) and right-to-left
scripts are **not** supported. **Fonts are user-supplied** (licensed
material, not in the repo): see `relettering/fonts/README.md`.

## Layout

```
<comic folder>/              source pages (jpg/png) — the folder NAME is the
                             comic id passed to the scripts. NEVER modified.
upscaled/<comic>/            pages at exact placement size (Real-ESRGAN);
                             the re-lettering stage overwrites these with
                             cleaned pages
xcf/<comic>/                 one .xcf per page (template + page + text layers)
pdf/<comic>/                 one .pdf per page
pdf/<comic>.pdf              merged book
relettering/<comic>/         re-lettering workdir (bubbles, sheets, parts,
                             transcripts, layout, pristine copies, QA)
template.xcf                 print template (6.25x9.25" @300dpi incl. bleed)
relettering/fonts/           YOUR comic font (user-supplied, untracked):
                             regular.ttf + bolditalic.ttf + fonts.json —
                             see relettering/fonts/README.md
gimp-plugin/                 versioned copy of the GIMP compose plugin
```

Git tracks **only the process** (whitelist `.gitignore`). All comic data is
untracked and reproducible from the source folder alone.

## Setup (one-time)

Prerequisites: macOS, GIMP 2.10 in `/Applications`, plus `just` and `uv`.
Steps 2–4 pull in material that is deliberately **not** in the repo
(third-party binaries, model weights, licensed fonts).

**1. Python environments** — two venvs; `cv2` lives only in the second:

```
just setup                             # .venv: playwright pillow pikepdf numpy
uv venv .venv-reletter
uv pip install --python .venv-reletter/bin/python \
    opencv-python-headless pillow numpy fonttools
```

**2. Upscaler (Real-ESRGAN ncnn/Vulkan)** — a prebuilt upstream binary plus
~74 MB of model weights. `process.py` expects
`tools/realesrgan/realesrgan-ncnn-vulkan` with a sibling `models/` dir,
which is exactly the zip's layout, so unpack it straight into place:

```
mkdir -p tools/realesrgan
curl -L -o /tmp/realesrgan.zip \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-macos.zip
unzip -o /tmp/realesrgan.zip -d tools/realesrgan
chmod +x tools/realesrgan/realesrgan-ncnn-vulkan
xattr -dr com.apple.quarantine tools/realesrgan    # macOS blocks it otherwise
# smoke test (prints GPU lines + progress, writes /tmp/smoke.png):
tools/realesrgan/realesrgan-ncnn-vulkan -n realesr-animevideov3 -s 2 \
  -i tools/realesrgan/input.jpg -o /tmp/smoke.png -m tools/realesrgan/models
```

v0.2.5.0 is the last release carrying ncnn binaries (use the `-ubuntu` /
`-windows` asset of the same release on other platforms). The pipeline runs
`realesr-animevideov3`, not `realesrgan-x4plus-anime`, which visibly
brightens and desaturates the artwork. **Always pass `-m <models dir>`** if
you invoke the binary by hand: it resolves `models/` relative to the
current directory and otherwise exits 1 printing nothing.

**3. GIMP compose plugin** — copy the versioned plugin into GIMP's plug-ins
dir and make it executable (GIMP silently skips plug-ins without the
executable bit), then restart GIMP:

```
D=~/Library/Application\ Support/GIMP/2.10/plug-ins
cp gimp-plugin/webcomics_compose.py "$D"/ && chmod +x "$D"/webcomics_compose.py
```

**4. Comic font** — user-supplied and licensed; follow
`relettering/fonts/README.md` (needed only for the re-lettering stage).

## Base pipeline (`process.py`, via `justfile`)

```
just setup                   base venv only (full install: Setup above)
just process "<comic>"       upscale + compose + merge
just upscale|compose|merge   individual steps (resumable)
```

- **upscale**: Real-ESRGAN (`realesr-animevideov3`) to the exact size GIMP
  will place; story pages cropped to content; spreads split at the gutter.
- **compose**: GIMP plugin (`webcomics_compose.py` in the GIMP plug-ins
  dir; versioned copy in `gimp-plugin/`) places each page 1:1 centered on
  `template.xcf`, draws page numbers, exports PDFs
  (then recompressed + sRGB-tagged by `postprocess_pdf`).
- **merge**: concatenates page PDFs, inserting blanks so covers land right,
  index pages left, and spread halves on facing pages.

## Re-lettering stage (between upscale and compose)

Re-typesets all dialogue with your chosen comic font at a book-wide
uniform size (the 35th percentile of per-bubble maxima), as **editable
text layers** in the XCFs. **First supply the font** — copy it to
`relettering/fonts/regular.ttf` + `bolditalic.ttf`, install it system-wide
for GIMP, and set the GIMP family names in `relettering/fonts/fonts.json`
(full instructions in `relettering/fonts/README.md`). Scripts run with
`.venv-reletter/bin/python`, live in `relettering/`, and take the comic
folder name as their required first argument:

```
relettering/reletter_detect.py "<comic>"   text-first bubble detection
   -> relettering/<comic>/bubbles/ (per-page JSON + mask/crop PNGs)
relettering/make_sheets.py "<comic>"       pack bubble crops into sheets
   (transcribe sheets -> relettering/<comic>/parts/sheet-NNN.json, then:)
relettering/merge_transcripts.py "<comic>" -> transcripts.json
relettering/reletter_fit.py "<comic>"      fits text -> layout.json and
   cleans the bubbles in upscaled/ (wipe old lettering, keep the fill)
   (then: just compose, and, from the repo root:)
RELETTER_COMIC="<comic>" /Applications/GIMP-2.10.app/Contents/MacOS/gimp \
   -i -d --batch-interpreter python-fu-eval \
   -b "execfile('relettering/reletter_gimp.py')" -b "pdb.gimp_quit(0)"
   (adds text layers to the XCFs, re-exports PDFs;
    then postprocess_pdf on those PDFs and just merge)
relettering/qa_scan.py "<comic>"           pixel-diff cleaned pages vs
   relettering/<comic>/pristine/ (repaint footprints must be text-shaped)
```

### Per-comic editorial data (untracked, in `relettering/<comic>/`)

- `parts/sheet-NNN.json` — transcriptions; `parts/zz-*.json` — later
  override rounds (merge order is sorted filename, **later wins**; when a
  page's bubble count changes, update `sheets/manifest.json` too).
- `typo_fixes.json` — `[wrong, fixed]` pairs applied at typeset time
  (transcripts stay a verbatim record of the source).
- `layout_overrides.json` — per-bubble taste calls keyed `"<short> bNN"`:
  `{"size": N}` pins a font size past group caps, `{"anchor": "visual"}`
  centers on the visible lobe, `{"anchor": "lines"}` places line-by-line
  following the original letterer's per-line axes.

### Transcript conventions

One string per bubble, in `bNN` order (bubbles sort by text-block
position, so transcripts stay aligned even when masks change);
`*word*` = bold-italic; `\n` = hard line break (hyphenation, or mirroring
the original's line structure); `\n\n` = paragraph break (compound
balloons); `""` = leave the region untouched (SFX, display lettering,
onomatopoeia overflowing its bubble, art false positives). Covers,
index/credits and art pages are skipped automatically.

### Fixing a reported page (single-page rebuild)

Never re-run the global fit (it would re-encode every working JPEG).
Instead: regenerate the page's pristine upscale if missing → re-detect that
page only (compare bubble counts/blocks before promoting) → transcript
overrides in a new `parts/zz-*.json` + manifest counts → single-page fit
with the book size pinned → delete the page's `.xcf`/`.pdf`, compose, GIMP
pass with `layout.json` temporarily pruned to the affected stems →
`postprocess_pdf` → merge. Full recipe and scope-discipline notes in
`ARCHITECTURE.md`.

### QA

- `qa_scan.py` gate plus per-bubble diffs against pristine.
- **Leftover scan**: run detection on the cleaned working pages — any text
  found where the transcript has real text is lettering that survived
  cleaning.
- Render pages at 300 dpi and overlay 1:1 on the source (the composed page
  is centered on the 1875×2775 canvas) — compare placement and size
  against the original lettering, which is the gold standard.
