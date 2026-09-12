# Architecture

How the pipeline turns a folder of webcomic pages into a print-ready book,
and how the re-lettering subsystem re-typesets every speech bubble. This is
the deep reference; README.md has the how-to, CLAUDE.md the session rules.

## Design principles

- **Git tracks only the process.** The `.gitignore` is a whitelist: scripts,
  the template, docs. Comic fonts are licensed material and are **not
  tracked** — the user supplies them in `relettering/fonts/` (see the
  README there). All comic data — sources, `upscaled/`, `xcf/`,
  `pdf/`, `relettering/<comic>/` (transcripts included) — is untracked and
  must be reproducible from the source folder alone. A new comic = drop its
  source folder in, run the pipeline. No comic-specific content (names,
  page numbers) appears in tracked files; per-comic editorial data lives in
  untracked JSON files inside `relettering/<comic>/`.
- **Every step is resumable.** Outputs are skipped when they already exist
  (compose additionally skips when the XCF is newer than the upscaled page —
  an mtime rule, see "Footguns"). Deleting a page's outputs re-runs just
  that page.
- **The original letterer is the gold standard.** Wherever the re-lettering
  system needs a judgment call — where to center text, how to break lines,
  how close to the outline text may go — it measures what the original
  letterer did on that very bubble and reproduces it at the larger size.
  Geometry (centroids, inscribed circles) is only a fallback.

## Language support

Nothing language-specific is hardcoded: all text lives in per-comic
transcript data, detection is pixel-based, and the fit wraps
space-separated words. Supported: left-to-right, space-separated,
Latin-script-like languages (accents included if the chosen font has the
glyphs — verify with a render before committing to a book). Not supported:
CJK (no word spaces breaks wrapping; the letter-blob size heuristics assume
Latin-like caps) and right-to-left scripts (run order and centering assume
LTR). The letter-size heuristics (`LETTER_H` = 9-48 px etc.) are tuned for
pages ~1600 px wide — re-tune for very different resolutions.

## Data flow

```
<comic>/  ──upscale──▶  upscaled/<comic>/  ──compose──▶  xcf/<comic>/ + pdf/<comic>/  ──merge──▶  pdf/<comic>.pdf
                 │                ▲
                 │   (optional re-lettering, between upscale and compose)
                 │                │
                 ▼                │
        relettering/<comic>/   cleaned pages written back over upscaled/,
        (bubbles, transcripts,  text added later as GIMP layers on the XCFs
         layout, pristine)
```

Two virtualenvs: `.venv` (base pipeline: pillow, pikepdf, numpy,
playwright) and `.venv-reletter` (opencv-python-headless, pillow, numpy,
fonttools). GIMP batch scripts run under GIMP 2.10's python-fu =
**Python 2**.

## Base pipeline (`process.py` + GIMP compose plugin)

1. **upscale** — Real-ESRGAN (`realesr-animevideov3`) produces each page at
   the exact size GIMP will place it. Story pages are cropped to their
   content box and fitted to the story box (which reserves the page-number
   strip); covers fit the trim; index pages fit the full canvas; landscape
   spreads are upscaled whole (seam consistency) and split at the detected
   gutter into `-1`/`-2` halves.
2. **compose** — the GIMP plugin (`webcomics_compose.py`, versioned copy in
   `gimp-plugin/`) opens `template.xcf` (1875×2775 px @300 dpi = 6.25×9.25"
   with bleed), places the page layer **1:1, centered** (pages arrive
   pre-sized; never upscaled again), paints a background layer in the
   page's paper color, draws the page number, saves the XCF and exports a
   300 dpi raster PDF. `postprocess_pdf` then recompresses (JPEG) and tags
   sRGB. The 1:1 centered placement means page pixel (x, y) lands at canvas
   `((1875 − W)//2 + x, (2775 − H)//2 + y)` — used by QA to overlay renders
   on source pages exactly.
3. **merge** — concatenates the page PDFs, inserting white pages so covers
   land on the right, index/credits pages on the left, and spread halves on
   facing pages.

## Re-lettering subsystem (`relettering/`)

Re-typesets all dialogue with the user-supplied comic font
(`relettering/fonts/regular.ttf` + `bolditalic.ttf`; GIMP family names in
`fonts/fonts.json`) at a book-wide uniform size (the "book size": the 35th percentile of every bubble's maximum
fitting size — most bubbles carry the same size, only the densest shrink),
as **editable text layers** in the XCFs. Every script takes the comic
folder name as `argv[1]`; `reletter_gimp.py` reads the `RELETTER_COMIC`
env var and runs from the repo root.

### Stage 1 — detection (`reletter_detect.py`)

Text-first, never shape-first (shape detection false-fires on armor and
smoke):

1. **Letters**: letter-sized dark blobs whose surrounding ring is light
   (`letter_mask`). Variants: white-on-black caption letters
   (`white_letter_mask`) and dark-on-tinted-box letters (the `tint` pass).
2. **Blocks**: letters dilate/close into paragraph blocks — the block box
   is the original text's envelope and is the anchor for everything later
   (ordering, transcripts, centering).
3. **Region**: the light component enclosing the block becomes the bubble
   mask, walked by `follow_lobe` (per-row single connected run, growth
   clamped, drift-cut) so leaks never paint rectangles over art. Leaked or
   edge components are re-segmented by `strict_bubble`: a flood of the free
   space around the block using the balloon's own dark **outline as the
   barrier** (morphologically closed to seal small gaps), with escalating
   retries — brightness-step barrier (outline-less boxes on pale fog),
   pure-white **gutter rows** as barrier (retry-only: unconditional gutter
   clipping cuts balloons that legitimately straddle panels), and a luma
   barrier (saturated stripe fills read as ink in the min channel).
4. **strict_bubble hardening** (each earned by a real failure):
   - acceptance allows text-dense balloons whose mask barely exceeds the
     block area, as long as it covers ≥85 % of the block's rows;
   - ring components with ≥10 % share are unioned and **bridged** straight
     through dark border rows (a balloon poking above a panel border splits
     its interior into stacked components; the bridge is a row union —
     an intersection can be empty and veto every line crossing it);
   - a **dark-row cut** removes mask beyond a mostly-dark full-width row
     outside the block (a border sealed into the mask by hole-fill);
   - a **neck cut** removes mask beyond the narrowest pinch outside the
     block when the profile re-expands past it (flood escaped through an
     outline gap; a balloon's own arc tip never re-expands).
5. **Compound balloons — separate entries, never one.** A second block in
   the same light component becomes its own entry (own mask, own transcript
   string); short utterances below the letter-count floor get their own
   lobe via the cluster pass. Overlapping entries are font-**grouped** and
   their masks partitioned by the merge stage.
6. **Mask refinement for grouped / multi-paragraph bubbles**: each lobe's
   mask is recomputed with a per-block `strict_bubble` flood (union splits
   carve flat tops and steal from siblings). Overlapping pair masks are
   then resolved in order:
   - **containment** (overlap ≥85 % of the clearly smaller mask): a lobe
     drawn in front keeps its body, the container cedes it;
   - **ink interface** (letters excluded from the measure): the side whose
     interface with the overlap crosses an inked arc is the *back* balloon
     and loses the whole overlap — the front keeps its full oval, where the
     original letterer centers;
   - **notch split** (ink on neither interface = genuinely joined shape):
     the union is cut straight at the shallowest column/row between the two
     blocks; each lobe keeps its whole side, repairing rows its own flood
     missed.
7. **Assembly**: entries sort by **text-block (y, x)** — stable across mask
   changes, so transcripts stay aligned. Row profiles (`rows`) are the
   inset (eroded) mask intervals, **extended with the original letters'
   per-row extent** inside the block (any pixel the original text occupied
   is usable by the new text) and with short pinch runs bridged. Outputs
   per page: `bubbles/<stem>.json` (kind, striped, strict, group,
   paragraphs, bbox, block, rows), a mask PNG and a 2× crop PNG per bubble.

Bubble kinds: `bubble` (masked balloon), `dark` (white-on-black caption →
white text), `tint` (dark text on tinted box → row-median clean, black
text), `open` (grown rectangle on light ground), `margin` (page-margin
note). Covers, index/credits and art pages are skipped (`is_story_page`).

### Stage 2 — transcription

`make_sheets.py` packs the bubble crops into sheets; transcription fills
`parts/sheet-NNN.json` with one string per bubble key (`"<short> bNN"`).
`merge_transcripts.py` assembles `transcripts.json` from all `parts/*.json`
in **sorted filename order — later files win**, sized by
`sheets/manifest.json` counts. Fix-up rounds therefore write full-page or
single-key overrides into `parts/zz-*.json` files (each new round sorts
after the previous) and bump the manifest count when detection changes a
page's bubble count.

Transcript conventions: verbatim text (typos included — corrections go in
`typo_fixes.json`), `*word*` = bold-italic, `\n` = hard line break (also
used to mirror the original letterer's line structure or hyphenation when
the automatic wrap can't), `\n\n` = paragraph (compound balloon), `""` =
leave the region untouched (SFX, display lettering, onomatopoeia
overflowing its bubble, art false positives — never cleaned, zero risk).

### Stage 3 — fit & clean (`reletter_fit.py`)

For every bubble with text, finds the largest size whose wrapped lines fit
the row profile, then wipes the original lettering.

**Sizing**: per-bubble maxima are scanned first; the **book size** =
35th-percentile of dialogue maxima. Caps: dialogue/dark/open/tint at
`max(book, original size)` (shouts keep their size); margin notes near
their original (cap-height ratio measured from the font, never hardcoded);
grouped lobes share their group's smallest size (`relax_group_caps` trades
1 px when a member then wraps in fewer lines, up to 3 px when a ≤4-word
utterance reaches a single line).

**Placement — the anchor ladder** (all anchors *hold* their vertical
target; the old clearance re-balancing re-centered geometrically on
asymmetric masks and dropped blocks a full line low):

1. the **original letterer's text center** (block-box center);
2. hybrid: visible-lobe x (inscribed-circle peak), letterer's y;
3. the lobe's visual center;
4. last rung — **per-line placement**: full row-span widths on *near-raw*
   mask rows (`rows_raw`: 3×3-eroded mask merged with the letter-extended
   rows — the assembly inset keeps 10-15 px/side the letterer demonstrably
   used), each line centered on the **original letterer's per-line axis**
   (`orig_axis_map`, measured from the original letter blobs), confined to
   the original text's vertical envelope (tight above — rows above the
   original's first line can be border/gutter zone).

Each rung runs a **minimal-deviation vertical search** (dy = 0, ±3 …
±0.9 line-height): an anchor that can't host a size exactly often can a
few px away, and a small slide beats falling to the next rung whose x sits
tens of px off the original axis. Two-paragraph masks with a real waist are
banded per lobe first (`find_waist`), falling through to the ladder when
banding fails; multi-paragraph wraps re-anchor each paragraph on the
original letterer's per-paragraph axis (`orig_para_axes`) or the band's
visual center. Line breaking is a DP that minimizes squared slack per line
("pyramid" look); each line finally centers on its axis clamped into its
own row span (`line_cx`).

**Per-bubble editorial overrides** — `relettering/<comic>/
layout_overrides.json`, keyed `"<short> bNN"`: `{"size": N}` pins the font
size (bypassing group caps), `{"anchor": "visual"}` forces the visual
center, `{"anchor": "lines"}` forces the per-line mode. The escape hatch
for taste calls no global rule should be bent around.

**Cleaning** (only bubbles that got a layout — a fit failure must never
blank a bubble):
- `bubble`: row-median repaint of **ink + halo only**, sampled from the
  mask's own fill outside the halo — flat-filling every mask pixel plateaus
  the fill and reads as a white rectangle wherever the mask has a straight
  edge. Plus an **overflow inpaint** for letters inside the block but
  outside any honest mask (the letterer spilling past the outline), which
  skips blobs touching the search window's border (those are clipped
  background wedges, not text).
- `dark` / `tint`: row-median wipe of the letters (bright cores dilated so
  anti-aliased rims go too — else hollow letter ghosts survive) keyed on
  min ≥ 150 (`dark`) or max ≤ 85 (`tint`), preserving the box fill.
- `open` / `margin`: inpaint of letter-sized dark blobs only, so outlines,
  borders and art are never touched.

The cleaned page is written **over** `upscaled/<comic>/<stem>.jpg`
(quality 95, 4:4:4) and the page's entries into `layout.json` (font size,
line height, per-line `y_top`/`cx`/width/styled runs, text color).

### Stage 4 — typesetting (`reletter_gimp.py`, GIMP batch)

Replaces each XCF's `relettering` layer group with fresh text layers
(group → `bubble-NN` → one text layer per line/style run, positioned from
`layout.json`), then re-exports the page PDF. Idempotent. Runs after
compose; the re-exported PDFs need `postprocess_pdf` again, then merge.

### Verification tooling

- **qa_scan.py** — pixel-diffs every cleaned page against
  `relettering/<comic>/pristine/` (pristine = untouched upscales,
  regenerated on demand from sources via `upscale_realesrgan`): every
  repaint footprint must be text-shaped (no blob with area > 4000 px and
  min-dimension > 35). Also eyeball diffs per affected bubble for blobs
  taller than a text line or wedge-shaped — small wedges slip the gate.
- **Leftover scan** — run detection on the *cleaned working pages*: any
  text block found where the transcript has real text = lettering that
  survived cleaning. The single most effective regression net.
- **Exact render overlay** — render a page PDF at 300 dpi and map it 1:1
  onto the source page via the centered-placement formula; imprecise
  fraction-based crops hide 10-20 px placement errors that reviewers see.

### Single-page rebuild loop (bug-fix rounds)

1. Regenerate the page's pristine upscale if missing (into `pristine/`).
2. Re-detect that page only (stage into a scratch dir, compare counts and
   blocks against the old JSON before promoting).
3. Remap transcripts: full-page keys into a new `parts/zz-*.json`, update
   the manifest count, re-merge.
4. Single-page fit with the **book size pinned** (a global refit would
   re-encode every working JPEG = generational loss); the driver must
   mirror `reletter_fit.main()` exactly.
5. Delete the page's `.xcf`/`.pdf`, compose, run the GIMP pass with
   `layout.json` temporarily pruned to the affected stems (restore after),
   `postprocess_pdf`, merge.
6. Verify: leftover scan + qa_scan + exact render overlay vs the source.
7. **Scope discipline**: snapshot `layout.json` before the round and diff
   the entries of already-approved pages afterwards — detection/fit
   improvements can silently change (or regress) pages the reviewer signed
   off; restore their entries verbatim if placement moved.

## Footguns

- **mtime trap**: compose skips a page iff its XCF is newer than the
  upscaled JPEG. Bulk-copying into `upscaled/` without `cp -p` makes
  compose rebuild *every* page — and a plain recompose wipes the
  `relettering` text groups (recovery = full GIMP re-letter pass).
- GIMP python-fu is Python 2: no f-strings, encode UTF-8 before `pdb` text
  calls. `gimp` is not on PATH — use the full app-bundle binary path.
- OpenCV morphology on a mask cropped to its bbox treats out-of-border as
  foreground: `np.pad` before close/flood-fill or corner voids fill into
  rectangles.
- Pages refit without re-detection keep their OLD masks; any detection
  improvement only reaches a page when it is re-detected.
- Fit drivers that import the scripts via importlib must set
  `sys.argv = [name, comic]` **before** `exec_module` (module-level argv
  check and workdir paths).
