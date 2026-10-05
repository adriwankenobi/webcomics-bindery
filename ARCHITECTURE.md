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

### Re-lettered book driver (`--relettering`)

`process.py <comic> all --relettering` (`cmd_reletter_book`, the one
extension inside `process.py`; helpers unit-tested in `tests/`) chains the
base pipeline with the `relettering/` scripts — each run under
`.venv-reletter` via subprocess, the GIMP pass via GIMP batch — as one
resumable run. There is no state file: every step is skipped by a marker
on disk, so the same command resumes after the pause or after a failure.

| step | skipped when | notes |
|---|---|---|
| preflight | — | fonts, `.venv-reletter`, GIMP present; refuses to start otherwise |
| upscale | existing rule | |
| pristine copies → `pristine/` | dir already holds the page; whole step once `layout.json` exists | `copy2` keeps mtimes; never overwrites |
| detect | `bubbles/*.json` exists | never re-run automatically: it could reorder bubbles under the transcripts |
| sheets | `sheets/manifest.json` exists | |
| **transcription** | every manifest bubble has a key in `parts/*.json` | else **pause**: prints the sheets/parts paths and the gap count; a terminal blocks on Enter and re-checks, otherwise exit 0 |
| merge transcripts | — | |
| fit + clean | `layout.json` exists | never re-run automatically (re-encodes every working page) |
| compose | existing mtime rule | cleaned pages are newer → recomposed |
| GIMP text pass | per page: `qa/<stem>.png` mtime ≥ `.xcf` mtime | see Stage 4 |
| postprocess | — | only the stems the pass reported `RELETTER DONE`, so no PDF is recompressed twice |
| merge | — | |
| qa_scan | — | report only |

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
   carve flat tops and steal from siblings). The refined mask is accepted
   when it still covers 85 % of the block's rows **and** is not truncated at
   an end of the block — the rows it misses there are given back from the
   mask it replaces, bounded to the lobe's own block, because a truncation
   inside the block puts a whole line of the original outside every mask and
   the cleaner is confined to the mask. Overlapping pair masks are then
   resolved in order:
   - **containment** (overlap ≥85 % of the clearly smaller mask): a lobe
     drawn in front keeps its body, the container cedes it;
   - **ink interface** (letters excluded from the measure): the side whose
     interface with the overlap crosses an inked arc is the *back* balloon
     and loses the whole overlap — the front keeps its full oval, where the
     original letterer centers;
   - **notch split** (ink on neither interface = genuinely joined shape):
     the union is cut straight at the shallowest column/row between the two
     blocks; each lobe keeps its whole side, repairing rows its own flood
     missed;
   - **seam split** (`seam_split`, when the two blocks overlap on *both*
     axes = diagonal lobes): no straight cut separates them and the notch
     window inverts, so every pixel goes to the lobe whose own *lettering*
     is nearer. The blocks are not usable here — a block that overlaps its
     sibling's is precisely one that swallowed the sibling's line.

   Every guard that protects "a lobe's own block" takes it as the block
   **minus the sibling's** (`own_block_mask`): where two blob clusters
   overlap, neither claim says whose lettering it is. Mask **debris** (a
   component under 2 % of the mask, left behind by a cede) is dropped before
   the final crop, or it holds the bbox open far past the balloon.
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

**Where the per-bubble preparation lives.** Everything that turns a
detection entry plus its mask into the inputs the fit *and* the cleaning
read — the `lobes` override substitution, the bite repair, caption-box
membership and `sole_box`, the `open`/`margin` balloon clip, the three row
profiles (`rows`, `rows_fit`, `rows_raw`), the `allowed` cleaning band and
the cleaning jobs — is `prepare_bubble`, with `page_caption_boxes`,
`boxes_to_wipe` and `is_strip` beside it. `main()` calls it, and so does
`qa-tools/reclean_all.py`, the tool that proves a cleaning change is safe
book-wide. That preparation used to be inline in `main()` and hand-mirrored
in `reclean_all.py` (four "mirror main()" blocks by the end), so each new
cleaning behaviour had to be copied across by hand or the verification
quietly stopped reproducing the pipeline. **Add cleaning behaviour to
`prepare_bubble`, never to a caller.**

**Mask repair** (`repair_letter_bites`, before anything reads the mask, so
the profile, the lobes and the cleaning all see the same balloon):
detection's barrier is the drawn ink **closed with a 7×7 kernel**, which
welds any letter within that distance of the outline to it. The letter stops
being an enclosed hole, so the hole-fill leaves a **bite** out of the mask —
and a text line that all but spans the balloon seals whole regions off the
interior. The fit re-floods with the *raw* ink as the only wall (the 2-3 px
the letterer left survives, the interior stays one component, the letters are
enclosed holes again) and adds back only what the mask and the ink **already
enclose**: a bite is a pocket, a leak is enclosed by nothing. The block bound
alone is not enough — the blob cluster reaches past the balloon's own arc, so
an escape onto pale sky lands *inside* it.

**Row profile** (`resolve_rows`): the balloon's own fill is the authority in
both directions — clip where the profile leaked past it, restore the width
where the profile sits further inside it than its own clearance (`rows` is
the mask eroded, so it carries the same bites, and the 6×6 erosion smears one
three rows past the letter that caused it).

**Writing area for `open` / `margin`.** These kinds have no balloon by
definition (the block is a bare rectangle over the artwork and the cleaning
is a letter-only inpaint) — but `open` is *also* detection's fallback when
its walled flood fails on a balloon that is drawn, and then nothing bounds
the new type at all. `unfound_balloon` floods out from the block; what comes
back is accepted as the writing area only when it is **narrower than the
block**, which is the proof that it is an interior (the block runs the width
of the lettering, so a real interior contains it, while an escaped flood
comes back wider). Three `open` entries exist book-wide and one is a
mis-detected balloon.

**A mask that leaked past recognition** can be replaced per-comic with
`{"lobes": [[x0,y0,x1,y1], …]}` in `layout_overrides.json` — one rectangle
per balloon, in paragraph order. `lobes_from_rects` re-floods each balloon's
interior inside its own rectangle, and the entry then skips every repair and
clip that reads the letterer's block (its block spans all the balloons). The
alternative — splitting the entry in detection — renumbers the page and
invalidates its positional transcripts.

Such an entry is also **cleaned one lobe at a time**: `lobes_from_rects`
returns a `(bbox, block, mask)` job per balloon, the block being the original
ink inside that interior. Cleaned as one entry the cleaner reads the entry's
own block — 836x407 across a panel of sky on p276 — and the strict ink term
repainted the artwork behind the left balloon. Every verification gate that
reads a mask PNG applies the override too (`gate_mask` in `leftover.py`,
`welded.py`, `faint_scan.py`), or it measures a region nothing ever cleaned:
`letter_mask` found 16 letter-shaped pieces of artwork inside that leaked
detection mask and the leftover scan read them as failed cleaning.

**Type metrics**: `cap_height` measures the cap glyph's own ink height from
the font at run time (`cap_ratio`, fontTools glyph bounds / unitsPerEm). It
used to be PIL's `getbbox("H")` — the layout box from the ascender origin,
rounded out to whole pixels — which claimed 22px where GIMP renders 19 and
jumped 3px between adjacent sizes. Since `old_cap_h` is measured off the
original letterer's pixels, the "is the new text at least as big as the
original?" gate was comparing a model against a measurement and could not
fail. Measured against the printed PDFs the ratio was 0.864 book-wide.

**Tracking**: the comic font is ~1.5x wider per unit of cap height than the
one the source edition used, so a tight balloon cannot hold the original's
cap height at natural width. `best_fit` therefore tries, at each size, the
LOOSEST tracking first and tightens only if it must (`track_steps`, capped
at 5% of the em — about an 8% narrower line). A balloon with room is never
tracked. The chosen value rides in the layout entry as `track` and the GIMP
pass sets it with `gimp_text_layer_set_letter_spacing`, so the XCF layer
stays a *text* layer and the glyphs keep their shapes; the width reduction
is exactly `spacing x (chars - 1)`, which is the model the fit wraps with.

**Sizing**: per-bubble maxima are scanned first; the **book size** =
35th-percentile of dialogue maxima. Caps: dialogue/dark/open/tint at
`max(book, original size)` (shouts keep their size); margin notes near
their original (cap-height ratio measured from the font, never hardcoded);
joined lobes size INDEPENDENTLY (`lobe_size_caps`), each filling its own
balloon — they used to share the group's smallest maximum, faithful to the
original letterer but leaving a pair at 18px beside neighbours at 23px;
the user chose (Sep 16 2026) to let them differ. `relax_group_caps` now only
runs for groups that still share one size. A compound balloon returned as a
single entry is raised per lobe by `grow_lobes` (its lines then carry their
own `font_px`/`line_height`/`track`, and the GIMP pass falls back to the
entry's when they do not) — give it the KIND CEILING, not the bubble's
`max_size`, which is the tightest lobe's and leaves no room to grow. With `--book` **pinned** the maxima scan
caps its search at `max(book, original size)` — every dialogue size is
clamped there anyway, so probing above it is waste; 7× faster and
byte-identical. Unpinned it must run free, since the book size *is* the
percentile of those maxima.

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

**Caption boxes are one block of type.** `caption_boxes` pairs a `tint`
strip with the `bubble` strip above or below it — the two halves of a
gradient-filled box, split where the fill crosses `LIGHT_MIN`. The pairing
requires their x ranges to **agree** (`X_AGREE`, intersection over union
0.85), not merely to overlap: the halves are cut from one rectangle, so
their extents match to a few px, while an oval balloon of similar width
sitting a little to one side passes an overlap test easily and then both
inflates the box for the fit *and* leaves it with a member nothing typesets,
which skips the wipe entirely. `stack_box_lines` then puts every line of the
box on **one line grid**, centred on the union of its members' blocks — the
strips are an artefact of the fill threshold and mean nothing to the layout,
and centring each in its own strip printed a blank line at the seam. And the
wrap runs `tight` for anything in a box: the 0.16-of-a-line-height overhang
`wrap_at` grants a balloon's first and last line (glyphs fill ~70% of the
line box, and an arc forgives the rest) is refused at a drawn frame, which
forgives nothing — 47px of box interior otherwise "fits" two 24px line
boxes and the type prints across the rules.

**Caption boxes found from their frame** (`frame_box`,
`auto_caption_boxes`, run per page before anything is prepared — the fit's
`collect_page` and `reclean_all.py` both call it). Detection splits a
gradient box wherever the fill crosses `LIGHT_MIN`, which yields more shapes
than the strip pair: one strip missing its pale lines, slivers side by side
with the top lines in none, one line cut in two. The drawn frame does not
care: a flood from the lettering walled only by true black returns a
rectangle inside a frame (ovals do not), and the box is then handled
exactly as a `box` override — one member carries the whole caption, the
others set nothing, the box is wiped whole, and its cap height is measured
on every letter inside the frame. It only fires where detection
demonstrably lost part of the box; hand overrides win (`override_for`
consults `AUTO_OVERRIDES` second), and strip pairs whose members are in a
framed box are dropped from `caption_boxes`' results.

**Balloons found from the lettering** (`letter_lobes`, in
`prepare_bubble`). Where the mask cannot give the balloons — two paragraphs
side by side in one entry, or a mask that leaked well past or stops well
short of the balloons the lettering is in — the letterer's own letters are
split into one group per paragraph (`letter_groups`, MST gaps), put in
comic reading order (`reading_order`: left before right when two share most
of their height; `mask_lobes` uses it too), and each balloon is the walled
flood out from its own group. Where floods meet (balloons joined with no
outline between them, including a neighbouring entry's), each pixel goes to
the nearest lettering measured INSIDE the fill (`geodesic_owner`). The
result is used like a `lobes` override (per-lobe rows, per-lobe cleaning
jobs), and `lobe_bands` prefers it to the waist cut.

**Per-bubble editorial overrides** — `relettering/<comic>/
layout_overrides.json`, keyed `"<short> bNN"`: `{"size": N}` pins the font
size (bypassing group caps), `{"anchor": "visual"}` forces the visual
center, `{"anchor": "lines"}` forces the per-line mode, `{"box": [x0,y0,
x1,y1]}` names a caption box detection only half-found, and `{"lobes":
[[x0,y0,x1,y1], …]}` replaces a mask that leaked past recognition. The
escape hatch for taste calls no global rule should be bent around.

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
- coloured **caption boxes** are wiped whole (`clean_caption_box`), *in place
  of* their member entries, with the repaint confined inside the box's drawn
  frame (`frame_bounds`: near-solid dark rows/columns at the box's edges).
  The frame is found on the MAX channel — a yellow fill has a low blue
  channel, so the min channel calls the whole box dark and no frame can be
  picked out of it — and the guard self-disables on a white-on-black caption,
  where every row is "solid" and there is no frame to find. Without it the
  repaint wiped the frame and left a tab of fill colour on the artwork. The fill under a word is estimated by
  **interpolating across the ink** from the fill either side of it, per row,
  and down the column for a row the lettering fills end to end. A
  morphological close is right for *finding* the ink and wrong for replacing
  it: it takes the brightest value in a window that must be as wide as a
  whole word, so on a box whose fill grades along the row it repaints the
  line as a flat, too-bright slab with straight edges. The fill is measured
  on `~(ink | halo)` — never on pixels merely spared from the repaint, or the
  frame drags the estimate dark beside it.
- `open` / `margin`: inpaint of letter-sized dark blobs only, so outlines,
  borders and art are never touched.

The cleaned page is written **over** `upscaled/<comic>/<stem>.jpg`
(quality 95, 4:4:4) and the page's entries into `layout.json` (font size,
line height, per-line `y_top`/`cx`/width/styled runs, text color).

### Stage 4 — typesetting (`reletter_gimp.py`, GIMP batch)

Replaces each XCF's `relettering` layer group with fresh text layers
(group → `bubble-NN` → one text layer per line/style run, positioned from
`layout.json`), then re-exports the page PDF and a half-size QA PNG into
`qa/` (created if missing). Runs after compose; the re-exported PDFs need
`postprocess_pdf` again, then merge.

**Done marker.** The QA PNG is written last, so a page is skipped
(`RELETTER SKIP <stem>`) when `qa/<stem>.png` is at least as new as its
`.xcf`. Compose recomposing a page makes the XCF newer and the page is
re-lettered on the next pass; a `layout.json`-only change does not — delete
the page's PNG to force it. This mirrors compose's own mtime rule.

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
- **Faint-residue scan** — letter-shaped specks left *above* the ink
  cut-off, which `letter_mask` (a "dark text on light paper" ring test) is
  structurally blind to. This is what readers call "little almost white
  objects", and at its worst it is a whole paragraph still legible as a grey
  ghost under the new text.
- **Size metrics** — `new_cap_h < old_cap_h` says the balloon demonstrably
  had room for more than the fit took; dialogue set well under the book size
  is the wider population those come from. Both read straight off
  `layout.json`, so they cost nothing and find "text too small" before a
  reviewer does.

**A subset promote ships the layout fix but not the cleaning fix.**
`promote.py` only compares the pages the subset re-fit staged, so a change
to `clean_bubble` never reaches the rest of the book: after round 4, 209
pages still carried round-3 pixels. Re-clean the whole book
(`reclean_all.py`, ~1.4 min, writes to a scratch dir and touches nothing)
whenever the cleaning changes, and gate it with the faint-residue scan.

### Single-page rebuild loop (bug-fix rounds)

1. Regenerate the page's pristine upscale if missing (into `pristine/`).
2. Re-detect that page only (stage into a scratch dir, compare counts and
   blocks against the old JSON before promoting).
3. Remap transcripts: full-page keys into a new `parts/zz-*.json`, update
   the manifest count, re-merge.
4. Single-page fit with the **book size pinned** — `reletter_fit.py
   "<comic>" --book N` (a global refit would re-encode every working JPEG =
   generational loss, and an unpinned re-fit can raise the common size and
   move every already-approved page); the driver must mirror
   `reletter_fit.main()` exactly. For a whole-book algorithm fix, point
   `PAGES_DIR` at a scratch copy of `pristine/`, run the real `main()` with
   `--book` at the shipped size, then promote ONLY the pages whose layout
   entries or cleaned pixels actually changed — untouched pages keep their
   old mtime, which is what stops compose from rebuilding the whole book.
5. Delete the page's `.xcf`/`.pdf`, compose, run the GIMP pass — the
   recomposed XCF is now newer than its `qa/<stem>.png`, so the pass
   re-letters only that page (pruning `layout.json` is no longer needed) —
   then `postprocess_pdf` on that PDF, merge.
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
- The `--relettering` driver never re-runs detection or the fit once their
  outputs exist — that is deliberate (bubble order under the transcripts;
  generational loss). Fix pages with the single-page loop, not by deleting
  `bubbles/` or `layout.json` wholesale.
- Fit drivers that import the scripts via importlib must set
  `sys.argv = [name, comic]` **before** `exec_module` (module-level argv
  check and workdir paths).
