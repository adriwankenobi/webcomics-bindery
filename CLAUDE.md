# CLAUDE.md

Print pipeline for webcomics (language-agnostic for LTR, space-separated,
Latin-script-like languages; all text lives in per-comic data). README.md =
the flow, ARCHITECTURE.md = how everything works inside. Key facts a
session needs:

## Ground rules
- Sources live in the folder named after the comic. NEVER modify sources.
  Variants live only in the layer folders (upscaled/, xcf/, pdf/,
  relettering/) — do not duplicate the source folder under another name.
- `template.xcf` is the print template every page is composed onto (pages
  are placed 1:1, centered on the 1875x2775 canvas — exact overlays for QA).
- process.py, the justfile and the GIMP compose plugin
  (~/Library/Application Support/GIMP/2.10/plug-ins/webcomics_compose.py,
  versioned copy in gimp-plugin/) are the stable base pipeline — extend
  around them, don't edit the base steps. The one extension inside
  process.py is the `--relettering` driver (`cmd_reletter_book` + helpers,
  unit-tested in tests/, `just test`). The re-lettering scripts ARE the
  place for algorithm fixes; prefer code over hand-edited masks
  (reproducibility).
- Re-lettering code lives in relettering/ (reletter_detect/fit/gimp.py,
  qa_scan, merge_transcripts, make_sheets). Every
  script takes the comic name as argv[1] (reletter_gimp.py: RELETTER_COMIC
  env var, run from the repo root). Git tracks ONLY the process (whitelist
  .gitignore); all comic data — sources, upscaled/, xcf/, pdf/,
  relettering/<comic>/ (transcripts included) — is untracked and must be
  reproducible from the source folder alone. No comic-specific content in
  tracked files. Per-comic editorial data = untracked JSONs in
  relettering/<comic>/: typo_fixes.json, layout_overrides.json
  ({"size": N} | {"anchor": "visual"} | {"anchor": "lines"} | {"box":
  [x0,y0,x1,y1]} | {"lobes": [[x0,y0,x1,y1], ...]}), fit_policy.json
  (per-book typographic calls an approved book must not inherit:
  {"frame_pairs": true} = a split caption pair in a frame is one box),
  parts/zz-*.json
  transcript overrides (merge order = sorted filename, later wins; keep
  sheets/manifest.json counts in sync).
- The QA gates and rebuild drivers in relettering/qa-tools/ are PROCESS and
  are tracked (whitelisted by name in .gitignore); the eyeball helpers
  beside them (crop/zoom/montage/...) are scratch and stay untracked. They
  take the comic as argv[1] like everything else, and the two that re-run
  the fit (rebuild_fit.py, refit_subset.py) REQUIRE BOOK=<size> — an
  unpinned re-fit re-derives the book size and moves approved pages.
- `just process "<comic>" --relettering` = the whole re-lettered book in
  one resumable run; it pauses once (`== TRANSCRIPTION NEEDED ==`, Enter or
  re-run to resume). Markers: pristine/ (skipped once layout.json exists),
  bubbles/*.json, sheets/manifest.json, all manifest keys in parts/,
  layout.json — detection and the fit are NEVER re-run by the driver — and
  for the GIMP text pass qa/<stem>.png newer than the .xcf (delete the PNG
  to force one page; a recompose forces it too, so the single-page recipe
  no longer prunes layout.json). Postprocess runs only on the stems the
  pass reported RELETTER DONE.
- Steps are resumable: outputs are skipped when they exist. MTIME TRAP:
  compose skips iff xcf mtime >= upscaled mtime — bulk copies into
  upscaled/ must preserve mtimes (`cp -p`) or EVERY page recomposes and
  loses its "relettering" text group (recovery = full GIMP re-letter pass).
  To force one page: delete its .xcf/.pdf.
- Bug-fix rounds: single-page rebuild only (recipe in ARCHITECTURE.md);
  never a global refit (re-encodes every working JPEG). Snapshot
  layout.json before a round and diff already-approved pages after —
  detection/fit improvements can silently move approved pages; restore
  their entries verbatim.

## Environments
- Base pipeline: `.venv/bin/python` (pillow, pikepdf, numpy, playwright).
- Upscaler: untracked third-party binary at
  tools/realesrgan/realesrgan-ncnn-vulkan + models/ (install steps in
  README "Setup"). Pass `-m <models dir>` by hand or it exits 1 silently.
- Re-lettering: `.venv-reletter/bin/python` (opencv-python-headless, pillow,
  numpy, fonttools). cv2 is ONLY here.
- GIMP batch = GIMP 2.10 python-fu = **Python 2**: no f-strings, encode
  UTF-8 before pdb text calls. `gimp` is not on PATH — use
  /Applications/GIMP-2.10.app/Contents/MacOS/gimp.
- Fonts are USER-SUPPLIED and LICENSED — never commit font files. The fit
  reads relettering/fonts/regular.ttf + bolditalic.ttf; the same font must
  be installed system-wide (~/Library/Fonts) for GIMP, whose family names
  come from relettering/fonts/fonts.json (defaults: "CC WildWords Lower" /
  "CC WildWords Lower Bold Italic"). Setup instructions the user must
  follow: relettering/fonts/README.md. Measure cap-height/metrics from the
  font at run time, never hardcode ratios.

## Re-lettering gotchas (hard-won)
- Detection is TEXT-FIRST (letter blobs → paragraph clusters → enclosing
  light region). Shape-first detection false-fires on armor/smoke.
- np.pad masks before cv2 morphology close + hole-fill: OpenCV treats
  out-of-border as foreground and turns corner voids into filled RECTANGLES.
- Never fill row intervals across disconnected runs; follow the single
  connected run (follow_lobe). Balloons crossing a panel border need the
  ring-component union + border-row bridge (bridge = row UNION, never
  intersection) inside strict_bubble; leaks get the dark-row cut and the
  neck cut.
- Joined/overlapping balloons are SEPARATE entries, font-grouped; their
  masks resolve as: containment (front lobe keeps its body) → ink-interface
  test with letters EXCLUDED (front balloon keeps its full oval) → notch
  split of the union at the shallowest column/row between the blocks.
  Never a plain midline split (flat tops, stolen sibling area).
- Cleaning: bubble = row-median repaint of ink+halo ONLY (flat-filling the
  whole mask plateaus → reads as a white rectangle at straight mask edges)
  + overflow inpaint that skips blobs touching the window border; dark/tint
  captions = row-median wipe of dilated letter cores (else hollow ghosts);
  open/margin = letter-only inpaint. Kinds: bubble/dark/tint/open/margin
  (tint = dark text on tinted box, text stays black).
- THE ORIGINAL LETTERING CHEWS THE MASK. Detection walls its flood with the
  drawn ink CLOSED by a 7x7 kernel (to seal outline gaps), so a letter within
  ~7px of the outline is WELDED to it: no longer an enclosed hole, so the
  hole-fill leaves a BITE, and a line that all but spans the balloon seals
  whole regions off the interior (p65 b08: the fit saw 127px of a 252px
  balloon and dropped 4 size steps; p88 b06 landed BELOW the original's own
  size). Both user-visible halves come from this one fault — small text that
  does not fill the balloon, AND the letters inside the bite never being
  cleaned (the cleaner is confined to the mask), which read as "little almost
  white objects". `repair_letter_bites` re-floods with the RAW ink as the
  only wall and adds back only what the mask and the ink ALREADY ENCLOSE (a
  bite is a pocket; a leak is enclosed by nothing). Bounding it to the
  letterer's block is NOT enough on its own: the blob cluster reaches past
  the balloon's own arc, so p104's escape onto pale sky landed INSIDE the
  block — the cleaner then read the fill off sky, painted blue over the
  artwork and left the old lettering standing. An area ratio cannot see it
  either: that escape was 1.7% of the mask.
- The fill is the authority on the row profile in BOTH directions
  (`resolve_rows`): clip where the profile leaked past it, and give the rows
  their width back where the profile sits further inside it than its own
  clearance. `rows` is the mask ERODED, so it carries every bite, detection's
  letter-union repair only reaches rows that hold letter pixels, and the 6x6
  erosion smears a bite three rows past the letter that caused it. Those
  rows are what a line box lands on and `rows_span` takes the NARROWEST row
  of the band, so two chewed rows cost p88 b06 three size steps.
- `clean_caption_box` estimates the fill under a word by INTERPOLATING across
  it from the fill either side, per row. A morphological close is right for
  FINDING the ink (it is only a threshold) and wrong for replacing it: a
  close takes the BRIGHTEST value in its window, and the window has to be as
  wide as a whole word, so on a box whose fill grades along the row it reads
  the bright end as the fill everywhere and repaints the line as a flat,
  too-bright slab with straight edges ("yellow square very bad, rectangle",
  p42, 20-45 levels bright). Measure the fill on `~(ink | halo)` — never on
  pixels merely excluded from the repaint, or the spared frame drags the
  estimate 100 levels dark beside it.
- With `--book` PINNED the sizing pass may cap its search at
  max(book, the original's own size) — every dialogue size is clamped there
  anyway. 7x faster and byte-identical (verified). Unpinned it must still run
  free: the book size IS the percentile of those maxima.
- The bubble repaint must protect the OUTLINE without stranding lettering —
  both halves matter and it took two rounds to get right. Masks leak (a tail
  that leaves the crop window floods the white page; a balloon on pale sky
  has nothing to stop the fill), so everything dark in the mask cannot just
  be wiped — that erased the outline and the balloon vanished, leaving the
  new text on bare art (225 bubbles). But "letters = dark islands ENCLOSED
  by the fill" is ALSO wrong: a joined-balloon split truncates the mask
  mid-text and letters straddling that edge are not enclosed, so they
  survive — 341 stranded letters on 81 pages, which read as "wild letters"
  and as the new text sitting on top of the old. The rule that satisfies
  both: wipe every dark pixel the mask covers EXCEPT long thin CURVES
  (extent >= 0.45 * bbox span and area <= 0.28 * its own bbox = an outline
  or a panel line; lettering is compact even when several letters merge),
  plus the enclosed islands. Classify on the WHOLE region — the mask
  boundary chops the outline into short arcs that stop looking like curves.
  Never size-cap what counts as lettering.
- The ink cut-off adapts to each balloon's own fill (`ref - 60`, capped at
  DARK_MAX+40): a tinted caption (yellow, blue) sits near the global
  cut-off, so a fixed threshold split its fill into bands and spared the
  letters. The WALL is found with a stronger threshold (DARK_MAX-20) so the
  fill can never fragment the interior.
- One long word must never shrink a bubble: `hyphenate`/`split_word` break
  it (prefer a vowel|consonant point, ≥2 chars a side, reuse an existing
  hyphen). Cap the fragments (MAX_PIECES) and return the words UNTOUCHED
  past it — at the top of the size search every word is too wide, and
  shattering them all runs the line-break DP over fragments for minutes.
- A compound balloon carries ONE PARAGRAPH PER LOBE. `find_bands` cuts the
  row-width profile for stacked lobes; lobes side by side (a diagonal run)
  have no waist — a row span is a single x-range — so `mask_lobes` falls
  back to distance-transform peaks with nearest-peak assignment. Missing
  this crammed 3 paragraphs into one lobe at 12px with two lobes empty.
- The letterer's block is THE anchor, so clip it (`block_anchor`) to the
  balloon's rows AND its bbox: the blob cluster swallows artwork beside the
  balloon (fur, a panel line) and slid text clean out of the lobe.
- Keep the new lettering clear of the drawn outline (`inset_rows`,
  ~3.5% of the profile's extent) — the row profile runs to the mask edge and
  the mask includes the outline stroke, so type set at the book size lands
  ON the line. Apply it to the FIT ONLY: inset the CLEANING and rim letters
  survive.
- Where a mask LEAKED past the balloon, clip the fit's row profile (and the
  last rung's `rows_raw`) to the balloon's own fill — p5's profile was a
  417px-wide rectangle for a 264px oval, so lines were set where the oval
  had long since narrowed.
- Overlap between two balloons is ceded whole to one of them; a balloon must
  never cede pixels inside its OWN text block, or its lettering is left
  uncleaned AND the fit has no room (p101 stuck at 15px vs 21px). But "its
  OWN" must exclude whatever the SIBLING's block claims (`own_block_mask`):
  a block is a blob cluster, and p294 b07's reached over b12's first line —
  so the guard held b12's "SI DESEAS" inside b07's mask and with it the
  slice of b12's balloon b07's flood had filled, and b07's own last line was
  cut away instead. Where two blocks collide neither claim is credible.
- Joined lobes are stacked or side by side — EXCEPT when their blocks
  overlap on BOTH axes, which means they are DIAGONAL and no straight cut
  separates them. The row window then runs BACKWARDS (from the upper's block
  bottom UP to the lower's block top) and the cut lands inside the upper's
  own lettering: p294's fell at y1954, the mask lost that line, the cleaner
  is confined to the mask so it survived, and the profile kept the part of
  the sibling's balloon the flood had filled (the last line was set 274px
  wide in a 220px balloon). `seam_split` assigns each pixel to the lobe
  whose own LETTERING is nearer — never the block, which is the polluted
  thing. Only 3 pairs book-wide take it. Clamp the straight cuts out of the
  upper/left lobe's own block too, but SKIP a clamped cut that would leave a
  lobe with under a quarter of its mask.
- The refine pass (`strict_bubble` per lobe) may miss 15% of the block's
  rows, but not as a TRUNCATION at an end of it: p149 b08's flood stopped
  12px above its block's bottom — 89% covered, so accepted — and the mask
  lost the last line, "SU PODER.", which the reader saw standing under the
  new text. Give those rows back from the mask being replaced, bounded to
  the lobe's own block. And drop mask DEBRIS before cropping: ceding a
  sibling's oval left p294 b07 two slivers of that balloon's rim, 1.4% and
  0.4% of its mask, which held its bbox 130px taller than its own balloon.
- Detection changes are only safe while the bubble COUNT per page holds —
  the transcripts are positional. Re-detect into a scratch compare first and
  REFUSE any page whose count moved. Widening `letter_mask` to accept tinted
  fill (for gradient caption boxes) creates entries (12->15 on one page), so
  it cannot be done without re-transcribing those pages.
- The overflow inpaint's window is the bubble's OWN bbox, never the raw
  `block`: a blob cluster swallows the balloon next door, and the step then
  erased a neighbour's lettering that nothing types back — p78 and p234
  shipped with an EMPTY balloon. Measured book-wide, reaching past the bbox
  bought 3 stray blobs (<=13px out) and cost 45 of a neighbour's letters and
  line art; all 1704 blobs it legitimately cleans are inside the bbox. No
  margin helps — the balloons are adjacent.
- A coloured caption box with a gradient fill comes back as TWO stacked
  entries: a `tint` strip plus a `bubble` strip, split where the fill crosses
  LIGHT_MIN, each with a RAGGED mask that follows the fill instead of the
  box. 28 boxes on 19 pages. Clean it as ONE box (`caption_boxes` ->
  `clean_caption_box`) INSTEAD of its members — never after them, or the box
  pass reads their whitened strip as the fill level and paints a white band
  right across it. The transcripts are already complete (one half per entry),
  so this needs NO re-transcription. `{"box": [x0,y0,x1,y1]}` in
  layout_overrides.json covers a box detection half-masked without splitting.
- The two strips of one box must AGREE on their x range (`X_AGREE`, IoU
  0.85), not merely overlap: they are cut from ONE rectangle by a horizontal
  fill threshold, so their extents match to a few px, while an oval of
  similar width sitting a little to one side passes an overlap test easily.
  p164's 257x86 balloon two panels up joined a caption box and
  ONE fault produced both halves the user saw — the box grew 233x73 to
  277x179 so the fit had 265px of writing width in a 230px box (both lines
  printed past the frame), AND the box now had a member nothing typesets, so
  the wipe skipped the whole box and its strips were cleaned as separate
  entries (a flat yellow slab, a white band at the seam, a tab of fill on
  the artwork). Book-wide the 31 real pairs run 0.893-0.998 and the two
  false ones sit at 0.716 and 0.769; the 3.0 strip-shape ratio does NOT
  separate them (p164's false member is 2.99, two real tint halves are 2.25
  and 2.49).
- A caption box is ONE BLOCK OF TYPE (`stack_box_lines`). The strips are an
  artefact of the fill crossing LIGHT_MIN and carry no meaning for the
  layout, but each was centred in its own strip — so the box printed with a
  blank line at the seam: "too much space between paragraphs, as if it were
  2 bubbles not one" (p147's bottom-left box, p219). Put every line of the
  box on one grid centred on the union of its members' blocks. A `sole_box`
  entry already spans the whole box and is left alone.
- A drawn caption FRAME gets NO overhang (`wrap_at(..., tight=True)` for any
  entry in a box). `wrap_at` lets the first and last line hang 0.16 of a line
  height past their rows, because glyphs fill ~70% of the line box and a
  balloon's arc forgives it; a frame forgives nothing, the fill IS the
  writing area. p214's recovered box is 47px of interior and two 24px line
  boxes "fit" it — the type printed across the top and bottom rules. Also:
  a `box` override's rect is the frame's INNER edge (boxrect.py's
  convention), never the outer frame. Measured on the outside, that box read
  11px taller than it is and the size came out a step and a half too big.
- `clean_caption_box` estimates the fill LOCALLY (horizontal grayscale close,
  then paint from that same estimate per channel). A per-row level reads a
  highlight as the fill and everything yellow as ink, and paints the row
  WHITE. Keying on shape alone (a rectangular mask, a drawn frame) is not
  safe: both also match oval balloons and leaked masks.
- `bubble_interior` walls the flood on a COLOUR change as well as on dark
  strokes (COLOUR_WALL). Where a balloon sits on pale sky and its outline has
  a gap, the fill escapes and the "interior" becomes balloon + page — which
  silently defeats clipping the fit's row profile to that interior (p65 b07
  kept 283 of 311 rows and set two lines above the balloon's top arc).
- Row spans follow ONE connected run (`widest_run`), never first-to-last.
  Low in a balloon the fill splits into the body and the mouth of the tail;
  spanning that gap let the wrap set its last lines out into the tail, hard
  against the outline (p127's "text outside the bubble").
- `parse_runs` resolves style PER CHARACTER, then splits on whitespace. A
  regex alternation cannot: `\S+` swallows `¡--*WORD*` whole and the book
  printed literal asterisks (5 bubbles).
- The ink cut-off leaves the lettering's PALE strokes (150-220 on a 250 fill)
  behind as faint letter-shaped specks — 1428 bubbles on 263 pages, visible
  at 1:1 300dpi and invisible to letter_mask. A softer cut-off fixes it but
  ONLY inside the letterer's block AND inside the balloon's own fill
  (`solid`): bounded to the block alone it caught artwork through a leaked
  mask and painted white rectangles (17 artifact blobs vs 2).
- Verify EVERY batch, and NEVER substitute a proxy metric for these — a
  layout-geometry check passed while 341 stranded letters shipped:
  (1) leftover scan = re-run detection's own `letter_mask` on the CLEANED
  working pages over the mask UNION the block UNION every laid-out line's
  box; any letter blob where the transcript has text = failed cleaning
  (healthy baseline: <20 book-wide). Intersecting with the MASK ALONE hides
  the worst failure there is — lettering the mask never covered is exactly
  what puts new text on top of old — and reported 14 blobs "healthy" while
  p170 and p147 carried whole lines of old text under the new (widened: 51).
  It is still blind to text on a COLOURED fill and to faint residue; (2) qa_scan
  pixel-diff vs relettering/<comic>/pristine/ — this is the gate that catches
  a cleaning change repainting ART (it found 17 white rectangles against a
  shipped baseline of 2); `protrude.py` cannot stand in for it, since it
  measures against the MASK and a leaked mask is the very thing that puts
  text outside the balloon (it flagged none of p45/p65/p127) (text-shaped footprints only;
  also eyeball blobs taller than a text line — wedges slip the 4000/35
  gate, and an erased balloon outline is LOW-area but wide, so gate on
  min-dim ≥45 / area ≥400 to catch it); (3) EXACT 1:1 300dpi render overlays vs the source — fraction
  crops hide 10-20px placement errors the reviewer sees.
- A SUBSET promote ships the layout fix but NOT the cleaning fix. `promote.py`
  only ever looks at the pages the subset re-fit staged, so a cleaning change
  never reaches the rest: after round 4, 209 pages still carried round-3
  pixels, and p168 shipped with its whole first paragraph legible as a grey
  ghost under the new text. Cleaning changes need a full re-clean
  (`reclean_all.py`, 1.4 min) — measured with `faint_scan.py`, which the
  leftover scan cannot see (8466 specks shipped vs 5814 re-cleaned).
- THE PER-BUBBLE PREP HAS ONE HOME: `prepare_bubble` (with
  `page_caption_boxes` / `boxes_to_wipe` / `is_strip`). The fit's `main()`
  and `reclean_all.py` both call it. reclean_all used to hand-mirror that
  preparation — four "mirror main()" blocks by the time it was extracted —
  so every new cleaning behaviour (the `lobes` override, sole_box,
  unfound_balloon) had to be copied there by hand or the book-wide
  verification silently stopped reproducing the pipeline. Add cleaning
  behaviour THERE, never in a caller.
- MEASURE THE PDF, NOT THE PLAN. `cap_height` used to return PIL's
  `getbbox("H")` — the layout box from the ascender origin, rounded out to
  whole px — while `old_cap_h` is measured off the ORIGINAL'S PIXELS. At 23px
  it claimed 22 where GIMP renders 19 (printed/predicted = 0.864 book-wide),
  so `new_cap_h >= old_cap_h` COULD NOT FAIL: 0 of 56 sampled bubbles looked
  small while 5 printed smaller than the original, and four rounds of
  "verified fixed" reached a reader who could still see it. cap_height now
  measures the cap glyph's own ink height from the font at run time
  (fontTools, x size / unitsPerEm). Any metric computed from a MODEL has to
  be checked against the RENDERED artefact once, or it certifies the model.
- Our font is ~1.5x WIDER per unit of cap height than the original edition's
  (p88 b06: the original's "PUEDO..." is 57px at cap 17; ours needs 90px in a
  90px balloon). So a tight balloon cannot hold the original's cap height at
  natural width, and every "text small, not occupying the whole bubble" the
  user reported was width-bound within a pixel or two. Two levers, both the
  USER'S POLICY CALLS (Sep 16 2026, both approved):
  (1) joined lobes size INDEPENDENTLY (`lobe_size_caps`) — each fills its own
  balloon, even though the original letterer used one size per joined pair;
  (2) TRACKING up to 5% of the em (`track_steps`), tried loosest-first at
  each size in `best_fit`, so a roomy balloon is never tracked. GIMP applies
  it with `gimp_text_layer_set_letter_spacing`, which KEEPS the layer a text
  layer (editable, glyphs undistorted — scaling loses both); the width
  reduction is exactly `spacing x (chars - 1)`, verified in GIMP.
  Do NOT retry the ink-width-vs-advance idea: this font's side bearings are
  ~0, and the 4% a GIMP probe seems to show is the alpha threshold clipping
  the anti-aliased fringe.
- A compound balloon detection returned as ONE entry shares one size set by
  its TIGHTEST lobe. `grow_lobes` raises each lobe on its own and the lines
  carry their own `font_px`/`line_height`/`track` (GIMP falls back to the
  entry's). Pass it the KIND CEILING, never the bubble's `max_size` — that
  IS the tightest lobe's, so there is no room to grow.
- `open`/`margin` is also detection's FALLBACK when its walled flood fails
  on a balloon that IS drawn, and then nothing bounds the new type: p242's
  shout was set 141px wide in a 132px oval and printed across the
  outline on both sides. Its block could not catch it (the blob cluster had
  swallowed both arcs, so the block was 148px) and neither could the size
  ceiling (those arcs carried the measured cap height from ~25px to 36px,
  which is 9 size steps). `unfound_balloon` floods out from the block, and
  the proof that what comes back IS a balloon is that it is NARROWER than
  the block — the block runs the width of the lettering, so a real interior
  contains it, and an escaped flood comes back wider (p222 b04's ran down
  the tail: 194px for a 132px block, correctly refused). Do NOT try to get
  this from the ink components instead: dropping the ones that cross the
  block border by border also drops a letter WELDED to the outline (p182's
  "S", p222's "!"), and keeping them lets artwork in.
- `{"lobes": [[x0,y0,x1,y1], ...]}` in layout_overrides.json is the escape
  hatch for a mask that leaked past all recognition: p276 b04 is TWO
  balloons on pale sky returned as ONE 1269x528 entry whose mask covers 79%
  of the panel — mask_lobes sees one blob, the fit places nothing, cleaning
  it would repaint sky, and splitting the entry in detection would renumber
  the page and invalidate its positional transcripts. One rectangle per
  balloon, IN PARAGRAPH ORDER, each roughly centred on its balloon; each is
  re-flooded inside itself (`lobes_from_rects`) and the entry then bypasses
  every block-driven repair, because its block spans both balloons. CLEAN IT
  PER LOBE — `lobes_from_rects` hands back one job per balloon, each with the
  original ink inside that interior as its block. Cleaned as ONE entry the
  cleaner reads the entry's own block (836x407 over a panel of sky) and the
  strict ink term repainted the green sphere and the darts behind p276's
  left balloon; per lobe it changes 0 pixels outside the rectangles. EVERY
  gate that reads the mask PNG must apply the override too (`gate_mask` in
  leftover/welded/faint_scan; outside.py floods its own fill, so it takes
  the per-lobe interiors instead — and in the PAGE frame, since these lobes
  reach far past the entry's bbox: p231 b01's is a 72px slice across two
  balloons 292px tall). Otherwise the gate measures a region nothing ever
  cleaned: letter_mask found 16 letter-shaped pieces of ARTWORK in p276
  b04's leaked mask, and outside.py called both `lobes` entries the two
  worst in the book while their text is demonstrably inside their balloons.
- A CAPTION IS CENTRED ON THE LETTERER'S BLOCK, like a balloon. `tint`/
  `dark`/`open` used the width-weighted centroid of their own row profile,
  and `caption_boxes` finds a box by its FILL — so on a yellow->white graded
  box the rectangle is only the coloured part and its centre sits ABOVE where
  the letterer set the type. p42: box centre 175.5 vs the letterer's 181.5,
  printed ink on rows 167-183 against the original's 174-189, and the 7px of
  flat colour left under it read as "yellow square very bad, rectangle" for
  three rounds — the FILL was reconstructed to within 7 levels, the TYPE was
  in the wrong place. The non-bubble `wrap_at` call must also pass
  `hold_cy=True`: without it wrap_at re-balances and re-centres the block
  geometrically in the rows, throwing away the anchor you just measured.
  Every caption box in the book was set high (30 of 39 entries moved down
  4-11px).
- A caption box's drawn FRAME is found on `region.max(axis=2)`, never on the
  min channel: a yellow fill has a low BLUE channel, so min() called 93.9% of
  p42's box "dark" and no frame could be picked out of it. `frame_bounds`
  confines the repaint inside the frame — without it the repaint wiped the
  frame and left a tab of fill colour on the artwork beside it. Do not pad
  the box outward to catch ink just outside it: that lets artwork into the
  region, the frame detection loses edges and the repaint spills further.
- A CAPTION BOX IS FOUND FROM ITS FRAME (`frame_box` +
  `auto_caption_boxes`), not from detection's strips. A gradient box
  defeats detection in more shapes than the tint+bubble pair: one strip that
  misses its pale lines, three slivers side by side with the top lines in
  none (2-066), one line cut in two. Flood the fill from the lettering with
  only TRUE BLACK (max channel <= FRAME_MAX) as the wall — yellow and white
  are one region — and inside a drawn frame it comes back a rectangle
  (>=0.93; ovals 0.55-0.78), with nothing taller than a line of type inside
  and a frame on all four sides. It then becomes exactly a `box` override:
  the first member with text carries the whole box's text, the rest set
  nothing, the box is wiped whole, and its cap height is read off every
  letter in the frame (off a strip with the tint threshold it read 19px for
  letters 21-25px tall). It fires ONLY where detection demonstrably lost
  part of the box (>=4 letters outside every member mask and every strip
  pair, or a line cut side by side, or several members no pair covers):
  across both shipped books it finds nothing a hand `box` override does not
  already cover — and those overrides match its frames to 2px. A hand
  override always wins.
- BALLOONS FROM THE LETTERING (`letter_lobes`). Two balloons side by side in
  ONE entry: 4 of 11 in book 3 did not split from the mask at all (the
  distance-transform peaks miss a balloon much smaller than its
  neighbour), 4 came back SWAPPED (lobes sorted on their top row, and the
  right balloon's top sat a few px higher), and 2-030-2's mask had leaked
  across the panel so the cleaning painted out both outlines and the
  panel's colour. The letterer says which is which: split his letters at
  the widest MST gaps into one group per paragraph (never cutting off a lone
  speck), order them with `reading_order` (left before right when they share
  most of their height), flood each balloon out from its own letters, and
  where floods meet give each pixel to the lettering nearest it INSIDE the
  fill (`geodesic_owner` — straight-line distance gave a big balloon's
  corner to the small one beside it; a watershed on the distance-to-edge
  handed the strip under one text to the other balloon). Neighbouring
  entries' letters compete too (4-024 b05: a rectangle over half its text,
  the joined neighbour had the rest). Letters are only ink pieces WHOLLY
  inside the block — the block rectangle chops the outline into arcs that
  pass for letters, and one got repainted as a white tab. Each lobe owns its
  letters (welded to the outline, they fall outside the flood and survived
  as fragments). Used only where the old path fails: paragraphs side by
  side, a COMPOUND entry's mask well past (>1.4x) its balloons, or a mask
  well under (<60%) the balloon that ALSO lost the entry's own letters. A
  single balloon's mask commonly runs 1.6-2.9x its walled interior and the
  ordinary path copes — re-finding those moved 14 approved book-2 balloons
  for no gain, two of them worse (crowding a joined balloon).
- UPSCALE "DONE" IS NOT ONLY A SIZE. The resume check used to compare the
  output's size alone, and erasing a folio changes the pixels without
  changing the size wherever the number did not stretch the crop box — 15
  sources of book 3 kept their pre-erasure outputs, so the fit cleaned
  pages that still carried their printed numbers (and detection had made
  an empty bubble of each). An output older than folios.json, on a page the
  plan erases something from, is stale. The pristine/ copy of such a page is
  stale too: refresh it (`cp -p`) and re-detect the page.
- A FOLIO PAGE'S CROP must lose the folio's band: once the number is
  erased, scan dust left there (a 4px speck on the last row, a stripe at
  x=0) still held crop_to_content open, so 10 pages kept 50-90px of empty
  paper and printed their art ~4% smaller than their neighbours.
  `trim_folio_band` (folio pages only, bottom band + a narrow edge stripe
  on the sides) fixes exactly those 10 in book 3 and nothing in book 2. A
  general dust rule in crop_to_content was measured and REFUSED: it moved
  31 book-3 crops, 53 in approved book 2, 100+ in the UK set.
  Full-bleed pages (art to the side edges) are width-bound already and
  cannot print bigger.
- A SCAN-EDGE STRIPE holds the crop open on BOTH axes: book 3 p205 had 8px
  of grey gutter shadow at x=0 down its top 555 rows, which kept the left
  margin AND the top one (art 4% small). `drop_edge_stripes` (in
  crop_to_content, every page) clears a content run <=1% wide that touches
  the scan edge a gap away from the art, then re-measures. Narrower than the
  refused dust rule: 9 pages in approved book 2, 4/3 in comics 3/4, 75 in
  UK, and 22 in book 3 — 14 by the stripe, 8 by `trim_folio_band` no
  longer returning early when the bottom is already tight (it skipped its
  side check there: a 1px speck in from the edge held 8 folio pages ~3%
  small). Every one grew, to its neighbours' scale; boxes eyeballed.
- A fitted book's upscaled/ pages ARE the cleaned working pages, so
  `cmd_upscale` KEEPS (prints `KEPT:`) any existing page whose expected
  upscale changed once layout.json exists — it used to re-upscale it in
  place, under a layout, masks and positional transcripts in the old
  geometry. Rebuilding such a page = re-upscale + remap transcripts through
  the source + re-letter, by hand.
- REBUILDING AN APPROVED PAGE IS NOT A RE-FIT. A subset promote keeps
  entries verbatim, so an approved book's shipped entries can predate even
  the commit it was approved on: re-fitting 9 book-2 pages on IDENTICAL
  masks set 5 balloons 1px smaller with worse breaks (a 4-line balloon as a
  6-line column) — with today's code AND with the approval commit's. Per
  entry: take the new fit where it is LARGER or identical, otherwise carry
  the shipped entry through the page-geometry map (old page -> source ->
  new page: cx and each line box's centre; font, breaks, width unchanged).
  The balloon only grew, so the shipped lines still fit.
- Re-upscaling a relettered page changes its scale AND crop origin, so the
  positional transcripts must be remapped through the SOURCE: old page ->
  source px (old crop box, old scale) -> new page. Matching on raw or
  merely scaled block centres mis-pairs a whole page.
- A joined pair's straight cut is wrong whenever the two blocks OVERLAP
  on the axis it cuts across (the window between them runs backwards), not
  only when they overlap on both: 3-047 b03 lost its lines' first letters
  to its neighbour. Both go to `seam_split`. 8 book-3 pairs take the path;
  only that one had lost letters, so only that page was re-detected.
- `sibling_letter_box`: the FIT never writes over a sibling's lettering
  that a joined balloon's mask covers (2-064 b10 printed its first line on
  b07's last). The sibling's LETTERS inside the mask, not its block.
- A GRADED caption box can have too few letters for EITHER letter pass:
  on its tinted part letter_mask sees letter + fill as one dark blob, on its
  white part the tint ring test fails, and a two-word time stamp has 7
  letters against MIN_LETTERS 8 anyway. Detection's graded-box pass takes letters on
  either ground (BOX_MIN_LETTERS 5) and requires a drawn FRAME on all four
  sides as the evidence; it only ADDS boxes that overlap no existing entry,
  so no transcript key moves except by the inserted entry.
- A FRAME DETECTION ONLY PART-HOLDS is a box too: when the entries cover
  under FRAME_HELD (0.85) of the frame's rows, `auto_caption_boxes` makes
  the frame the box. Measured on book 3: every box held whole covers
  >= 0.90, every reported one 0.67-0.82. Held part-way, the fit got a
  fraction of the box (text small) and the cleaner, confined to the strip,
  read white off the pale rows and painted a band across it.
- `clean_caption_box` finds ink on the MAX channel as well as the min: a
  saturated yellow's blue channel sits by the ink (89 vs a letter's 65), so
  the min-channel test missed line ends there, and it also turns up a long
  band of FILL that a touching letter fused with and was spared as a
  "curve". Curves are classified per test, and a min-only band never spares
  what the max channel calls lettering.
- A balloon whose fill grades ALONG the row (announcer bursts: white glow,
  blue rim) cannot be repainted with one row median — every old line came
  back as a too-white band and the letters at the blue ends as white
  ghosts. `fill_trend` estimates the fill in 2-D from the fill around the
  lettering, ONLY when the fill strays >= FILL_TREND_MIN from its own row
  median (white and hologram-striped balloons stay on the row median), and
  `fill_wall` widens the colour wall by the ground the letters themselves
  stand on (the blue rim was 56 hue levels off the white glow, one over the
  wall). A balloon leaked onto sky keeps the plain wall: its letters stand
  on white.
- `repair_letter_bites`' "sealed" test reads the window edge as outside, and
  the rim channel between the eroded mask and the outline runs down a TAIL
  and out of the window: every rim pocket then counted as a leak (2-053's
  line end, welded to the arc, survived). Only a narrow opening on the window
  edge (TAIL_APERTURE) is closed. Do NOT close narrow passages everywhere —
  tried: 101 masks moved in book 3, and in book 2 the p104/p31/p294 leaks
  this guard exists for came straight back (they escape through a narrow
  outline gap, then open out).
- Two cheap size metrics catch "text too small" before the user does:
  `new_cap_h < old_cap_h` (the balloon demonstrably had room — 18 bubbles
  book-wide, and every page the user reported was among the worst of them),
  and dialogue set well under the book size.
- Transcripts: one string per bubble in bNN order; `*word*` = bold-italic;
  `\n` = hard break (hyphenation, or mirroring the original's line
  structure); `\n\n` = paragraph; `""` = leave untouched (SFX, display
  shouts/onomatopeia outside bubbles, art false positives). Bubble order
  keys on the TEXT BLOCK (y,x). New-bubble transcripts anchor to BLOCK
  COORDS printed by detection, never crop-order guesses.
- Fit: book size = 35th percentile of per-bubble maxima; dialogue capped at
  max(book, original size); grouped lobes share the group's smallest size;
  margin notes near their original (measure cap-height ratio from the font,
  never hardcode). THE ORIGINAL LETTERER IS THE GOLD STANDARD: anchor
  ladder = letterer's block center → (visual-x, letterer-y) → visual
  center, all with hold + a small vertical search (±0.9 lh), then the LAST
  RUNG: per-line spans on near-raw mask rows following the letterer's
  measured per-line axes, confined to the original text's vertical
  envelope. Users judge centering against the VISIBLE lobe and the
  original page — verify by rendering, never by layout numbers alone.
