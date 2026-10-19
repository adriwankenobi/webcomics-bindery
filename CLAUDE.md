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
- The docs (this file, README.md, ARCHITECTURE.md, the qa-tools README)
  explain HOW things work and WHY — never which comic, book, page or bubble
  showed it, incident counts, round numbers, dates or quoted dialogue. That
  evidence belongs in commit messages; per-book state belongs in memory.
  Tests use placeholder names ("Series 13 - Title 012"), never real titles
  or transcript text.
- The QA gates and rebuild drivers in relettering/qa-tools/ are PROCESS and
  are tracked (whitelisted by name in .gitignore); the eyeball helpers
  beside them (crop/zoom/montage/...) are scratch and stay untracked. They
  take the comic as argv[1] like everything else, and the two that re-run
  the fit (rebuild_fit.py, refit_subset.py) REQUIRE BOOK=<size> — an
  unpinned re-fit re-derives the book size and moves approved pages.
- `just process "<comic>" --relettering` = the whole re-lettered book in
  one resumable run; it pauses once (`== TRANSCRIPTION NEEDED ==`, Enter or
  re-run to resume). Markers: pristine/ (skipped once layout.json exists),
  bubbles/<stem>.json for EVERY page (a partial set = an interrupted
  detection, never "done": it re-runs whole while no transcripts exist and
  is refused after — "any json = done" once made sheets for a fraction of a
  book), sheets/manifest.json, all manifest keys in parts/,
  layout.json — detection and the fit are NEVER re-run by the driver — and
  for the GIMP text pass qa/<stem>.png newer than the .xcf (delete the PNG
  to force one page; a recompose forces it too, so the single-page recipe
  no longer prunes layout.json). Postprocess runs only on the stems the
  pass reported RELETTER DONE.
- KEEP OR RE-LETTER is decided per ISSUE (`issue_of` = the issue part of
  short_key) after detection, BEFORE the sheets (`reletter_decide.py`,
  rule in `issue_plan.py`): keep only when the font is similar AND our
  predicted letters are < SIZE_BAR - UNSURE_MARGIN (1.10 - 0.03) of the
  original's; either gate saying re-letter wins, unsure re-letters and is
  listed. The size is predicted by the REAL fit on placeholder text shaped
  from the original's blobs (two-pass: book on a sample, then search from
  max(book, original)). The font gate judges the issue's MEDIAN
  per-balloon difference to our font; no single statistic separated the
  labelled sets, "wider AND more upright" did. `issue_decisions.json` is
  written ONCE;
  `issue_overrides.json` wins (typos stop the run). Kept pages get no
  sheets/fit/text. A comic whose sheets or layout.json predate the step
  (no decisions file) = everything re-lettered, the step writes NOTHING
  there — never create a decisions file (or fit_meta.json) in an approved
  comic. Flips after the fit: keep -> restore_kept_pages (pristine, drop
  xcf/pdf/qa — mtime trap; no pristine copy = refused); re-letter ->
  sheets appended + `reletter_fit.py --add` at fit_meta.json's book size.
  LAYOUT.JSON is the only authority on which pages are fitted. Calibrate against fitted comics with
  `qa-tools/decide_calibrate.py` (read-only, $SP/decide/).
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
  whole regions off the interior — the fit then sees a fraction of the
  balloon and drops several size steps, even below the original's own size.
  Both visible symptoms come from this one fault: small text that does not
  fill the balloon, AND the letters inside the bite never being cleaned (the
  cleaner is confined to the mask), left as faint near-white fragments.
  `repair_letter_bites` re-floods with the RAW ink as the only wall and adds
  back only what the mask and the ink ALREADY ENCLOSE (a bite is a pocket; a
  leak is enclosed by nothing). Bounding it to the letterer's block is NOT
  enough on its own: the blob cluster reaches past the balloon's own arc, so
  an escape onto pale sky can land INSIDE the block — the cleaner then reads
  the fill off the sky, paints it over the artwork and leaves the old
  lettering standing. An area ratio cannot see it either: such an escape can
  be under 2% of the mask.
- The fill is the authority on the row profile in BOTH directions
  (`resolve_rows`): clip where the profile leaked past it, and give the rows
  their width back where the profile sits further inside it than its own
  clearance. `rows` is the mask ERODED, so it carries every bite, detection's
  letter-union repair only reaches rows that hold letter pixels, and the 6x6
  erosion smears a bite three rows past the letter that caused it. Those
  rows are what a line box lands on and `rows_span` takes the NARROWEST row
  of the band, so two chewed rows can cost several size steps.
- `clean_caption_box` estimates the fill under a word by INTERPOLATING across
  it from the fill either side, per row. A morphological close is right for
  FINDING the ink (it is only a threshold) and wrong for replacing it: a
  close takes the BRIGHTEST value in its window, and the window has to be as
  wide as a whole word, so on a box whose fill grades along the row it reads
  the bright end as the fill everywhere and repaints the line as a flat,
  too-bright slab with straight edges (20-45 levels bright). Measure the fill
  on `~(ink | halo)` — never on pixels merely excluded from the repaint, or
  the spared frame drags the estimate ~100 levels dark beside it.
- With `--book` PINNED the sizing pass may cap its search at
  max(book, the original's own size) — every dialogue size is clamped there
  anyway. 7x faster and byte-identical (verified). Unpinned it must still run
  free: the book size IS the percentile of those maxima.
- The bubble repaint must protect the OUTLINE without stranding lettering —
  both halves matter. Masks leak (a tail that leaves the crop window floods
  the white page; a balloon on pale sky has nothing to stop the fill), so
  everything dark in the mask cannot just be wiped — that erases the outline
  and the balloon vanishes, leaving the new text on bare art. But "letters =
  dark islands ENCLOSED by the fill" is ALSO wrong: a joined-balloon split
  truncates the mask mid-text and letters straddling that edge are not
  enclosed, so they survive as stray letters under the new text. The rule
  that satisfies both: wipe every dark pixel the mask covers EXCEPT long
  thin CURVES (extent >= 0.45 * bbox span and area <= 0.28 * its own bbox =
  an outline or a panel line; lettering is compact even when several
  letters merge), plus the enclosed islands. Classify on the WHOLE region —
  the mask boundary chops the outline into short arcs that stop looking like
  curves. Never size-cap what counts as lettering.
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
  this crams every paragraph into one lobe at a tiny size and leaves the
  other lobes empty.
- The letterer's block is THE anchor, so clip it (`block_anchor`) to the
  balloon's rows AND its bbox: the blob cluster swallows artwork beside the
  balloon (fur, a panel line) and slides text clean out of the lobe.
- Keep the new lettering clear of the drawn outline (`inset_rows`,
  ~3.5% of the profile's extent) — the row profile runs to the mask edge and
  the mask includes the outline stroke, so type set at the book size lands
  ON the line. Apply it to the FIT ONLY: inset the CLEANING and rim letters
  survive.
- Where a mask LEAKED past the balloon, clip the fit's row profile (and the
  last rung's `rows_raw`) to the balloon's own fill — a leaked profile can
  be a rectangle far wider than the oval, so lines get set where the oval
  has long since narrowed.
- Overlap between two balloons is ceded whole to one of them; a balloon must
  never cede pixels inside its OWN text block, or its lettering is left
  uncleaned AND the fit has no room. But "its OWN" must exclude whatever the
  SIBLING's block claims (`own_block_mask`): a block is a blob cluster and
  can reach over the sibling's first line — the guard then holds that line,
  and the slice of the sibling's balloon this flood filled, inside this
  mask, and this balloon's own last line is cut away instead. Where two
  blocks collide neither claim is credible.
- Joined lobes are stacked or side by side — EXCEPT when their blocks
  overlap on BOTH axes, which means they are DIAGONAL and no straight cut
  separates them. The row window then runs BACKWARDS (from the upper's block
  bottom UP to the lower's block top) and the cut lands inside the upper's
  own lettering: the mask loses that line, the cleaner (confined to the
  mask) leaves it standing, and the profile keeps the part of the sibling's
  balloon the flood filled, so lines are set wider than the balloon.
  `seam_split` assigns each pixel to the lobe whose own LETTERING is nearer
  — never the block, which is the polluted thing. Clamp the straight cuts
  out of the upper/left lobe's own block too, but SKIP a clamped cut that
  would leave a lobe with under a quarter of its mask.
- The refine pass (`strict_bubble` per lobe) may miss 15% of the block's
  rows, but not as a TRUNCATION at an end of it: a flood that stops short
  of the block's bottom is accepted on coverage, the mask loses the last
  line, and the old line stands under the new text. Give those rows back
  from the mask being replaced, bounded to the lobe's own block. And drop
  mask DEBRIS before cropping: slivers of a ceded sibling's rim, a percent
  or two of the mask, can hold the bbox far taller than its own balloon.
- Detection changes are only safe while the bubble COUNT per page holds —
  the transcripts are positional. Re-detect into a scratch compare first and
  REFUSE any page whose count moved. Widening `letter_mask` to accept tinted
  fill (for gradient caption boxes) creates entries, so it cannot be done
  without re-transcribing those pages.
- The overflow inpaint's window is the bubble's OWN bbox, never the raw
  `block`: a blob cluster swallows the balloon next door, and the step then
  erases a neighbour's lettering that nothing types back — an EMPTY
  balloon. Measured: reaching past the bbox buys a handful of stray blobs a
  few px out and costs a neighbour's letters and line art; everything it
  legitimately cleans is inside the bbox. No margin helps — the balloons
  are adjacent.
- A coloured caption box with a gradient fill comes back as TWO stacked
  entries: a `tint` strip plus a `bubble` strip, split where the fill crosses
  LIGHT_MIN, each with a RAGGED mask that follows the fill instead of the
  box. Clean it as ONE box (`caption_boxes` -> `clean_caption_box`) INSTEAD
  of its members — never after them, or the box pass reads their whitened
  strip as the fill level and paints a white band right across it. The
  transcripts are already complete (one half per entry), so this needs NO
  re-transcription. `{"box": [x0,y0,x1,y1]}` in layout_overrides.json
  covers a box detection half-masked without splitting.
- The two strips of one box must AGREE on their x range (`X_AGREE`, IoU
  0.85), not merely overlap: they are cut from ONE rectangle by a horizontal
  fill threshold, so their extents match to a few px, while an oval of
  similar width sitting a little to one side passes an overlap test easily.
  A false member does double damage: the box grows past its frame so the
  fit gets more writing width than the box has (lines print past the
  frame), AND the box now has a member nothing typesets, so the wipe skips
  the whole box and its strips are cleaned as separate entries (a flat
  slab, a white band at the seam, a tab of fill on the artwork). Real pairs
  measure ~0.89-1.0, false ones ≤ ~0.77; the strip-shape ratio does NOT
  separate them.
- A caption box is ONE BLOCK OF TYPE (`stack_box_lines`). The strips are an
  artefact of the fill crossing LIGHT_MIN and carry no meaning for the
  layout, but centring each in its own strip prints the box with a blank
  line at the seam, as if it were two balloons. Put every line of the box
  on one grid centred on the union of its members' blocks. A `sole_box`
  entry already spans the whole box and is left alone.
- A drawn caption FRAME gets NO overhang (`wrap_at(..., tight=True)` for any
  entry in a box). `wrap_at` lets the first and last line hang 0.16 of a line
  height past their rows, because glyphs fill ~70% of the line box and a
  balloon's arc forgives it; a frame forgives nothing, the fill IS the
  writing area — with overhang a short box "fits" two line boxes and the
  type prints across its top and bottom rules. Also: a `box` override's
  rect is the frame's INNER edge (boxrect.py's convention), never the outer
  frame — measured on the outside, the box reads taller than it is and the
  size comes out too big.
- `clean_caption_box` estimates the fill LOCALLY (horizontal grayscale close,
  then paint from that same estimate per channel). A per-row level reads a
  highlight as the fill and everything yellow as ink, and paints the row
  WHITE. Keying on shape alone (a rectangular mask, a drawn frame) is not
  safe: both also match oval balloons and leaked masks.
- `bubble_interior` walls the flood on a COLOUR change as well as on dark
  strokes (COLOUR_WALL). Where a balloon sits on pale sky and its outline has
  a gap, the fill escapes and the "interior" becomes balloon + page — which
  silently defeats clipping the fit's row profile to that interior (lines
  get set above the balloon's top arc).
- Row spans follow ONE connected run (`widest_run`), never first-to-last.
  Low in a balloon the fill splits into the body and the mouth of the tail;
  spanning that gap lets the wrap set its last lines out into the tail, hard
  against the outline.
- `parse_runs` resolves style PER CHARACTER, then splits on whitespace. A
  regex alternation cannot: `\S+` swallows `¡--*WORD*` whole and prints
  literal asterisks. `\*` is a literal asterisk (a footnote mark) and never
  toggles emphasis; a lone unmatched `*` also stays literal, but only when
  no other `*` follows it.
- The ink cut-off leaves the lettering's PALE strokes (150-220 on a 250 fill)
  behind as faint letter-shaped specks — visible at 1:1 300dpi and invisible
  to letter_mask. A softer cut-off fixes it but ONLY inside the letterer's
  block AND inside the balloon's own fill (`solid`): bounded to the block
  alone it catches artwork through a leaked mask and paints white
  rectangles.
- Verify EVERY batch, and NEVER substitute a proxy metric for these — a
  layout-geometry check can pass while stranded letters ship:
  (1) leftover scan = re-run detection's own `letter_mask` on the CLEANED
  working pages over the mask UNION the block UNION every laid-out line's
  box; any letter blob where the transcript has text = failed cleaning
  (healthy baseline: <20 book-wide). Intersecting with the MASK ALONE hides
  the worst failure there is — lettering the mask never covered is exactly
  what puts new text on top of old — and reads "healthy" while whole lines
  of old text stand under the new. It is still blind to text on a COLOURED
  fill and to faint residue; (2) qa_scan pixel-diff vs
  relettering/<comic>/pristine/ — this is the gate that catches a cleaning
  change repainting ART (white rectangles); `protrude.py` cannot stand in
  for it, since it measures against the MASK and a leaked mask is the very
  thing that puts text outside the balloon (text-shaped footprints only;
  also eyeball blobs taller than a text line — wedges slip the 4000/35
  gate, and an erased balloon outline is LOW-area but wide, so gate on
  min-dim ≥45 / area ≥400 to catch it); (3) EXACT 1:1 300dpi render
  overlays vs the source — fraction crops hide 10-20px placement errors the
  reviewer sees.
- A SUBSET promote ships the layout fix but NOT the cleaning fix. `promote.py`
  only ever looks at the pages the subset re-fit staged, so a cleaning change
  never reaches the rest, which keep the old cleaning's pixels (a whole
  paragraph can stay legible as a grey ghost under the new text). Cleaning
  changes need a full re-clean (`reclean_all.py`, ~1.4 min) — measured with
  `faint_scan.py`, which the leftover scan cannot see.
- THE PER-BUBBLE PREP HAS ONE HOME: `prepare_bubble` (with
  `page_caption_boxes` / `boxes_to_wipe` / `is_strip`). The fit's `main()`
  and `reclean_all.py` both call it. A hand-mirrored copy of that
  preparation in a caller means every new cleaning behaviour has to be
  copied there by hand, or the book-wide verification silently stops
  reproducing the pipeline. Add cleaning behaviour THERE, never in a caller.
- MEASURE THE PDF, NOT THE PLAN. `cap_height` used to return PIL's
  `getbbox("H")` — the layout box from the ascender origin, rounded out to
  whole px — while `old_cap_h` is measured off the ORIGINAL'S PIXELS. At 23px
  it claimed 22 where GIMP renders 19, so `new_cap_h >= old_cap_h` COULD NOT
  FAIL and text printed smaller than the original passed as "verified".
  cap_height now measures the cap glyph's own ink height from the font at
  run time (fontTools, x size / unitsPerEm). Any metric computed from a
  MODEL has to be checked against the RENDERED artefact once, or it
  certifies the model.
- The configured font can be much WIDER per unit of cap height than the
  original edition's (~1.5x for the default font), so a tight balloon cannot
  hold the original's cap height at natural width, and "text small, not
  filling the bubble" is usually width-bound within a pixel or two. Two
  levers, both deliberate POLICY CALLS by the user:
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
  on a balloon that IS drawn, and then nothing bounds the new type: it can
  be set wider than the oval and print across the outline on both sides.
  Its block cannot catch it (the blob cluster swallows both arcs, so the
  block is wider than the interior) and neither can the size ceiling (those
  arcs inflate the measured cap height by many size steps).
  `unfound_balloon` floods out from the block, and the proof that what
  comes back IS a balloon is that it is NARROWER than the block — the block
  runs the width of the lettering, so a real interior contains it, and an
  escaped flood (e.g. down the tail) comes back wider and is refused. Do
  NOT try to get this from the ink components instead: dropping the ones
  that cross the block border also drops a letter WELDED to the outline,
  and keeping them lets artwork in.
- `{"lobes": [[x0,y0,x1,y1], ...]}` in layout_overrides.json is the escape
  hatch for a mask that leaked past all recognition — e.g. two balloons on
  pale sky returned as ONE entry whose mask covers most of the panel:
  mask_lobes sees one blob, the fit places nothing, cleaning it would
  repaint sky, and splitting the entry in detection would renumber the page
  and invalidate its positional transcripts. One rectangle per balloon, IN
  PARAGRAPH ORDER, each roughly centred on its balloon; each is re-flooded
  inside itself (`lobes_from_rects`) and the entry then bypasses every
  block-driven repair, because its block spans both balloons. CLEAN IT PER
  LOBE — `lobes_from_rects` hands back one job per balloon, each with the
  original ink inside that interior as its block. Cleaned as ONE entry the
  cleaner reads the entry's own block (spanning the sky between the
  balloons) and the strict ink term repaints the artwork behind them; per
  lobe it changes nothing outside the rectangles. EVERY gate that reads the
  mask PNG must apply the override too (`gate_mask` in
  leftover/welded/faint_scan; outside.py floods its own fill, so it takes
  the per-lobe interiors instead — and in the PAGE frame, since these lobes
  can reach far past the entry's bbox). Otherwise the gate measures a
  region nothing ever cleaned: letter_mask finds letter-shaped pieces of
  ARTWORK in a leaked mask, and outside.py ranks correctly-placed `lobes`
  entries as the worst in the book.
- A CAPTION IS CENTRED ON THE LETTERER'S BLOCK, like a balloon. `tint`/
  `dark`/`open` used the width-weighted centroid of their own row profile,
  and `caption_boxes` finds a box by its FILL — so on a yellow->white graded
  box the rectangle is only the coloured part and its centre sits ABOVE where
  the letterer set the type: the fill is reconstructed perfectly, the TYPE
  is set high, and the strip of flat colour left under it reads as a
  coloured rectangle. The non-bubble `wrap_at` call must also pass
  `hold_cy=True`: without it wrap_at re-balances and re-centres the block
  geometrically in the rows, throwing away the anchor you just measured.
- A caption box's drawn FRAME is found on `region.max(axis=2)`, never on the
  min channel: a yellow fill has a low BLUE channel, so min() calls nearly
  the whole box "dark" and no frame can be picked out of it. `frame_bounds`
  confines the repaint inside the frame — without it the repaint wipes the
  frame and leaves a tab of fill colour on the artwork beside it. Do not pad
  the box outward to catch ink just outside it: that lets artwork into the
  region, the frame detection loses edges and the repaint spills further.
- A CAPTION BOX IS FOUND FROM ITS FRAME (`frame_box` +
  `auto_caption_boxes`), not from detection's strips. A gradient box
  defeats detection in more shapes than the tint+bubble pair: one strip that
  misses its pale lines, several slivers side by side with the top lines in
  none, one line cut in two. Flood the fill from the lettering with only
  TRUE BLACK (max channel <= FRAME_MAX) as the wall — yellow and white are
  one region — and inside a drawn frame it comes back a rectangle (>=0.93;
  ovals 0.55-0.78), with nothing taller than a line of type inside and a
  frame on all four sides. It then becomes exactly a `box` override: the
  first member with text carries the whole box's text, the rest set
  nothing, the box is wiped whole, and its cap height is read off every
  letter in the frame (off a strip with the tint threshold it reads
  several px short). It fires ONLY where detection demonstrably lost part
  of the box (>=4 letters outside every member mask and every strip pair,
  or a line cut side by side, or several members no pair covers); where
  hand `box` overrides exist they match its frames to ~2px. A hand
  override always wins.
- BALLOONS FROM THE LETTERING (`letter_lobes`). Two balloons side by side in
  ONE entry often fail from the mask: the distance-transform peaks miss a
  balloon much smaller than its neighbour, lobes sorted on their top row
  come back SWAPPED when the right balloon's top sits a few px higher, and a
  mask leaked across the panel gets both outlines and the panel's colour
  painted out. The letterer says which is which: split his letters at the
  widest MST gaps into one group per paragraph (never cutting off a lone
  speck), order them with `reading_order` (left before right when they share
  most of their height), flood each balloon out from its own letters, and
  where floods meet give each pixel to the lettering nearest it INSIDE the
  fill (`geodesic_owner` — straight-line distance gives a big balloon's
  corner to the small one beside it; a watershed on the distance-to-edge
  hands the strip under one text to the other balloon). Neighbouring
  entries' letters compete too (a joined neighbour can hold half of this
  entry's text). Letters are only ink pieces WHOLLY inside the block — the
  block rectangle chops the outline into arcs that pass for letters, which
  then get repainted as white tabs. Each lobe owns its letters (welded to
  the outline, they fall outside the flood and survive as fragments). Used
  only where the old path fails: paragraphs side by side, a COMPOUND
  entry's mask well past (>1.4x) its balloons, or a mask well under (<60%)
  the balloon that ALSO lost the entry's own letters. A single balloon's
  mask commonly runs 1.6-2.9x its walled interior and the ordinary path
  copes — re-finding those moves approved balloons for no gain, some for
  the worse (crowding a joined balloon).
- UPSCALE "DONE" IS NOT ONLY A SIZE. Erasing a folio changes the pixels
  without changing the size wherever the number did not stretch the crop
  box, so a size-only resume check keeps pre-erasure outputs — the fit then
  cleans pages that still carry their printed numbers (and detection makes
  an empty bubble of each). An output older than folios.json, on a page the
  plan erases something from, is stale. The pristine/ copy of such a page is
  stale too: refresh it (`cp -p`) and re-detect the page.
- A FOLIO PAGE'S CROP must lose the folio's band: once the number is
  erased, scan dust left there (a speck on the last row, a stripe at x=0)
  still holds crop_to_content open, so the page keeps a band of empty paper
  and prints its art a few % smaller than its neighbours. `trim_folio_band`
  (folio pages only, bottom band + a narrow edge stripe on the sides) fixes
  that. A general dust rule in crop_to_content was measured and REFUSED: it
  moves many correct crops, including in approved books. Full-bleed pages
  (art to the side edges) are width-bound already and cannot print bigger.
- A SCAN-EDGE STRIPE holds the crop open on BOTH axes: a thin gutter shadow
  at x=0 down part of the page keeps the left margin AND the top one.
  `drop_edge_stripes` (in crop_to_content, every page) clears a content run
  <=1% wide that touches the scan edge a gap away from the art, then
  re-measures. Narrower than the refused dust rule. `trim_folio_band` must
  not return early when the bottom is already tight — it would skip its
  side check, and a 1px speck in from the edge holds the page small.
- A fitted book's upscaled/ pages ARE the cleaned working pages, so
  `cmd_upscale` KEEPS (prints `KEPT:`) any existing page whose expected
  upscale changed once layout.json exists — it used to re-upscale it in
  place, under a layout, masks and positional transcripts in the old
  geometry. Rebuilding such a page = re-upscale + remap transcripts through
  the source + re-letter, by hand.
- REBUILDING AN APPROVED PAGE IS NOT A RE-FIT. A subset promote keeps
  entries verbatim, so an approved book's shipped entries can predate even
  the commit it was approved on: re-fitting on IDENTICAL masks can set
  balloons smaller with worse breaks (a 4-line balloon as a 6-line column)
  — with today's code AND with the approval commit's. Per entry: take the
  new fit where it is LARGER or identical, otherwise carry the shipped entry
  through the page-geometry map (old page -> source -> new page: cx and each
  line box's centre; font, breaks, width unchanged). The balloon only grew,
  so the shipped lines still fit.
- Re-upscaling a relettered page changes its scale AND crop origin, so the
  positional transcripts must be remapped through the SOURCE: old page ->
  source px (old crop box, old scale) -> new page. Matching on raw or
  merely scaled block centres mis-pairs a whole page.
- A joined pair's straight cut is wrong whenever the two blocks OVERLAP
  on the axis it cuts across (the window between them runs backwards), not
  only when they overlap on both: the lines' first letters go to the
  neighbour. Both go to `seam_split`.
- `sibling_letter_box`: the FIT never writes over a sibling's lettering
  that a joined balloon's mask covers (else a first line prints on the
  sibling's last). The sibling's LETTERS inside the mask, not its block.
- A GRADED caption box can have too few letters for EITHER letter pass:
  on its tinted part letter_mask sees letter + fill as one dark blob, on its
  white part the tint ring test fails, and a two-word time stamp has 7
  letters against MIN_LETTERS 8 anyway. Detection's graded-box pass takes
  letters on either ground (BOX_MIN_LETTERS 5) and requires a drawn FRAME on
  all four sides as the evidence; it only ADDS boxes that overlap no
  existing entry, so no transcript key moves except by the inserted entry.
- A FRAME DETECTION ONLY PART-HOLDS is a box too: when the entries cover
  under FRAME_HELD (0.85) of the frame's rows, `auto_caption_boxes` makes
  the frame the box. Boxes held whole cover >= 0.90, part-held ones
  0.67-0.82. Held part-way, the fit gets a fraction of the box (text small)
  and the cleaner, confined to the strip, reads white off the pale rows and
  paints a band across it.
- A SCANNED FRAME IS NOT A RULER: two drawing faults make `frame_box` miss a
  box, which then ships half-wiped (white band, old letters standing, the
  last line set small and out of the frame). (1) TILT: a hand-drawn edge
  drifts 4-6px across the box, and a four-sides test looking in a straight
  band outside the fill's BOUNDING box reads the low end of the edge as
  unframed — `frame_sides` measures from each column's/row's OWN edge of
  the fill. (2) HAIRLINE: a few columns of an edge scan just above the
  FRAME_MAX cut-off and the flood runs out into the pale art — `frame_wall`
  takes pixels up to FRAME_SOFT only where they TOUCH true black.
  `qa-tools/boxleft.py` is the gate; qa_scan catches only some of these.
- A DASHED OUTLINE (a whisper) is an outline, not lettering
  (`dashed_rings` in detection). On near-white art (skin ~20 levels from
  the fill) the dashes are the ONLY wall, and the gaps between them run
  from ~7px on an arc to >21px at a corner where a dash fuses with the art,
  so no closing kernel seals them (7-21px all tried: the flood still takes
  the art and the cleaner wipes half the outline). The ring is found as
  >= 8 plain strokes (most of a letter long, so accents stay letters)
  chained at <= 1.4 letter heights, outside every PARAGRAPH's hull (one
  hull over all the text swallows a figure-8's waist), round >= 8 of 12
  sides, pointing along the chain and every which way (speed lines and
  hologram scan lines chain round text too, but are parallel). Its dashes
  leave the letter mask and the gaps are drawn shut on a working copy of
  the page (neighbours, next round the ring, and each dash end to the
  nearest ink), never across the lettering. The entry is marked `"dashed"`,
  and the fit then skips the bite repair (it re-floods to the raw ink, i.e.
  back over the dashes), protects every dark piece crossing the sealed
  mask's edge, and skips the overflow inpaint (all that is out there is the
  outline). No ring = the page detects exactly as before; it is rare.
- `clean_caption_box` finds ink on the MAX channel as well as the min: a
  saturated yellow's blue channel sits by the ink (89 vs a letter's 65), so
  the min-channel test misses line ends there, and it also turns up a long
  band of FILL that a touching letter fused with and was spared as a
  "curve". Curves are classified per test, and a min-only band never spares
  what the max channel calls lettering.
- A balloon whose fill grades ALONG the row (announcer bursts: white glow,
  blue rim) cannot be repainted with one row median — every old line comes
  back as a too-white band and the letters at the blue ends as white
  ghosts. `fill_trend` estimates the fill in 2-D from the fill around the
  lettering, ONLY when the fill strays >= FILL_TREND_MIN from its own row
  median (white and hologram-striped balloons stay on the row median), and
  `fill_wall` widens the colour wall by the ground the letters themselves
  stand on (a blue rim can sit just over the wall from the white glow). A
  balloon leaked onto sky keeps the plain wall: its letters stand on white.
- `repair_letter_bites`' "sealed" test reads the window edge as outside, and
  the rim channel between the eroded mask and the outline runs down a TAIL
  and out of the window: every rim pocket then counts as a leak (a line end
  welded to the arc survives). Only a narrow opening on the window edge
  (TAIL_APERTURE) is closed. Do NOT close narrow passages everywhere —
  tried: it moves many correct masks, and the leaks this guard exists for
  come straight back (they escape through a narrow outline gap, then open
  out).
- THE JOINED-PAIR CUTS. Four faults, each of which leaves a lobe's first or
  last line outside every mask (never cleaned, the new text printed over
  it): (1) `cut_mask` must subtract the origin ON THE CUT'S AXIS —
  subtracting bbox[axis] (x for a row cut) puts every merge-stage straight
  cut off by the mask's x−y offset; (2) a notch whose narrowest row is the
  END of the search window is no waist (offset pair) — `notch_cut` returns
  None and the pair takes `seam_split`; a real waist is kept between the
  blocks; (3) in a stack of 3+ joined lobes every first mask is the union of
  all, so the merge host must be the NEAREST lobe above, and every merge
  pair (not only two-member groups) gets the overlap resolution; (4)
  `follow_lobe(text_drift=True)` (main walk ONLY) judges drift against the
  block centre too — a burst's spike at the seed row stops the walk inside
  the first line. Enabling it in the short-utterance / joined-lobe walks
  CREATES entries (count change) — refused. Dry-run the whole book after
  any detection change (count must hold; diff letter coverage per entry)
  and re-detect only pages that GAIN coverage.
- A short balloon's host can already be FOLDED by the main pass (its mask
  the re-cropped union of two balloons), so its frame is not the light
  component's frame the walk ran in: put both masks on one `common_frame`
  before combining them, or detection crashes on the shape mismatch.
- `letter_lobes`: a rival entry's letters never re-seed letters this entry
  already owns — a phantom entry ("" transcript) whose block is one of this
  entry's paragraphs empties that lobe and the compound is never re-found.
  A recognised strip pair also shares ONE cx (union of its blocks), not
  just one line grid.
- An ELLIPSIS splits a short balloon between detection's two passes: the
  dots are below the letter-blob floor, so two short words either side of
  "..." are two clusters, each < MIN_LETTERS, to `paragraph_blocks`, and
  one cluster to the short-utterance pass's wider join, which only took
  clusters UNDER MIN_LETTERS. It now takes MIN_LETTERS or more too, unless
  the cluster overlaps a paragraph block (that one was refused on its own
  merits). This ADDS entries on a re-detect, so an already-transcribed book
  must not be re-detected wholesale for it (it can also split a caption
  line) — dry-run and take only the pages that gain a missed balloon.
- A token with no letter in it ("...", "¡...", "--") is a word of its own
  to the wrap, and the balance pass breaks after it, printing a lone "..."
  over the first line. `glue_bare_punct` (end of `parse_runs`) glues it to
  the next word (the last one to the word before), and `split_word` never
  breaks at or beside the glue.
- `notch_cut` judges "no waist" on the profile's VALUE: a plateau running
  to the window's end has its first minimum a few px short of the end, so
  an index test lets it through and the cut is clamped onto the second
  block's start — a leading "..." (no letters, so outside the block) goes
  to the other lobe.
- Margin color: art that runs to all four edges in one flat colour (sky,
  sea) reads as PAPER and the margins come out that colour. The rule is
  kept (coloured stock, dark pages, printed coloured borders); per-page
  override in untracked `upscaled/<comic>/paper_overrides.json`
  ({source filename: [r,g,b]}), read when compose writes numbers.txt —
  delete the page's .xcf/.pdf to recompose it.
- Two cheap size metrics catch "text too small" before the user does:
  `new_cap_h < old_cap_h` (the balloon demonstrably had room — the pages a
  reviewer reports are reliably among the worst of these), and dialogue set
  well under the book size.
- Transcripts: one string per bubble in bNN order, keyed `"<short> bNN"`
  where `<short>` = `make_sheets.short_key` (issue-page; a dotted issue
  keeps its decimals and letter suffix, "7.02a-003"). It is THE key —
  overrides call the same function, never a copy of its regex — and
  make_sheets refuses two pages with bubbles sharing one. `*word*` =
  bold-italic; `\*` = literal asterisk (footnote mark); `\n` = hard break
  (hyphenation, or mirroring the original's line structure); `\n\n` =
  paragraph; `""` = leave untouched (SFX, display shouts/onomatopeia outside
  bubbles, art false positives). Bubble order keys on the TEXT BLOCK (y,x).
  New-bubble transcripts anchor to BLOCK COORDS printed by detection, never
  crop-order guesses.
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
