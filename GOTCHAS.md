# GOTCHAS.md

Hard-won rules behind the code, by topic. CLAUDE.md keeps the ones
most tasks need; read the section for the part you are changing before you
change it. Like the other docs: mechanisms and reasons only, never which
comic, page or round showed them.

## Upscale, crop and margins
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
- Margin color: art that runs to all four edges in one flat colour (sky,
  sea) reads as PAPER and the margins come out that colour. The rule is
  kept (coloured stock, dark pages, printed coloured borders); per-page
  override in untracked `upscaled/<comic>/paper_overrides.json`
  ({source filename: [r,g,b]}), read when compose writes numbers.txt —
  delete the page's .xcf/.pdf to recompose it.

## Rebuilding approved pages
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
- A SUBSET promote ships the layout fix but NOT the cleaning fix. `promote.py`
  only ever looks at the pages the subset re-fit staged, so a cleaning change
  never reaches the rest, which keep the old cleaning's pixels (a whole
  paragraph can stay legible as a grey ghost under the new text). Cleaning
  changes need a full re-clean (`reclean_all.py`, ~1.4 min) — measured with
  `faint_scan.py`, which the leftover scan cannot see.
- With `--book` PINNED the sizing pass may cap its search at
  max(book, the original's own size) — every dialogue size is clamped there
  anyway. 7x faster and byte-identical (verified). Unpinned it must still run
  free: the book size IS the percentile of those maxima.

## Keep or re-letter
- KEEP OR RE-LETTER is decided per ISSUE (`issue_of` = the issue part of
  short_key) after detection, BEFORE the sheets (`reletter_decide.py`,
  rule in `issue_plan.py`): keep only when the font is similar AND the
  original's dialogue cap height PRINTS at >= MIN_CAP_MM (1.25 mm: pixels
  at the template's resolution, pages placed 1:1); either gate saying
  re-letter wins, unsure re-letters and is listed. Judge the PAPER, not
  our gain: a ratio bar re-lettered issues whose originals already read
  well, for a gain of a fraction of a mm nobody sees in print (the fit is
  held by the balloons). The predicted ratio (the REAL fit on placeholder
  text shaped from the original's blobs; two-pass: book on a sample, then
  search from max(book, original)) is still reported beside the verdict,
  deciding nothing. The font gate judges the issue's MEDIAN
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

## Detection
- Detection is TEXT-FIRST (letter blobs → paragraph clusters → enclosing
  light region). Shape-first detection false-fires on armor/smoke.
- np.pad masks before cv2 morphology close + hole-fill: OpenCV treats
  out-of-border as foreground and turns corner voids into filled RECTANGLES.
- Never fill row intervals across disconnected runs; follow the single
  connected run (follow_lobe). Balloons crossing a panel border need the
  ring-component union + border-row bridge (bridge = row UNION, never
  intersection) inside strict_bubble; leaks get the dark-row cut and the
  neck cut.
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
  BEFORE the fit the count can move: `qa-tools/redetect_parts.py` carries
  the parts/ transcripts to the new numbering by geometry (and refuses a
  fitted comic). "Same count, same index" is not "unchanged" — an entry can
  keep its index and become a caption box; compare kind, bbox and block.
- `bubble_interior` walls the flood on a COLOUR change as well as on dark
  strokes (COLOUR_WALL). Where a balloon sits on pale sky and its outline has
  a gap, the fill escapes and the "interior" becomes balloon + page — which
  silently defeats clipping the fit's row profile to that interior (lines
  get set above the balloon's top arc).
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
- A short balloon's host can already be FOLDED by the main pass (its mask
  the re-cropped union of two balloons), so its frame is not the light
  component's frame the walk ran in: put both masks on one `common_frame`
  before combining them, or detection crashes on the shape mismatch.
- An ELLIPSIS splits a short balloon between detection's two passes: the
  dots are below the letter-blob floor, so two short words either side of
  "..." are two clusters, each < MIN_LETTERS, to `paragraph_blocks`, and
  one cluster to the short-utterance pass's wider join, which only took
  clusters UNDER MIN_LETTERS. It now takes MIN_LETTERS or more too, unless
  the cluster overlaps a paragraph block (that one was refused on its own
  merits). This ADDS entries on a re-detect, so an already-transcribed book
  must not be re-detected wholesale for it (it can also split a caption
  line) — dry-run and take only the pages that gain a missed balloon.
- A short balloon of THREE stacked short lines whose letters FUSED in the
  upscale has too few blobs for `paragraph_blocks` and was taller than the
  short-utterance pass's one-or-two-line cap. The taller cap (three lines
  of the cluster's own letter height) is guarded on every side, because
  raised alone it admits mostly art (armour, faces, splatter, hatching):
  the cluster must still be UNDER MIN_LETTERS, at least three blobs, inked
  in solid strokes (ink >= 0.3 of the blobs' boxes — linework fills far
  less), in a lobe JOINED to an already-detected balloon (its "comp"
  host), and above or below the host's text — level with it, it is a piece
  of those lines (a caption's column of small-caps initials).
- Two balloons or caption boxes set CLOSE can have their lettering nearer
  than `paragraph_blocks`' join (stacked: a gap no wider than a line gap;
  side by side: a short neighbour's line running on from this one's), so
  both fuse into ONE block. The block then picks one light component, the
  other balloon never becomes an entry (no key, no mask, original lettering
  left standing and partly wiped by the neighbour's clean), and no gate
  ever rejected it — nothing downstream can see it is missing.
  `split_by_balloon` splits such a block when its letters vote (by the
  paper in a thin ring round each) for two or more light components with
  MIN_LETTERS each. Three guards, each from a real false split: the groups
  must not INTERLEAVE (letters of one group inside the other's blocks), each
  group's blocks must lie INSIDE its component's box (halftone and hatched
  art votes for the paper scraps between dots), and the area the old block
  covered stays CLAIMED for the short-utterance pass, or the art marks it
  swept up come back as new entries. A compound balloon's lobes are ONE
  component and never split here. It ADDS entries: on a transcribed book,
  re-detect only the pages that gain a missed balloon, and remap their keys.
  It can also REGROUP at the same count (the freed lines join the balloon
  they are joined to as a second paragraph, while the balloon that held
  them loses its paragraph), which shifts the positional order without
  moving the count — so a re-detect guard compares each bNN's box with the
  old bNN's, never the count alone.

## Joined and compound balloons
- Joined/overlapping balloons are SEPARATE entries, font-grouped; their
  masks resolve as: containment (front lobe keeps its body) → ink-interface
  test with letters EXCLUDED (front balloon keeps its full oval) → notch
  split of the union at the shallowest column/row between the blocks.
  Never a plain midline split (flat tops, stolen sibling area).
- A compound balloon carries ONE PARAGRAPH PER LOBE. `find_bands` cuts the
  row-width profile for stacked lobes; lobes side by side (a diagonal run)
  have no waist — a row span is a single x-range — so `mask_lobes` falls
  back to distance-transform peaks with nearest-peak assignment. Missing
  this crams every paragraph into one lobe at a tiny size and leaves the
  other lobes empty. When the mask's lobes ARE side by side (their x ranges
  overlap by under a quarter of the narrower: `lobes_side_by_side`), use
  them BEFORE `find_bands`: two balloons that also share rows — one sits a
  little lower — still have a waist in the union's row profile, and its
  lower band spans both balloons, so that paragraph is set in the gap
  between them, over whatever balloon sits there.
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
- A `lobes` rectangle also LIMITS WHAT IS CLEANED, so one rectangle around
  part of a balloon re-letters that part and keeps the rest (a display word
  kept original beside dialogue re-set in our font). Its edges must not
  graze the lettering it is meant to clean: the interior is hole-filled, a
  letter touching the rectangle's edge is not a hole, and it falls outside
  both the interior and the lobe's block — that letter is never wiped and
  the new text prints over it. Put the edge THROUGH what is to be kept (a
  piece crossing the edge is excluded the same way) and check the job's
  block spans the whole line before fitting.
- `"oval": true` on a `lobes` override is for an oval balloon whose outline
  has GAPS (a dashed whisper over pale art the walls cannot tell from the
  fill): the rectangle's flood runs out between the dashes into its
  corners, and the cleaner paints sky and wall white there. The flood is
  clipped to the rectangle's ellipse a hair inside it (`OVAL_SCALE`), so
  MEASURE THE RECTANGLE ON THE FILL (near-white flood from the text), not
  on the outline — an ellipse that reaches the dashes wipes the ones that
  touch a panel border or a neighbour. Its jobs are cleaned as `dashed`.
  Measure it in page pixels, never by eye off a scaled preview.
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
- A joined pair's straight cut is wrong whenever the two blocks OVERLAP
  on the axis it cuts across (the window between them runs backwards), not
  only when they overlap on both: the lines' first letters go to the
  neighbour. Both go to `seam_split`.
- `sibling_letter_box`: the FIT never writes over a sibling's lettering
  that a joined balloon's mask covers (else a first line prints on the
  sibling's last). The sibling's LETTERS inside the mask, not its block.
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
- `letter_lobes`: a rival entry's letters never re-seed letters this entry
  already owns — a phantom entry ("" transcript) whose block is one of this
  entry's paragraphs empties that lobe and the compound is never re-found.
  A recognised strip pair also shares ONE cx (union of its blocks), not
  just one line grid.
- `notch_cut` judges "no waist" on the profile's VALUE: a plateau running
  to the window's end has its first minimum a few px short of the end, so
  an index test lets it through and the cut is clamped onto the second
  block's start — a leading "..." (no letters, so outside the block) goes
  to the other lobe.

## Masks and row profiles
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
- Where a mask LEAKED past the balloon, clip the fit's row profile (and the
  last rung's `rows_raw`) to the balloon's own fill — a leaked profile can
  be a rectangle far wider than the oval, so lines get set where the oval
  has long since narrowed.
- Row spans follow ONE connected run (`widest_run`), never first-to-last.
  Low in a balloon the fill splits into the body and the mouth of the tail;
  spanning that gap lets the wrap set its last lines out into the tail, hard
  against the outline.
- `repair_letter_bites`' "sealed" test reads the window edge as outside, and
  the rim channel between the eroded mask and the outline runs down a TAIL
  and out of the window: every rim pocket then counts as a leak (a line end
  welded to the arc survives). Only a narrow opening on the window edge
  (TAIL_APERTURE) is closed. Do NOT close narrow passages everywhere —
  tried: it moves many correct masks, and the leaks this guard exists for
  come straight back (they escape through a narrow outline gap, then open
  out).

## Cleaning
- Cleaning: bubble = row-median repaint of ink+halo ONLY (flat-filling the
  whole mask plateaus → reads as a white rectangle at straight mask edges)
  + overflow inpaint that skips blobs touching the window border; dark/tint
  captions = row-median wipe of dilated letter cores (else hollow ghosts);
  open/margin = letter-only inpaint. Kinds: bubble/dark/tint/open/margin
  (tint = dark text on tinted box, text stays black).
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
- The overflow inpaint's window is the bubble's OWN bbox, never the raw
  `block`: a blob cluster swallows the balloon next door, and the step then
  erases a neighbour's lettering that nothing types back — an EMPTY
  balloon. Measured: reaching past the bbox buys a handful of stray blobs a
  few px out and costs a neighbour's letters and line art; everything it
  legitimately cleans is inside the bbox. No margin helps — the balloons
  are adjacent.
- The ink cut-off leaves the lettering's PALE strokes (150-220 on a 250 fill)
  behind as faint letter-shaped specks — visible at 1:1 300dpi and invisible
  to letter_mask. A softer cut-off fixes it but ONLY inside the letterer's
  block AND inside the balloon's own fill (`solid`): bounded to the block
  alone it catches artwork through a leaked mask and paints white
  rectangles.
- THE PER-BUBBLE PREP HAS ONE HOME: `prepare_bubble` (with
  `page_caption_boxes` / `boxes_to_wipe` / `is_strip`). The fit's `main()`
  and `reclean_all.py` both call it. A hand-mirrored copy of that
  preparation in a caller means every new cleaning behaviour has to be
  copied there by hand, or the book-wide verification silently stops
  reproducing the pipeline. Add cleaning behaviour THERE, never in a caller.
- A balloon whose fill grades ALONG the row (announcer bursts: white glow,
  blue rim) cannot be repainted with one row median — every old line comes
  back as a too-white band and the letters at the blue ends as white
  ghosts. `fill_trend` estimates the fill in 2-D from the fill around the
  lettering, ONLY when the fill strays >= FILL_TREND_MIN from its own row
  median (white and hologram-striped balloons stay on the row median), and
  `fill_wall` widens the colour wall by the ground the letters themselves
  stand on (a blue rim can sit just over the wall from the white glow). A
  balloon leaked onto sky keeps the plain wall: its letters stand on white.

## Caption boxes
- A caption box on a FLAT COLOURED fill (white on purple or blue, red on
  yellow) is invisible to every letter pass: white-on-black wants a black
  ring, the paper and tint passes want paper or letters dark in EVERY
  channel. It comes back missing, or as strips seeded on a few letters,
  with no key for the rest. `flat_caption_boxes` judges a letter by its
  ring being ONE flat colour (median, most within FLAT_TOL, coloured:
  spread >= FLAT_SPREAD); dark/coloured letters only on a VIVID fill —
  muted tan or sky with dark marks is art. Grow over fill OR ink but only
  while the step is mostly fill colour (a light frame line stops it), then
  PEEL to the fill, alternately: ascenders above the box first make the
  sideways steps cross art rows. A box must END at its frame (just outside
  every side the fill colour is gone); a flat patch of art fades into more
  of itself. The bbox stays INSIDE the frame because the 'dark' wipe takes
  every white pixel in it — a white frame inside the bbox would be erased.
  Small flat patches with white marks are everywhere in art (faces,
  stripes, sign fragments): a box needs FLAT_MIN_BOX. Not a higher letter
  count: in a narrow italic only some letters pass the ring test, so a real
  one-word caption seeds a 5-letter block.
  A drawn frame does NOT separate boxes from art — the line just outside a
  real box is frame + art, no more uniform than art's own edges.
- A caption box never holds a BALLOON. A flat-coloured PANEL (a sign, a
  screen) with a paper balloon over it passes every box test, and replacing
  would delete the balloon: refuse a box when any entry it would replace
  has its letters on paper. Judge the very entries the replace step takes
  (block area inside) — a containment test of their bbox misses a balloon
  that sticks out of the panel. Paper balloons measure ~0.75 light, strips
  of white type on colour <= 0.5.
- The flat box REPLACES the entries seeded inside it by the AREA of their
  letter blocks over all the boxes, not by the block's centre: one entry
  seeded across two neighbouring boxes has its centre in the gap and
  survives beside them, lettered over both. White letters -> 'dark' (the
  row-median wipe works on any fill colour); coloured letters -> 'tint',
  whose ink is `tint_ink` — dark in every channel OR far below the fill's
  luma (a red letter's red channel is as bright as yellow). One predicate
  for the wipe, the cap height and the original colour: orig_ink_rgb's
  general cut (weakest channel dark) takes a whole yellow fill for ink.
- `clean_caption_box` estimates the fill under a word by INTERPOLATING across
  it from the fill either side, per row. A morphological close is right for
  FINDING the ink (it is only a threshold) and wrong for replacing it: a
  close takes the BRIGHTEST value in its window, and the window has to be as
  wide as a whole word, so on a box whose fill grades along the row it reads
  the bright end as the fill everywhere and repaints the line as a flat,
  too-bright slab with straight edges (20-45 levels bright). Measure the fill
  on `~(ink | halo)` — never on pixels merely excluded from the repaint, or
  the spared frame drags the estimate ~100 levels dark beside it.
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
- `clean_caption_box` finds ink on the MAX channel as well as the min: a
  saturated yellow's blue channel sits by the ink (89 vs a letter's 65), so
  the min-channel test misses line ends there, and it also turns up a long
  band of FILL that a touching letter fused with and was spared as a
  "curve". Curves are classified per test, and a min-only band never spares
  what the max channel calls lettering.
- A SOLE-BOX caption keeps its HEIGHT: the writing area's inset is 2% of
  the box's WIDTH, and taken vertically on a wide one-line strip it removed
  a third of the height from each side — the caption came out at half the
  book size in a box with room for all of it. The vertical inset is capped
  at a tenth of the box's height.
- A graded fill that turns DARK at one edge reads as frame there (under
  FRAME_SOFT and touching black), the flood stops short of that edge and
  the box fails the rectangle test — it then comes back as strips that do
  not agree on x, each typeset on its own. `regrow_fill` gives the fill
  back the wall pixels just under FRAME_SOFT (`REGROW_BAND`) that continue
  it smoothly, only inside the flood's own bounding box, and only as a
  SECOND attempt when the plain flood finds no box. Applied always, it took
  the inner pixels of faint frames for fill and failed their four-sides
  test; unbounded, or reaching down towards FRAME_MAX, it flowed back out
  through the hairline gaps `frame_wall` seals and through grey frames.
  Either way real boxes were lost by the dozen across a book — run the
  frame finder over every page of every book before and after any change
  here.
- A framed caption detection holds whole as ONE balloon-kind entry is a box
  too (`plain` in `auto_caption_boxes`): fitted as a balloon, its lines
  follow the mask's rows and the letterer's per-line axes, so a long
  caption wraps ragged inside a rectangle and its first line can sit alone
  off to one side.
- ALIGNMENT follows the letterer: where his lines share a start and their
  ends are ragged (`orig_flush_left`: 3+ lines, starts within 0.6 cap, ends
  spread 2+ cap) our lines are set flush left at his edge
  (`flush_left_lines`, each kept inside its own row span). Centred type has
  both edges ragged and a justified block neither. Read his letters with
  the entry's own ink (`orig_ink`): on a yellow fill "dark in every
  channel" is the whole box.

## Fit, sizing and placement
- One long word must never shrink a bubble: `hyphenate`/`split_word` break
  it (prefer a vowel|consonant point, ≥2 chars a side, reuse an existing
  hyphen). Cap the fragments (MAX_PIECES) and return the words UNTOUCHED
  past it — at the top of the size search every word is too wide, and
  shattering them all runs the line-break DP over fragments for minutes.
- The letterer's block is THE anchor, so clip it (`block_anchor`) to the
  balloon's rows AND its bbox: the blob cluster swallows artwork beside the
  balloon (fur, a panel line) and slides text clean out of the lobe.
- Keep the new lettering clear of the drawn outline (`inset_rows`,
  ~3.5% of the profile's extent) — the row profile runs to the mask edge and
  the mask includes the outline stroke, so type set at the book size lands
  ON the line. Apply it to the FIT ONLY: inset the CLEANING and rim letters
  survive.
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
- Two cheap size metrics catch "text too small" before the user does:
  `new_cap_h < old_cap_h` (the balloon demonstrably had room — the pages a
  reviewer reports are reliably among the worst of these), and dialogue set
  well under the book size.
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
- A balloon the artist ran off the PAGE has no outline there — its
  interior is the edge — so no inset keeps the type off it and the first
  line printed flush against the page border. `clear_page_edge` holds the
  row profile a fraction of the original cap height clear of the page's
  edges, as an outline would.

## Text parsing and transcripts
- `parse_runs` resolves style PER CHARACTER, then splits on whitespace. A
  regex alternation cannot: `\S+` swallows `¡--*WORD*` whole and prints
  literal asterisks. `\*` is a literal asterisk (a footnote mark) and never
  toggles emphasis; a lone unmatched `*` also stays literal, but only when
  no other `*` follows it.
- A token with no letter in it ("...", "¡...", "--") is a word of its own
  to the wrap, and the balance pass breaks after it, printing a lone "..."
  over the first line. `glue_bare_punct` (end of `parse_runs`) glues it to
  the next word (the last one to the word before), and `split_word` never
  breaks at or beside the glue.
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

## Verification
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
