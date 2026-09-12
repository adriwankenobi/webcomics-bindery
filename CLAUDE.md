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
  around them, don't edit them. The re-lettering scripts ARE the place for
  algorithm fixes; prefer code over hand-edited masks (reproducibility).
- Re-lettering code lives in relettering/ (reletter_detect/fit/gimp.py,
  qa_scan, merge_transcripts, make_sheets). Every
  script takes the comic name as argv[1] (reletter_gimp.py: RELETTER_COMIC
  env var, run from the repo root). Git tracks ONLY the process (whitelist
  .gitignore); all comic data — sources, upscaled/, xcf/, pdf/,
  relettering/<comic>/ (transcripts included) — is untracked and must be
  reproducible from the source folder alone. No comic-specific content in
  tracked files. Per-comic editorial data = untracked JSONs in
  relettering/<comic>/: typo_fixes.json, layout_overrides.json
  ({"size": N} | {"anchor": "visual"} | {"anchor": "lines"}), parts/zz-*.json
  transcript overrides (merge order = sorted filename, later wins; keep
  sheets/manifest.json counts in sync).
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
- Verify EVERY batch: (1) leftover scan = detection on the cleaned working
  pages, hits where transcripts have text = failed cleaning; (2) qa_scan
  pixel-diff vs relettering/<comic>/pristine/ (text-shaped footprints only;
  also eyeball blobs taller than a text line — wedges slip the 4000/35
  gate); (3) EXACT 1:1 300dpi render overlays vs the source — fraction
  crops hide 10-20px placement errors the reviewer sees.
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
