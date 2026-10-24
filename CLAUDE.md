# CLAUDE.md

Print pipeline for webcomics (language-agnostic for LTR, space-separated,
Latin-script-like languages; all text lives in per-comic data). README.md =
the flow, ARCHITECTURE.md = how everything works inside, GOTCHAS.md = the
hard-won rules behind the code, by topic. Before changing detection,
cleaning, caption boxes, the fit or the upscale/crop step, read that
topic's section of GOTCHAS.md. Key facts a session needs:

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
  ({"size": N | "orig"} | {"color": "orig" | [r,g,b]} | {"anchor":
  "visual"} | {"anchor": "lines"} | {"box": [x0,y0,x1,y1]} | {"lobes":
  [[x0,y0,x1,y1], ...]} (+ "oval": true for a gapped outline)),
  fit_policy.json
  (per-book typographic calls an approved book must not inherit:
  {"frame_pairs": true} = a split caption pair in a frame is one box),
  parts/zz-*.json
  transcript overrides (merge order = sorted filename, later wins; keep
  sheets/manifest.json counts in sync).
- The docs (this file, README.md, ARCHITECTURE.md, GOTCHAS.md, the qa-tools
  README) explain HOW things work and WHY — never which comic, book, page
  or bubble showed it, incident counts, round numbers, dates or quoted
  dialogue. That evidence belongs in commit messages; per-book state
  belongs in memory. Tests use placeholder names ("Series 13 - Title 012"),
  never real titles or transcript text.
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
- KEEP OR RE-LETTER is decided per ISSUE after detection, BEFORE the sheets
  (`reletter_decide.py`, rule in `issue_plan.py`; how it decides:
  ARCHITECTURE.md Stage 1.5, rules: GOTCHAS.md). `issue_decisions.json` is
  written ONCE; `issue_overrides.json` wins. Never create a decisions file
  (or fit_meta.json) in an approved comic. LAYOUT.JSON is the only
  authority on which pages are fitted.
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

## Rules most re-lettering tasks hit (full reasoning in GOTCHAS.md)
- Transcripts are POSITIONAL: one string per bubble in bNN order, keyed
  `"<short> bNN"` (`make_sheets.short_key`, never a copy of its regex).
  Detection changes are only safe while the bubble COUNT and ORDER per
  page hold: re-detect into a scratch compare first and refuse any page
  whose count moved or whose entries regrouped at the same count
  (redetect_apply.py checks both).
- Verify EVERY batch with the three gates — leftover scan (mask UNION block
  UNION every laid-out line's box), qa_scan vs pristine/, and EXACT 1:1
  300dpi render overlays vs the source. Never substitute a proxy metric.
  Cleaning changes need a full re-clean (`reclean_all.py`) measured with
  `faint_scan.py`.
- Cleaning behaviour has ONE home, `prepare_bubble`; never mirror it in a
  caller.
- REBUILDING AN APPROVED PAGE IS NOT A RE-FIT: per entry take the new fit
  where it is larger or identical, otherwise carry the shipped entry
  through the page-geometry map. Re-upscaling remaps transcripts through
  the SOURCE (old page -> source -> new page).
- MEASURE THE PDF, NOT THE PLAN: a metric computed from a model must be
  checked once against the rendered artefact.
- THE ORIGINAL LETTERER IS THE GOLD STANDARD for size and placement; users
  judge centering against the visible lobe and the original page — verify
  by rendering, never by layout numbers alone.

## Environments
- Base pipeline: `.venv/bin/python` (pillow, pikepdf, numpy, playwright).
- Upscaler: untracked third-party binary at
  tools/realesrgan/realesrgan-ncnn-vulkan + models/ (install steps in
  README "Setup"). Pass `-m <models dir>` by hand or it exits 1 silently.
- Re-lettering: `.venv-reletter/bin/python` (opencv-python-headless, pillow,
  numpy, fonttools). cv2 is ONLY here (and in .venv-untext).
- Text removal (`just untext <image>`, standalone untext.py):
  `.venv-untext` from `just setup-untext` — torch 2.2.2 (last Intel-Mac
  wheels, needs numpy<2), easyocr, cv2; LaMa + CRAFT models untracked in
  tools/lama/, tools/easyocr/. Writes `-untexted` + `-untexted-mask.png`
  BESIDE the input (user ruling) — never run it on a comic source folder's
  image: every image there is a page. Its tests skip under .venv.
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
