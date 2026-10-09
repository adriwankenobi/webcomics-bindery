# QA + rebuild tooling (built during the Sep 13-14 2026 feedback rounds)

All scripts assume CWD = repo root and take **the comic name as argv[1]**,
like every other script in the pipeline:

    OUTDIR=/tmp/x .venv-reletter/bin/python \
        relettering/qa-tools/reclean_all.py "<comic folder name>"

Everything else still comes from env vars (`SP`, `PAGES`, `LAYOUT`, `CAND`,
`PGS`, `STEM`/`BI`, ...). `SP` defaults to an old session scratchpad — point
it at the current one (it only receives output PNGs/JSON).

The two scripts that re-run the fit (`rebuild_fit.py`, `refit_subset.py`)
require **`BOOK=<size>`**. It used to default to 23, which is this book's
size and no other's; an UNPINNED re-fit re-derives the book size from the
percentile of per-bubble maxima and silently moves already-approved pages,
so the pin is now explicit or the script refuses to run.

**Tracked vs scratch.** The verification gates and the rebuild drivers are
process — CLAUDE.md requires them on every batch and a book cannot be
verified without them — so they are whitelisted in `.gitignore`. The eyeball
helpers beside them (`crop`, `zoom`, `ozoom`, `overlay`, `montage`, `qamont`,
`mdiag`, `preview`, `bub`, `outside`, and the per-round one-offs) stay
untracked scratch: regenerate them per round rather than maintaining them.

`reclean_all.py` no longer reimplements the fit's per-bubble preparation. It
calls `reletter_fit.page_caption_boxes` / `boxes_to_wipe` / `prepare_bubble`
— the same functions `main()` runs — so it cannot drift from the pipeline it
is supposed to be verifying. Add a cleaning behaviour to `prepare_bubble` and
this tool gets it for free; that used to mean a hand-copy here, four blocks
of one by the end.

## Verification (run these on EVERY batch — see CLAUDE.md)
- `leftover.py`            re-runs detection's letter_mask on the CLEANED pages;
                           any letter blob where the transcript has text =
                           failed cleaning. Healthy: <20 book-wide.
                           `PAGES=<dir>` to scan a candidate dir.
                           Searches the mask UNION the text block UNION every
                           laid-out line's box. Until round 4 it intersected
                           with the MASK alone, which hid the worst failure
                           mode there is — lettering the mask never covered is
                           precisely what puts new text on top of old — and it
                           reported "14 blobs, healthy" while p170 and p147
                           carried whole lines of old text under the new.
                           STILL BLIND TO: text on a coloured fill (letter_mask
                           is a "dark text on light paper" ring test), and
                           faint residue above the ink cut-off. Look at pages.
- `qagate.py`              qa_scan's validated artifact gate against an
                           arbitrary cleaned dir (`CAND=<dir>`).
- `protrude.py`            per-line ink box vs the bubble MASK (text outside).
                           Blind by construction to the cases users report: a
                           leaked mask is exactly what puts text outside the
                           balloon, and protrude.py measures against that same
                           mask, so it flagged none of p45/p65/p127. Measure
                           against the balloon's own fill (bubble_interior of
                           the pristine page) instead — that ranks p65 b07 15th
                           of 67 and finds the unreviewed 168/170/253/219.
- `preview.py` / `montage.py`
                           render layout.json's text onto the cleaned page with
                           PIL — placement check in SECONDS, no GIMP pass.
- `qamont.py`              full-page ORIGINAL|FINAL from the fresh qa/ renders.
- `overlay.py`             EXACT 1:1 300dpi page-PDF render vs the source
                           (ghostscript), the check CLAUDE.md requires.
- `bub.py` / `crop.py`     per-page bubble dump; original-vs-result crops.
- `gate_split.py`         qa_scan's artifact gate SPLIT by caption-box
                          membership. Wiping a caption box whole is one large
                          contiguous change, which is_artifact cannot tell
                          from damage — count those separately and compare
                          only the blobs OUTSIDE a box (`CAND=<dir>`).
- `outside.py`            text outside the BALLOON'S OWN FILL (what
                          protrude.py cannot see — it measures against the
                          mask, and a leaked mask is the fault itself).
- `faint_scan.py`         faint letter residue above the ink cut-off, which
                          letter_mask is blind to (`PAGES=<dir>`).
- `unrelettered.py`       lettering that was never RE-SET: an entry the
                          cleaner never touched with no new line over it
                          (`PAGES=<dir>`). Neither other gate can see this —
                          leftover.py only looks where the transcript HAS
                          text (an empty one means "leave it", by design) and
                          qa_scan diffs against pristine, where an untouched
                          region is identical. Round 7 found 68 entries on 49
                          pages this way, incl. 5 of p287's 6 caption boxes:
                          detection splits a graded box into strips, so the
                          transcription sheet showed each line cut in half and
                          the strip was blanked. Droid speech (p204) is
                          deliberately original — check the list by eye.
- `capbox_scan.py`        every laid-out line of a caption box measured
                          against the BOX rectangle (`PGS=` to narrow).
                          outside.py cannot stand in: it skips `tint` strips
                          and measures against bubble_interior, whose flood on
                          a graded fill is the thing that was mis-detected.
- `boxleft.py`            old lettering still standing inside a drawn caption
                          FRAME on the cleaned pages (`PAGES=`, `STEMS=`).
                          leftover.py is blind to it (a yellow fill is not
                          light paper) and so is qa_scan (an untouched strip of
                          a box is identical to pristine): a box detection only
                          part-found keeps the lines nobody cleaned, with the
                          new text squeezed in beside them. Healthy: 0. Still
                          blind to residue already wiped into sub-letter
                          fragments — look at the box.
- `welded.py`             stranded lettering FUSED to a drawn line, which
                          leftover.py is blind to by construction (a ring test,
                          and a letter fused to the outline has no ring) and
                          qa_scan cannot see (the pixels never changed).
                          Over-reports — see the calibration note in the file;
                          use it as a delta, not an absolute.
- `three.py`              PRISTINE | SHIPPED | CANDIDATE for one bubble at 1:1
                          (`CAND=<dir>`) — the fastest way to judge a cleaning
                          change without a GIMP pass.
- `split_caps.py`         enumerate the coloured caption boxes detection
                          split into a tint strip + a bubble strip.
- `mdiag.py`              pristine | cleaned | mask-overlay for ONE bubble
                          (`mdiag.py <page> <bi> [pad] [out.png]`) — the
                          fastest way to see a mask that is short or leaked.
- `zoom.py` / `ozoom.py`  magnified original-vs-result: `zoom.py` from the
                          half-res qa render (fast, but its softness is an
                          ARTEFACT — never judge ghosting from it), `ozoom.py`
                          from the real 1:1 300dpi page PDF (authoritative).

All of these take `SP=<scratch dir>` for their output (default
/tmp/relettering-qa).

## Rebuild loop
- `reclean_all.py`         re-clean every page from pristine into `OUTDIR=<dir>`
                           (1.4 min) — fast cleaning-only feedback, touches
                           nothing in the pipeline.
- `rebuild_fit.py`         the REAL reletter_fit.main() against staged pristine
                           copies with `BOOK=<size>` pinned (~95 min).
- `refit_subset.py`        re-fit only `PGS=5,29,...` (drawn page numbers) into
                           a scratch WORK dir — minutes, not 95. Works because
                           a pinned book size makes the fit separable per page.
- `refit1.py`              re-fit ONE bubble (`STEM=`, `BI=`) and print lines.
- `promote.py`             promote ONLY pages whose layout entries or cleaned
                           pixels changed (`APPLY=1`); untouched pages keep
                           their mtime, which is what stops compose rebuilding
                           the whole book. Diffs against `layout.json.bak-*`.
- `redetect_dry.py`        DRY-RUN re-detection vs the current bubbles JSON;
                           flags bubble-COUNT changes (which invalidate the
                           positional transcripts) and block-row coverage.
- `redetect_apply.py`      apply re-detection (`PGS=ALL`, ~8 min book-wide);
                           backs up originals and REFUSES any count change.

## Two traps that cost hours
- Stage pages under their REAL on-disk filenames. transcripts.json is keyed on
  the NFD spelling; an NFC-normalised copy makes every lookup miss SILENTLY
  and the fit collects zero bubbles.
- `reletter_fit.main()` crashes on an empty `gains` list at the very END (after
  layout.json is written) when no bubble fits — harmless, pre-existing.

## Round 7 additions (Sep 18 2026)
- `dump.py <pages>`       per-page dump: every bubble's kind/bbox/block/group
                          beside its layout entry and its transcript. The
                          first thing to run on a reported page.
- `tsheet.py`             labelled crops for transcription:
                          `ITEMS=287:1,287:2 VPAD=75 ZOOM=1.6 MAXW=840`.
                          A split caption box is only HALF a box — pad
                          vertically or the first line is cut off and the
                          strip reads as unintelligible (which is why those
                          strips were blanked in the first place).
- `boxrect.py <p:bi,...>` measure a caption box's drawn frame and emit the
                          rect for `layout_overrides.json`.
- `merge_layout.py`       merge a subset re-fit (`work3/layout.json`) into the
                          book's layout for the staged stems ONLY, reporting
                          what moved (`APPLY=1` to write).
- `checkfit.py`           post-refit checks: do box strips share one size,
                          did anything become unplaced, the size gate, and a
                          per-page diff for the reported pages.
- `trace_reclaim.py`      `PG= BI=` — trace the weld repair on one bubble
                          (protect components, hull depth, what is reclaimed).
