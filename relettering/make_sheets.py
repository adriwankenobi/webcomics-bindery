#!/usr/bin/env python3
"""Pack the per-bubble crops into labeled contact sheets for transcription.
Writes sheets/sheet-NNN.jpg + sheets/manifest.json mapping short page keys
("1-012", "7-045") to full page stems and bubble counts."""

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAXW, MAXH = 880, 2350


_KEY_RE = re.compile(r"^\D*(\d+(?:\.\d+[a-z]?)?)\b.*?(\d+(?:-\d)?)$")


def short_key(stem: str) -> str:
    # "<series> N - <title> PPP" -> "N-PPP"; fall back to the full stem.
    # A dotted issue number ("UK 6.29 002") is kept whole: "6.29-002",
    # letter suffix included ("UK 7.02a 003" -> "7.02a-003").
    m = _KEY_RE.match(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem


_ISSUE_RE = re.compile(r"^\D*(\d+(?:\.\d+[a-z]?)?)\b")


def issue_of(stem: str) -> str:
    # the issue part of short_key ("6.01", "15", "7.02a"): the unit the
    # keep-or-re-letter decision is made on. A page short_key cannot parse
    # because it does not END in its page number ("... 002 cover",
    # "... 072a") is still its issue's page; a stem with no number at all
    # is an issue of its own.
    m = _KEY_RE.match(stem) or _ISSUE_RE.match(stem)
    return m.group(1) if m else stem


def check_unique(counts: dict) -> None:
    """{stem: bubble count}. The transcripts are keyed on short_key, so two
    pages with bubbles sharing one would silently overwrite each other."""
    seen = {}
    for stem, n in counts.items():
        if not n:
            continue
        key = short_key(stem)
        if key in seen:
            sys.exit(f"make_sheets: pages {seen[key]!r} and {stem!r} both "
                     f"map to key {key!r} — fix short_key")
        seen[key] = stem


def plan_pages(bub_dir: Path, work: Path) -> dict:
    """{stem: bubble count} of the pages that get sheets: every page with
    bubbles, minus the issues the keep-or-re-letter plan keeps (those are
    never transcribed)."""
    from issue_plan import kept_stems
    counts = {j.stem: len(json.loads(j.read_text()))
              for j in sorted(Path(bub_dir).glob("*.json"))}
    counts = {s: n for s, n in counts.items() if n}
    kept = kept_stems(work, counts)
    return {s: n for s, n in counts.items() if s not in kept}


def tiles_for(bub_dir: Path, stem: str, count: int) -> list:
    import cv2
    key = short_key(stem)
    tiles = []
    for bi in range(1, count + 1):
        crop = cv2.imread(str(bub_dir / f"{stem}-b{bi:02d}.png"))
        if crop is None:
            continue
        if crop.shape[1] > MAXW:
            sc = MAXW / crop.shape[1]
            crop = cv2.resize(crop, (MAXW, int(crop.shape[0] * sc)))
        crop = cv2.copyMakeBorder(crop, 26, 4, 4, 4,
                                  cv2.BORDER_CONSTANT,
                                  value=(255, 255, 255))
        cv2.putText(crop, f"{key} b{bi:02d}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
        tiles.append(crop)
    return tiles


def render_sheets(tiles: list, sheets_dir: Path, start: int) -> int:
    """Pack tiles into sheet-NNN.jpg from number `start`; returns how many
    sheets were written."""
    import cv2
    import numpy as np
    sheets, cur, curh = [], [], 0
    for t in tiles:
        if curh + t.shape[0] > MAXH and cur:
            sheets.append(cur)
            cur, curh = [], 0
        cur.append(t)
        curh += t.shape[0]
    if cur:
        sheets.append(cur)
    for si, sh in enumerate(sheets, start):
        W = max(t.shape[1] for t in sh)
        sh = [cv2.copyMakeBorder(t, 0, 0, 0, W - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(235, 235, 235))
              for t in sh]
        cv2.imwrite(str(Path(sheets_dir) / f"sheet-{si:03d}.jpg"),
                    np.vstack(sh), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return len(sheets)


def build(work: Path) -> None:
    """Make the sheets. Without a keep-or-re-letter plan (a comic fitted
    before that step existed): every page, from scratch, as always. With a
    plan: kept issues get no sheets, and when a manifest already exists it
    is SYNCED to the plan — pages of issues flipped to keep leave it, pages
    of issues flipped to re-letter get new sheets numbered after the last
    one, and no existing sheet is touched (transcription may be under way
    against them)."""
    from issue_plan import load_plan
    BUB = work / "bubbles"
    SHEETS = work / "sheets"
    MANIFEST = SHEETS / "manifest.json"
    counts = {j.stem: len(json.loads(j.read_text()))
              for j in sorted(BUB.glob("*.json"))}
    check_unique(counts)
    plan = load_plan(work)
    SHEETS.mkdir(exist_ok=True)
    if plan is not None and MANIFEST.is_file():
        want = plan_pages(BUB, work)
        manifest = {k: v for k, v in json.loads(MANIFEST.read_text()).items()
                    if v["stem"] in want}
        have = {v["stem"] for v in manifest.values()}
        start = 1 + max((int(p.stem.split("-")[1])
                         for p in SHEETS.glob("sheet-*.jpg")), default=0)
        add = [s for s in want if s not in have]
    else:
        want = (plan_pages(BUB, work) if plan is not None
                else {s: n for s, n in counts.items() if n})
        for old in SHEETS.glob("sheet-*.jpg"):
            old.unlink()
        manifest, start, add = {}, 1, list(want)
    tiles = []
    for stem in sorted(add):
        manifest[short_key(stem)] = {"stem": stem, "count": want[stem]}
        tiles += tiles_for(BUB, stem, want[stem])
    n = render_sheets(tiles, SHEETS, start) if tiles else 0
    MANIFEST.write_text(json.dumps(manifest, indent=0))
    print(f"{len(tiles)} crops -> {n} sheets"
          f"{f' (from sheet-{start:03d})' if start > 1 else ''}, "
          f"{len(manifest)} pages")


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: make_sheets.py "<comic folder name>"')
    build(REPO / "relettering" / sys.argv[1])


if __name__ == "__main__":
    main()
