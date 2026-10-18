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


def short_key(stem: str) -> str:
    # "<series> N - <title> PPP" -> "N-PPP"; fall back to the full stem.
    # A dotted issue number ("UK 6.29 002") is kept whole: "6.29-002",
    # letter suffix included ("UK 7.02a 003" -> "7.02a-003").
    m = re.match(r"^\D*(\d+(?:\.\d+[a-z]?)?)\b.*?(\d+(?:-\d)?)$", stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem


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


def main():
    import cv2
    import numpy as np
    if len(sys.argv) < 2:
        sys.exit('usage: make_sheets.py "<comic folder name>"')
    WORK = REPO / "relettering" / sys.argv[1]
    BUB = WORK / "bubbles"
    SHEETS = WORK / "sheets"
    check_unique({j.stem: len(json.loads(j.read_text()))
                  for j in sorted(BUB.glob("*.json"))})
    SHEETS.mkdir(exist_ok=True)
    for old in SHEETS.glob("sheet-*.jpg"):
        old.unlink()
    manifest = {}
    tiles = []          # (key, bi, image)
    for j in sorted(BUB.glob("*.json")):
        bs = json.loads(j.read_text())
        if not bs:
            continue
        key = short_key(j.stem)
        manifest[key] = {"stem": j.stem, "count": len(bs)}
        for bi in range(1, len(bs) + 1):
            crop = cv2.imread(str(BUB / f"{j.stem}-b{bi:02d}.png"))
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

    sheets, cur, curh = [], [], 0
    for t in tiles:
        if curh + t.shape[0] > MAXH and cur:
            sheets.append(cur)
            cur, curh = [], 0
        cur.append(t)
        curh += t.shape[0]
    if cur:
        sheets.append(cur)

    for si, sh in enumerate(sheets, 1):
        W = max(t.shape[1] for t in sh)
        sh = [cv2.copyMakeBorder(t, 0, 0, 0, W - t.shape[1],
                                 cv2.BORDER_CONSTANT, value=(235, 235, 235))
              for t in sh]
        cv2.imwrite(str(SHEETS / f"sheet-{si:03d}.jpg"), np.vstack(sh),
                    [cv2.IMWRITE_JPEG_QUALITY, 86])
    (SHEETS / "manifest.json").write_text(json.dumps(manifest, indent=0))
    print(f"{len(tiles)} crops -> {len(sheets)} sheets, "
          f"{len(manifest)} pages")


if __name__ == "__main__":
    main()
