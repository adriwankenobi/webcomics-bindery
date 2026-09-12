#!/usr/bin/env python3
"""Pack the per-bubble crops into labeled contact sheets for transcription.
Writes sheets/sheet-NNN.jpg + sheets/manifest.json mapping short page keys
("1-012", "7-045") to full page stems and bubble counts."""

import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if len(sys.argv) < 2:
    sys.exit('usage: make_sheets.py "<comic folder name>"')
COMIC = sys.argv[1]
WORK = REPO / "relettering" / COMIC
BUB = WORK / "bubbles"
SHEETS = WORK / "sheets"
MAXW, MAXH = 880, 2350


def short_key(stem: str) -> str:
    # "<series> N - <title> PPP" -> "N-PPP"; fall back to the full stem
    m = re.match(r"^\D*(\d+)\b.*?(\d+(?:-\d)?)$", stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem


def main():
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
