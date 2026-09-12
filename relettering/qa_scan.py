#!/usr/bin/env python3
"""Post-build QA: (1) artifact scan — diff each cleaned upscaled page
against the pristine (us trade) copy; every repaint footprint must be
text-shaped (no blob with area>4000 and min-dim>35). (2) coverage stats."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
if len(sys.argv) < 2:
    sys.exit('usage: qa_scan.py "<comic folder name>"')
COMIC = sys.argv[1]
CLEAN = REPO / "upscaled" / COMIC
# pristine (pre-cleaning) pages, regenerated on demand from sources via
# `process.py upscale`
PRISTINE = REPO / "relettering" / COMIC / "pristine"

bad = 0
pages = 0
for f in sorted(list(CLEAN.glob("*.jpg")) + list(CLEAN.glob("*.png"))):
    src = PRISTINE / f.name
    if not src.is_file():
        continue
    a = cv2.imread(str(src)).astype(np.int16)
    b = cv2.imread(str(f)).astype(np.int16)
    if a.shape != b.shape:
        print(f"SIZE MISMATCH {f.name}")
        bad += 1
        continue
    d = (np.abs(a - b).max(axis=2) > 25).astype(np.uint8)
    if not d.any():
        continue
    pages += 1
    n, lab, st, _ = cv2.connectedComponentsWithStats(d, 8)
    for i in range(1, n):
        x, y, w, h, area = st[i]
        if area > 4000 and min(w, h) > 35:
            print(f"ARTIFACT {f.name}: blob at ({x},{y}) {w}x{h} area={area}")
            bad += 1

lay = json.loads((REPO / "relettering" / COMIC / "layout.json").read_text())
entries = sum(len(v) for v in lay.values())
print(f"\n{pages} pages modified, {entries} text layouts, "
      f"artifact scan: {'CLEAN' if not bad else str(bad) + ' ISSUES'}")
sys.exit(1 if bad else 0)
