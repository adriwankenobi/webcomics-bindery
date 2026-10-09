"""BOOK-WIDE caption-box leftover scan: old lettering still standing inside
a drawn caption FRAME on the cleaned working pages.

leftover.py cannot see this (letter_mask is a dark-on-light ring test, and a
yellow fill is not light paper) and qa_scan diffs against pristine, where an
untouched strip of a box is identical. A box detection only part-found keeps
the lines nobody cleaned, with the new text squeezed in beside them — 10-076's
top two lines shipped that way and no gate flagged them.

For every frame `frame_box` finds on the PRISTINE page around an entry with
text, count letter-sized ink pieces inside it on the CLEANED page, ignoring
pieces that touch the frame's inner edge (frame shading, not lettering).
PAGES=<dir> scans a candidate dir; STEMS='a|b' limits the pages."""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

if len(sys.argv) < 2:
    sys.exit('usage: boxleft.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.path.insert(0, 'relettering')
sys.argv = ['reletter_fit.py', COMIC]
import reletter_fit as F  # noqa: E402

F.configure(COMIC)
R = Path('relettering') / COMIC
PAGES = Path(os.environ.get('PAGES', f'upscaled/{COMIC}'))
only = set(os.environ['STEMS'].split('|')) if os.environ.get('STEMS') else None
tr = json.loads((R / 'transcripts.json').read_text())
lay = json.loads((R / 'layout.json').read_text())
EDGE = 4


def letters_in(img, box):
    x0, y0, x1, y1 = box
    ink = (img[y0:y1, x0:x1].max(axis=2) <= F.DARK_MAX).astype(np.uint8)
    k, _, st, _ = cv2.connectedComponentsWithStats(ink, 8)
    h, w = ink.shape
    return sum(1 for j in range(1, k)
               if 9 <= st[j, 3] <= 48 and st[j, 4] >= 8
               and st[j, 0] >= EDGE and st[j, 1] >= EDGE
               and st[j, 0] + st[j, 2] <= w - EDGE
               and st[j, 1] + st[j, 3] <= h - EDGE)


total = 0
for stem in sorted(lay):
    if only and stem not in only:
        continue
    cp = PAGES / f'{stem}.jpg'
    pp = R / 'pristine' / f'{stem}.jpg'
    if not cp.exists() or not pp.exists():
        continue
    pr = cv2.cvtColor(cv2.imread(str(pp)), cv2.COLOR_BGR2RGB)
    cl = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
    if cl.shape != pr.shape:
        continue
    bubs = json.loads((R / 'bubbles' / f'{stem}.json').read_text())
    seen = set()
    for e in bubs:
        if e['kind'] == 'dark':
            continue
        box = F.frame_box(pr, e['block'], e['bbox'])
        if box is None or box in seen:
            continue
        seen.add(box)
        x0, y0, x1, y1 = box
        mem = [j for j, f in enumerate(bubs, 1)
               if x0 <= f['block'][0] + f['block'][2] / 2 < x1
               and y0 <= f['block'][1] + f['block'][3] / 2 < y1]
        if not any(tr[stem][j - 1].strip() for j in mem):
            continue                    # a box kept original on purpose
        left = letters_in(cl, box)
        if left >= 3:
            total += 1
            print(f'{stem} box {list(box)} members {mem}: {left} letters '
                  f'left (pristine {letters_in(pr, box)})')
print(f'caption boxes with old lettering left: {total}')
