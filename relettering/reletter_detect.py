#!/usr/bin/env python3
"""Detect speech-bubble text in comic pages, text-first: letters are small
dark blobs of regular height grouped into lines/paragraphs; the bubble is the
light (white or pale-striped) region enclosing the paragraph. Outputs
bubbles/<page>.json + a numbered crop PNG per bubble for transcription/QA."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
COMIC = PAGES_DIR = OUT = SRC_DIR = None


def configure(comic):
    """Point the module at one comic. Called automatically at import when
    the comic is on argv (how the pipeline and the qa-tools invoke it), so
    the command line is unchanged; importing this as a LIBRARY — for
    detection's own `letter_mask`, which the leftover scan re-runs over the
    cleaned pages — calls it explicitly instead of faking sys.argv.
    """
    global COMIC, PAGES_DIR, OUT, SRC_DIR
    COMIC = comic
    PAGES_DIR = REPO / "upscaled" / comic
    OUT = REPO / "relettering" / comic / "bubbles"
    SRC_DIR = REPO / comic


# only when THIS is the program being run — see the note in reletter_fit
_argv0 = Path(sys.argv[0]).name if sys.argv else ""
if ((__name__ == "__main__" or _argv0.startswith("reletter_"))
        and len(sys.argv) > 1 and not sys.argv[1].startswith("-")):
    configure(sys.argv[1])

DARK_MAX = 110
LIGHT_MIN = 160
LETTER_H = (9, 48)        # px height of one letter (caps ~17-24 here)
LETTER_W_MAX = 110
LETTER_AREA_MAX = 2200
MIN_LETTERS = 8           # a real text block has at least this many blobs
PAD_FRAC = 0.03


def letter_mask(img: np.ndarray) -> np.ndarray:
    """Mask of letter-sized dark blobs that sit on a light background."""
    dark = (img.min(axis=2) <= DARK_MAX).astype(np.uint8)
    light = img.min(axis=2) >= LIGHT_MIN
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (LETTER_H[0] <= h <= LETTER_H[1] and w <= LETTER_W_MAX
                and area <= LETTER_AREA_MAX and area >= 8):
            continue
        # background around the blob must be mostly light (text on paper)
        y0, y1 = max(0, y - 3), min(img.shape[0], y + h + 3)
        x0, x1 = max(0, x - 3), min(img.shape[1], x + w + 3)
        ring = light[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] != i)
        if ring.mean() < 0.45:
            continue
        keep[i] = True
    return keep[labels].astype(np.uint8)


def white_letter_mask(img: np.ndarray) -> np.ndarray:
    """Letters of white-on-black caption boxes: white blobs on dark ground."""
    white = (img.min(axis=2) >= 200).astype(np.uint8)
    dark = img.max(axis=2) <= 90
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (LETTER_H[0] <= h <= 40 and w <= 100
                and 8 <= area <= 1800):
            continue
        y0, y1 = max(0, y - 3), min(img.shape[0], y + h + 3)
        x0, x1 = max(0, x - 3), min(img.shape[1], x + w + 3)
        ring = dark[y0:y1, x0:x1] & (labels[y0:y1, x0:x1] != i)
        if ring.mean() < 0.5:
            continue
        keep[i] = True
    return keep[labels].astype(np.uint8)


def paragraph_blocks(letters: np.ndarray) -> list:
    """Group letter blobs into paragraph blocks (dilate then components)."""
    joined = cv2.dilate(letters, np.ones((7, 25), np.uint8))
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE,
                              np.ones((21, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    blocks = []
    for i in range(1, n):
        x, y, w, h, _ = stats[i]
        count = int(cv2.connectedComponentsWithStats(
            letters[y:y + h, x:x + w], 8)[0]) - 1
        if count >= MIN_LETTERS and w >= 40 and h >= 14:
            blocks.append((x, y, w, h))
    return blocks


def follow_lobe(comp, bcy, ref, bw):
    """A bubble is horizontally convex and, leaving its widest row, can only
    narrow: walk from the seed row outward, each row following the ONE
    contiguous run that overlaps the previous row's span (never bridging
    gaps — that painted rectangles over art). Leaks through outline gaps are
    clamped to +-3px per row and cut when the run's center drifts away.
    Confines the mask to the seed's own lobe."""
    def runs_of(y):
        xs = np.flatnonzero(comp[y])
        if not len(xs):
            return []
        cuts = np.flatnonzero(np.diff(xs) > 1)
        starts = np.concatenate([[0], cuts + 1])
        ends = np.concatenate([cuts, [len(xs) - 1]])
        return [(int(xs[s]), int(xs[e]) + 1) for s, e in zip(starts, ends)]

    def pick_run(y, r0, r1):
        best, bestov = None, 0
        for r in runs_of(y):
            ov = min(r[1], r1) - max(r[0], r0)
            if ov > bestov:
                best, bestov = r, ov
        return best

    bcy = min(max(0, bcy), comp.shape[0] - 1)
    start = pick_run(bcy, ref[0], ref[1])
    if start is None:
        return None
    clipped = np.zeros_like(comp)
    TOL = 3
    start_cx = (start[0] + start[1]) / 2.0
    max_drift = max(25, 0.30 * bw)
    for step in (1, -1):
        cur = start
        y = bcy
        while 0 <= y < comp.shape[0]:
            r = pick_run(y, cur[0], cur[1])
            if r is None:
                break
            x0, x1 = max(r[0], cur[0] - TOL), min(r[1], cur[1] + TOL)
            if x1 - x0 < 12:
                break
            if abs((x0 + x1) / 2.0 - start_cx) > max_drift:
                break
            clipped[y, x0:x1] = comp[y, x0:x1]
            cur = (x0, x1)
            y += step
    return clipped


def own_block_mask(shape, origin, block, others=(), letters=None, pad=6):
    """`block` as a bool mask in a frame starting at `origin`, minus the part
    of it that holds ANOTHER entry's lettering.

    Every guard in here protects "a lobe's own text block" — the region its
    lettering has to go in, which may never be ceded or cut away. But a block
    is a blob cluster, and where two of them overlap the clustering could not
    tell whose lettering it is: p294 b07's block reached over b12's first
    line, so the protection held b12's "SI DESEAS" inside b07's mask, and
    with it the slice of b12's balloon b07's flood had filled.

    Subtract where the sibling's LETTERS are, not its whole rect. Two blocks
    can overlap by a sliver that holds no sibling lettering at all and is
    where this lobe's own last line ends — p294 b04's block overlaps b06's by
    30px, and dropping that sliver cost it five size steps. `letters` is the
    page letter mask in the same frame as `shape`; without it the whole rect
    goes, which is only ever right when there is nothing better to go on."""
    ox, oy = origin
    m = np.zeros(shape, bool)
    bx, by, bw, bh = block
    m[max(0, by - oy):max(0, by + bh - oy),
      max(0, bx - ox):max(0, bx + bw - ox)] = True
    for cx, cy, cw, ch in others:
        ys = slice(max(0, cy - oy), max(0, cy + ch - oy))
        xs = slice(max(0, cx - ox), max(0, cx + cw - ox))
        if letters is None:
            m[ys, xs] = False
            continue
        hit = np.zeros(shape, bool)
        hit[ys, xs] = letters[ys, xs] > 0
        if not hit.any():
            continue                    # no sibling lettering in the overlap
        ry, rx = np.nonzero(hit)
        m[max(0, ry.min() - pad):ry.max() + 1 + pad,
          max(0, rx.min() - pad):rx.max() + 1 + pad] = False
    return m


def seam_split(union, seeds):
    """Split one joined region between two lobes along the seam equidistant
    from their own lettering. Returns [mask_a, mask_b] (uint8, same shape).

    The straight row/column cut between the two text blocks assumes the
    lobes are stacked or side by side. When their blocks overlap on BOTH
    axes the lobes are DIAGONAL and no straight line separates them: on p294
    b07's block reaches down and right into b12's first line, so the row
    window ran from b07's block bottom (1971) UP to b12's block top (1938)
    and the cut fell at 1954 — straight through b07's own last line. The
    mask lost that line, which the cleaner is confined to the mask and so
    could not reach, and kept the slice of b12's balloon the flood had
    filled, so the last new line was set 274px wide in a 220px balloon.

    Equidistant assignment gives each lobe the part of the shape nearer its
    own type, which is how the artist drew the pair, and can never cross
    either lobe's letters."""
    reg = (union > 0)
    d = []
    for s in seeds:
        m = reg & (np.asarray(s) > 0)
        if not m.any():                 # no letters of its own to go on
            d.append(np.full(reg.shape, np.inf, np.float32))
            continue
        d.append(cv2.distanceTransform((~m).astype(np.uint8),
                                       cv2.DIST_L2, 3).astype(np.float32))
    if np.isinf(d[0]).all() or np.isinf(d[1]).all():
        return [reg.astype(np.uint8), np.zeros(reg.shape, np.uint8)] \
            if np.isinf(d[1]).all() else \
            [np.zeros(reg.shape, np.uint8), reg.astype(np.uint8)]
    first = d[0] <= d[1]
    return [(reg & first).astype(np.uint8), (reg & ~first).astype(np.uint8)]


def strict_bubble(img, letters, bx, by, bw, bh, gutter=None, luma=False,
                  force_step=False):
    """Segment a bubble whose interior merges with a light background: use
    the bubble's own dark OUTLINE as the boundary — close small gaps in it,
    then take the free-space component around the text block. A caption box
    with no dark outline (white trapezoid on pale fog) has no such boundary
    and the walk fills its hard bounds; that spill signature triggers ONE
    retry with the barrier set to the brightness step between the box's own
    interior and the duller ground. `gutter` (bool per page row: pure-white
    page-wide bands between panels) also acts as a barrier — a balloon cut
    off by the panel edge opens white-into-white onto the gutter, which no
    outline or brightness step can stop. Returns (x, y, interior mask) or
    None when the region still leaks (touches the window edge or grows far
    beyond the text block)."""
    h, w = img.shape[:2]
    win = 300
    wx0, wy0 = max(0, bx - win), max(0, by - win)
    wx1, wy1 = min(w, bx + bw + win), min(h, by + bh + win)
    local = img[wy0:wy1, wx0:wx1]
    lb = letters[wy0:wy1, wx0:wx1]
    ly0, ly1 = by - wy0, by - wy0 + bh
    lx0, lx1 = bx - wx0, bx - wx0 + bw
    # hard bounds: a bubble never extends far beyond its own text block,
    # so anything past these limits is background leak by definition
    hy0 = max(0, ly0 - int(round(0.55 * bh)))
    hy1 = min(local.shape[0], ly1 + int(round(0.55 * bh)))
    hx0 = max(0, lx0 - int(round(0.6 * bw)))
    hx1 = min(local.shape[1], lx1 + int(round(0.6 * bw)))

    # saturated stripe fills (deep cyan holograms) have a LOW min channel
    # and read as barrier, shredding the interior into per-stripe bands —
    # the luma retry keys the barrier on brightness instead, so only true
    # ink (dark in every channel) blocks the flood
    lkey = local.mean(axis=2) if luma else local.min(axis=2)

    def segment(barrier_max, use_gutter=True):
        barrier = (lkey <= barrier_max).astype(np.uint8)
        if use_gutter and gutter is not None:
            barrier[gutter[wy0:wy1], :] = 1
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8))
        free = (1 - barrier).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(free, 4)
        ring = labels[max(0, ly0 - 4):ly1 + 4, max(0, lx0 - 4):lx1 + 4]
        ringl = lb[max(0, ly0 - 4):ly1 + 4, max(0, lx0 - 4):lx1 + 4]
        cand = ring[(ring > 0) & (ringl == 0)]
        if not len(cand):
            return None
        counts = np.bincount(cand)
        lab = int(counts.argmax())
        comp = (labels == lab).astype(np.uint8)
        # a panel border running through the block (a balloon poking above
        # the panel into the gutter) splits the interior
        # into stacked free components — union every component with a real
        # share of the ring; the border rows are bridged after the fill so
        # the row walk can cross them
        share = counts / counts.sum()
        extra = [l for l in range(1, len(counts))
                 if l != lab and share[l] >= 0.10]
        for l in extra:
            comp = np.maximum(comp, (labels == l).astype(np.uint8))

        # bound FIRST (bounding before hole-filling also stops gutter leaks
        # from enclosing whole panels, which the fill would then swallow)
        bounded = np.zeros_like(comp)
        bounded[hy0:hy1, hx0:hx1] = comp[hy0:hy1, hx0:hx1]
        comp = bounded

        # fill the text holes so the width profile below is the smooth
        # bubble shape, not the jagged free-space-between-letters (padded:
        # see the normal-path comment about erode border sealing)
        comp = np.pad(comp, 6)
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
        fh, fw = comp.shape
        flood = comp.copy()
        cv2.floodFill(flood, np.zeros((fh + 2, fw + 2), np.uint8), (0, 0), 1)
        comp = (comp | (1 - flood))[6:-6, 6:-6]

        if extra:
            # bridge straight through the dark border rows that separate the
            # unioned parts, so follow_lobe's row walk can cross them
            bar_row = barrier[:, hx0:hx1].mean(axis=1) >= 0.8
            yy = 0
            while yy < comp.shape[0]:
                if not bar_row[yy]:
                    yy += 1
                    continue
                r0 = yy
                while yy < comp.shape[0] and bar_row[yy]:
                    yy += 1
                r1 = yy - 1
                if (r1 - r0 <= 14 and r0 > 0 and r1 < comp.shape[0] - 1
                        and comp[r0 - 1].any() and comp[r1 + 1].any()):
                    # union, not intersection: an empty bridge row would
                    # veto every text line that crosses the border
                    span = comp[r0 - 1] | comp[r1 + 1]
                    for rr in range(r0, r1 + 1):
                        comp[rr] = span

        return follow_lobe(comp, by - wy0 + bh // 2,
                           (bx - wx0, bx - wx0 + bw), bw)

    def fills_bounds(c):
        ys, xs = np.nonzero(c)
        return (ys.max() - ys.min() + 1 >= (hy1 - hy0) - 2
                or xs.max() - xs.min() + 1 >= (hx1 - hx0) - 2)

    # base pass WITHOUT the gutter barrier: an outlined balloon may
    # legitimately straddle a panel gutter or overhang the page margin, and
    # cutting it there orphans its first/last text line. The gutter barrier
    # only comes in as a spill retry (a bubble with no outline toward the
    # gutter fills its bounds without it).
    comp = segment(150, use_gutter=False)
    if comp is not None and comp.sum() and (fills_bounds(comp) or force_step):
        g = segment(150)
        if (g is not None and g.sum() >= 1.2 * bw * bh
                and g.sum() <= 0.75 * comp.sum()):
            comp = g
    if comp is not None and comp.sum() and (fills_bounds(comp) or force_step):
        # spill signature: no dark outline stopped the walk. Measure the
        # box's own interior brightness between the letters and use the
        # step down to the ground as the barrier (never hardcoded — fog
        # levels vary page to page).
        vals = lkey[ly0:ly1, lx0:lx1]
        vals = vals[(lb[ly0:ly1, lx0:lx1] == 0) & (vals > 150)]
        if len(vals):
            step = int(np.median(vals)) - 25
            if step > LIGHT_MIN:
                retry = segment(step)
                # accept only if the step barrier found real structure:
                # the mask shrank well below the leaked one (a box can
                # legitimately fill the bounds in ONE dimension, so a
                # bounds test cannot judge the retry)
                if (retry is not None and retry.sum() >= 1.2 * bw * bh
                        and retry.sum() <= 0.75 * comp.sum()):
                    comp = retry
    if comp is None:
        return None
    if comp.sum() < 1.2 * bw * bh:
        # a text-dense balloon's interior barely exceeds its own block area,
        # so the pure area factor rejects honest masks (text-dense balloons,
        # figure-8s whose block spans both lobes) — accept a smaller mask
        # when it still covers the text block's rows
        cov = sum(1 for yy in range(ly0, ly1)
                  if 0 <= yy < comp.shape[0]
                  and comp[yy, max(0, lx0):lx1].sum() >= 0.5 * bw)
        if comp.sum() < 0.9 * bw * bh or cov < 0.85 * bh:
            return None
    # a balloon clipped by a panel border can flood through an outline gap
    # onto the strip beyond it, and the hole-fill then seals the border line
    # INTO the mask — the drawn border runs through the mask as a mostly-dark
    # full-width row. Cut the mask at any such row outside the text block,
    # keeping the side that holds the block (text rows inside the block never
    # qualify: their between-letter fill keeps the dark fraction low).
    dmin = local.min(axis=2)
    cuts = []
    for yy in range(comp.shape[0]):
        xs_ = np.flatnonzero(comp[yy])
        if len(xs_) >= max(20, 0.5 * bw) \
                and (dmin[yy, xs_] <= DARK_MAX).mean() >= 0.6:
            cuts.append(yy)
    top_cut = max((yy for yy in cuts if yy < ly0), default=None)
    bot_cut = min((yy for yy in cuts if yy > ly1), default=None)
    if top_cut is not None:
        comp[:top_cut + 1] = 0
    if bot_cut is not None:
        comp[bot_cut:] = 0
    # NECK cut: a flood escaping through a narrow break (an outline gap, a
    # clipped corner at a panel edge) re-expands beyond it — the pinch reads
    # as a neck in the width profile outside the text block. Cut at the
    # narrowest such neck, keeping the block's side; a balloon's own arc tip
    # never qualifies (nothing re-expands beyond it).
    widths = comp.sum(axis=1).astype(int)
    thr = max(14, int(0.15 * bw))
    necks = [yy for yy in range(0, max(0, min(ly0, comp.shape[0])))
             if 0 < widths[yy] <= thr
             and widths[:yy].max(initial=0) >= 3 * widths[yy]]
    if necks:
        comp[:min(necks, key=lambda r: widths[r]) + 1] = 0
    necks = [yy for yy in range(max(0, min(ly1, comp.shape[0])),
                                comp.shape[0])
             if 0 < widths[yy] <= thr
             and widths[yy + 1:].max(initial=0) >= 3 * widths[yy]]
    if necks:
        comp[min(necks, key=lambda r: widths[r]):] = 0
    if not comp.any():
        return None
    ys, xs = np.nonzero(comp)
    cy, cx = int(ys.min()), int(xs.min())
    interior = comp[cy:ys.max() + 1, cx:xs.max() + 1]
    # the barrier closing ate ~3px of the rim; take a bit more off so the
    # row-median clean never repaints over the outline's anti-aliasing
    interior = cv2.erode(interior, np.ones((3, 3), np.uint8))
    return wx0 + cx, wy0 + cy, interior


def detect_page(path: Path):
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    light = (img.min(axis=2) >= LIGHT_MIN).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(light, 8)
    letters = letter_mask(img)
    # pure-white page-wide rows = the gutter between stacked panels (tinted
    # panel fills like pale sky stay below the purity threshold). The
    # outermost rows are scan-edge artifacts, not gutters — flagging them
    # cuts the bottom off full-bleed captions that touch the page edge
    gutter = ((img.min(axis=2) >= 235).mean(axis=1) > 0.97)
    gutter[:4] = gutter[-4:] = False

    def filled_interior(lab):
        """Hole-filled interior of one light component (padded before the
        close/fill: on the tight crop, OpenCV's erode treats out-of-image as
        foreground, sealing bridges along the bbox edges — the corner voids
        then read as holes and fill solid (rectangles!))."""
        cx, cy, cw, ch = (int(v) for v in stats[lab][:4])
        comp = np.pad((labels[cy:cy + ch, cx:cx + cw] == lab).astype(np.uint8), 6)
        closed = cv2.morphologyEx(comp, cv2.MORPH_CLOSE,
                                  np.ones((9, 9), np.uint8))
        flood = closed.copy()
        cv2.floodFill(flood, np.zeros((closed.shape[0] + 2,
                                       closed.shape[1] + 2), np.uint8),
                      (0, 0), 1)
        return (closed | (1 - flood))[6:-6, 6:-6]

    entries = {}
    for (bx, by, bw, bh) in paragraph_blocks(letters):
        # bubble = the light component most present just around the block
        y0, y1 = max(0, by - 6), min(h, by + bh + 6)
        x0, x1 = max(0, bx - 6), min(w, bx + bw + 6)
        local = labels[y0:y1, x0:x1]
        cand = local[(local > 0) & (letters[y0:y1, x0:x1] == 0)]
        if not len(cand):
            continue
        lab = np.bincount(cand).argmax()
        cx, cy, cw, ch = stats[lab][:4]
        edge = cx == 0 or cy == 0 or cx + cw >= w or cy + ch >= h
        big = stats[lab][4] > 12 * bw * bh
        in_page_margin = (by + bh / 2 < 0.035 * h
                          or by + bh / 2 > 0.965 * h)
        if in_page_margin:
            # a white-on-white margin NOTE has no ink anywhere around its
            # text — but a real balloon whose center falls in the margin
            # band (drawn at the panel's very edge) has an outline, and
            # must go through the strict segmentation below
            ry0, ry1 = max(0, by - 30), min(h, by + bh + 30)
            rx0, rx1 = max(0, bx - 30), min(w, bx + bw + 30)
            ring = img[ry0:ry1, rx0:rx1].min(axis=2) <= 150
            ring = ring & ~(letters[ry0:ry1, rx0:rx1] > 0)
            in_page_margin = ring.mean() < 0.02
        if edge or big:
            # the light component leaked into the background (smoke, page
            # margin). Page-margin notes keep the simple rectangle handling
            # (white on white, invisible); panel bubbles get re-segmented
            # using their own dark outline as the boundary.
            strict = (None if in_page_margin
                      else strict_bubble(img, letters, bx, by, bw, bh,
                                         gutter))
            if strict is not None:
                sx, sy, interior = strict
                entries[("strict", bx // 50, by // 50)] = {
                    "kind": "bubble", "strict": True,
                    "block": [bx, by, bw, bh],
                    "bbox": [int(sx), int(sy),
                             int(interior.shape[1]), int(interior.shape[0])],
                    "mask": interior, "paragraphs": 1}
                continue
            # fall back: rectangle around the text,
            # grown while the surroundings stay light (inside the bubble)
            pad = 10
            if not edge or (0 < bx and bx + bw < w):
                gx0, gy0, gx1, gy1 = bx, by, bx + bw, by + bh
                lm = light > 0
                for _ in range(60):
                    grew = False
                    if gx0 > 4 and lm[gy0:gy1, gx0 - 4:gx0].mean() > 0.97:
                        gx0 -= 4; grew = True
                    if gx1 < w - 4 and lm[gy0:gy1, gx1:gx1 + 4].mean() > 0.97:
                        gx1 += 4; grew = True
                    if gy0 > 4 and lm[gy0 - 4:gy0, gx0:gx1].mean() > 0.97:
                        gy0 -= 4; grew = True
                    if gy1 < h - 4 and lm[gy1:gy1 + 4, gx0:gx1].mean() > 0.97:
                        gy1 += 4; grew = True
                    if not grew:
                        break
                entries[("rect", bx // 50, by // 50)] = {
                    "kind": "open", "block": [bx, by, bw, bh],
                    "bbox": [int(gx0), int(gy0),
                             int(gx1 - gx0), int(gy1 - gy0)],
                    "mask": None}
                continue
            key = ("rect", bx // 50, by // 50)
            entries[key] = {
                "kind": "margin", "block": [bx, by, bw, bh],
                "bbox": [max(0, bx - pad), max(0, by - pad),
                         min(w, bx + bw + pad) - max(0, bx - pad),
                         min(h, by + bh + pad) - max(0, by - pad)],
                "mask": None}
            continue
        key = ("comp", int(lab))
        if key in entries:
            # a second text block in the same light component = joined
            # balloons sharing one outline. NEVER one entry: a single row
            # profile cannot center text in two lobes, so this block gets
            # its OWN entry (own mask, own transcript string). Overlapping
            # walks are waist-split and font-grouped by the overlap-merge
            # stage below, exactly like lobes that arrive as separate
            # components; side-by-side lobes are split here at the vertical
            # midline between the blocks (the stage only knows the
            # horizontal waist split).
            host = entries[key]
            walked = follow_lobe(filled_interior(lab), by - cy + bh // 2,
                                 (bx - cx, bx - cx + bw), bw)
            if walked is None or walked.sum() < 1.2 * bw * bh:
                # can't segment a lobe of its own: fold into the host as an
                # extra paragraph (the fit's lobe-banding handles centering)
                hb = list(host["block"])          # before the merge
                host["block"] = merge_boxes(host["block"], [bx, by, bw, bh])
                host["paragraphs"] += 1
                # The host's mask must cover BOTH lobes, or the folded
                # paragraph has nowhere to go: p37 kept a 603x51 strip
                # spanning two balloons, so neither was ever cleaned or
                # typeset. Recover each block's own drawn balloon and union
                # them — the fit's lobe decomposition then hands each
                # paragraph its own lobe, and the bubble COUNT is unchanged
                # so the transcripts stay valid.
                pieces = []
                for (qx, qy, qw, qh) in (hb, [bx, by, bw, bh]):
                    sp = strict_bubble(img, letters, qx, qy, qw, qh, gutter)
                    if sp is not None:
                        pieces.append(sp)
                if len(pieces) == 2:
                    ux0 = min(p2[0] for p2 in pieces)
                    uy0 = min(p2[1] for p2 in pieces)
                    ux1 = max(p2[0] + p2[2].shape[1] for p2 in pieces)
                    uy1 = max(p2[1] + p2[2].shape[0] for p2 in pieces)
                    um = np.zeros((uy1 - uy0, ux1 - ux0), np.uint8)
                    for (px, py, pm) in pieces:
                        sl = um[py - uy0:py - uy0 + pm.shape[0],
                                px - ux0:px - ux0 + pm.shape[1]]
                        np.maximum(sl, pm, out=sl)
                    host["mask"] = um
                    host["bbox"] = [int(ux0), int(uy0),
                                    int(ux1 - ux0), int(uy1 - uy0)]
                    host["strict"] = True
                continue
            new = {"kind": "bubble", "strict": True,
                   "block": [int(bx), int(by), int(bw), int(bh)],
                   "bbox": [int(cx), int(cy), int(cw), int(ch)],
                   "mask": walked, "paragraphs": 1}
            hx0 = host["block"][0]
            hx1 = host["block"][0] + host["block"][2]
            xov = min(bx + bw, hx1) - max(bx, hx0)
            if xov < 0.25 * min(bw, hx1 - hx0):  # side-by-side lobes
                mini_left = bx + bw / 2.0 < (hx0 + hx1) / 2.0
                xsplit = (((bx + bw + hx0) // 2 if mini_left
                           else (hx1 + bx) // 2) - cx)
                hostm = np.maximum(host["mask"], walked)
                lobe = walked.copy()
                if mini_left:
                    lobe[:, xsplit:] = 0
                    hostm[np.any(lobe, axis=1), :xsplit] = 0
                else:
                    lobe[:, :xsplit] = 0
                    hostm[np.any(lobe, axis=1), xsplit:] = 0
                if lobe.any():
                    gid = -(len(entries) + 1)
                    host["mask"] = hostm
                    host["group"] = gid
                    new["mask"] = lobe
                    new["group"] = gid
            elif (walked & host["mask"]).any():
                # stacked: give BOTH entries the union of the walks — the
                # stage's waist split partitions it between them, so a poor
                # walk on either side (stopped by letters touching the
                # outline) is repaired by the other
                union = np.maximum(host["mask"], walked)
                host["mask"] = union
                new["mask"] = union.copy()
            entries[("comp2", int(lab), by // 50, bx // 50)] = new
            continue
        interior = filled_interior(lab)
        # same lobe-walk as the strict path: keeps the true arcs down to the
        # outline (or panel border) while clamping leaks into gutters
        walked = follow_lobe(interior, by - cy + bh // 2,
                             (bx - cx, bx - cx + bw), bw)
        if walked is not None and walked.sum() > 1.2 * bw * bh:
            interior = walked
        entries[key] = {"kind": "bubble", "strict": True,
                        "block": [bx, by, bw, bh],
                        "bbox": [int(cx), int(cy), int(cw), int(ch)],
                        "mask": interior, "paragraphs": 1}

    # short utterances (a repeated one-word shout) have too few letter blobs to pass
    # MIN_LETTERS, but a small letter cluster INSIDE an already-detected
    # bubble's light component, outside its text block, is certainly bubble
    # text: give it its own lobe. Stacked lobes (figure-8) take the union;
    # side-by-side lobes are split at the vertical midline between the two
    # text blocks, each text centering in its own lobe (a union there would
    # fake row intervals across the waist and center the short text between
    # the lobes instead of inside its own)
    claimed = np.zeros_like(letters)
    for e in entries.values():
        ebx, eby, ebw, ebh = e["block"]
        claimed[eby:eby + ebh, ebx:ebx + ebw] = 1
    spare = letters & (1 - claimed)
    # wider horizontal join than paragraph_blocks: ellipses between short
    # words (drawn-out interjections, ellipses) drop below the letter-blob
    # area floor, so the
    # words sit further apart than letters within a word
    joined = cv2.dilate(spare, np.ones((7, 45), np.uint8))
    joined = cv2.morphologyEx(joined, cv2.MORPH_CLOSE,
                              np.ones((21, 9), np.uint8))
    ncc, _, scc, _ = cv2.connectedComponentsWithStats(joined, 8)
    for i in range(1, ncc):
        bx, by, bw, bh = (int(v) for v in scc[i][:4])
        cnt = int(cv2.connectedComponentsWithStats(
            spare[by:by + bh, bx:bx + bw], 8)[0]) - 1
        if not (2 <= cnt < MIN_LETTERS and bw >= 30 and 12 <= bh <= 60):
            continue
        y0, y1 = max(0, by - 6), min(h, by + bh + 6)
        x0, x1 = max(0, bx - 6), min(w, bx + bw + 6)
        local = labels[y0:y1, x0:x1]
        cand = local[(local > 0) & (letters[y0:y1, x0:x1] == 0)]
        if not len(cand):
            continue
        lab = int(np.bincount(cand).argmax())
        host = entries.get(("comp", lab))
        if host is not None and host.get("mask") is not None:
            ox, oy = int(stats[lab][0]), int(stats[lab][1])
            ow, oh = int(stats[lab][2]), int(stats[lab][3])
            walked = follow_lobe(filled_interior(lab), by - oy + bh // 2,
                                 (bx - ox, bx - ox + bw), bw)
        else:
            host = None
            cx, cy, cw, ch = (int(v) for v in stats[lab][:4])
            edge = cx <= 0 or cy <= 0 or cx + cw >= w or cy + ch >= h
            area = int(stats[lab][4])
            if not edge and 2 * bw * bh <= area <= 40 * bw * bh:
                # STANDALONE short bubble (a one-word reply): its own small
                # closed white component around the cluster
                comp_px = img[cy:cy + ch, cx:cx + cw][
                    labels[cy:cy + ch, cx:cx + cw] == lab]
                if np.median(comp_px.min(axis=1)) >= 185:
                    walked = follow_lobe(filled_interior(lab),
                                         by - cy + bh // 2,
                                         (bx - cx, bx - cx + bw), bw)
                    if walked is not None and walked.sum() >= 1.2 * bw * bh:
                        entries[("mini", bx // 50, by // 50)] = {
                            "kind": "bubble", "strict": True,
                            "block": [bx, by, bw, bh],
                            "bbox": [cx, cy, cw, ch],
                            "mask": walked, "paragraphs": 1}
                continue
            # the cluster's own light component is unusable (page background
            # or gutter leak). A lobe of a strict-segmented balloon: find the
            # entry whose mask already covers the cluster and walk within it
            for e in list(entries.values()):
                m = e.get("mask")
                if m is None:
                    continue
                ex, ey, ew, eh = e["bbox"]
                if not (ex <= bx and bx + bw <= ex + ew
                        and ey <= by and by + bh <= ey + eh):
                    continue
                if m[by - ey:by - ey + bh, bx - ex:bx - ex + bw].mean() > 0.3:
                    host, ox, oy, ow, oh = e, ex, ey, ew, eh
                    break
            if host is None:
                # free-standing bubble whose interior leaks into the white
                # gutter (a short bubble cut by the panel edge): segment it
                # like a leaked block, gutter rows as barrier
                strict = strict_bubble(img, letters, bx, by, bw, bh, gutter)
                if strict is not None:
                    sx, sy, interior = strict
                    entries[("mini", bx // 50, by // 50)] = {
                        "kind": "bubble", "strict": True,
                        "block": [bx, by, bw, bh],
                        "bbox": [int(sx), int(sy), int(interior.shape[1]),
                                 int(interior.shape[0])],
                        "mask": interior, "paragraphs": 1}
                    continue
                # no outline at all (a whisper on plain white):
                # the same grown-rectangle handling as the main path's
                # open fallback
                gx0, gy0, gx1, gy1 = bx, by, bx + bw, by + bh
                lm = light > 0
                for _ in range(60):
                    grew = False
                    if gx0 > 4 and lm[gy0:gy1, gx0 - 4:gx0].mean() > 0.97:
                        gx0 -= 4; grew = True
                    if gx1 < w - 4 and lm[gy0:gy1, gx1:gx1 + 4].mean() > 0.97:
                        gx1 += 4; grew = True
                    if gy0 > 4 and lm[gy0 - 4:gy0, gx0:gx1].mean() > 0.97:
                        gy0 -= 4; grew = True
                    if gy1 < h - 4 and lm[gy1:gy1 + 4, gx0:gx1].mean() > 0.97:
                        gy1 += 4; grew = True
                    if not grew:
                        break
                entries[("mini", bx // 50, by // 50)] = {
                    "kind": "open", "block": [bx, by, bw, bh],
                    "bbox": [int(gx0), int(gy0),
                             int(gx1 - gx0), int(gy1 - gy0)],
                    "mask": None}
                continue
            walked = follow_lobe(host["mask"], by - oy + bh // 2,
                                 (bx - ox, bx - ox + bw), bw)
        if walked is None or walked.sum() < 1.2 * bw * bh:
            continue
        cx, cy = ox, oy
        hx0 = host["block"][0]
        hx1 = host["block"][0] + host["block"][2]
        # blocks carry ~12px of dilation margin, so side-by-side lobes can
        # still overlap a little in x — compare the overlap to the widths
        xov = min(bx + bw, hx1) - max(bx, hx0)
        mini_left = bx + bw / 2.0 < (hx0 + hx1) / 2.0
        if xov < 0.25 * min(bw, hx1 - hx0):  # side-by-side lobes
            xsplit = (((bx + bw + hx0) // 2 if mini_left
                       else (hx1 + bx) // 2) - cx)
            lobe = walked.copy()
            hostm = np.maximum(host["mask"], walked)
            if mini_left:
                lobe[:, xsplit:] = 0
                hostm[np.any(lobe, axis=1), :xsplit] = 0
            else:
                lobe[:, :xsplit] = 0
                hostm[np.any(lobe, axis=1), xsplit:] = 0
            if not lobe.any():
                continue
            gid = -(len(entries) + 1)  # negative: never collides with the
            # overlap-merge stage's group counter
            host["mask"] = hostm
            host["group"] = gid
            entries[("mini", bx // 50, by // 50)] = {
                "kind": "bubble", "strict": True, "group": gid,
                "block": [bx, by, bw, bh],
                "bbox": [cx, cy, int(ow), int(oh)],
                "mask": lobe, "paragraphs": 1}
        elif (walked & host["mask"]).any():
            # stacked lobes: own entry; the overlap-merge stage waist-splits
            # the masks and font-groups the pair
            entries[("mini", bx // 50, by // 50)] = {
                "kind": "bubble", "strict": True,
                "block": [bx, by, bw, bh],
                "bbox": [cx, cy, int(ow), int(oh)],
                "mask": walked, "paragraphs": 1}
        else:
            # disjoint lobes of one balloon (the waist pinched the walks
            # apart): own entry, font-grouped with the host like any
            # compound-balloon pair
            gid = host.get("group")
            if gid is None:
                gid = -(len(entries) + 1)
                host["group"] = gid
            entries[("mini", bx // 50, by // 50)] = {
                "kind": "bubble", "strict": True, "group": gid,
                "block": [bx, by, bw, bh],
                "bbox": [cx, cy, int(ow), int(oh)],
                "mask": walked, "paragraphs": 1}

    # SATURATED-STRIPE repair: deep-cyan hologram fills sit below LIGHT_MIN
    # and below the strict barrier, so masks shred into per-stripe bands (or
    # the entry falls back to an open rect whose letter-inpaint cannot clean
    # a striped ground). For striped entries whose mask misses part of the
    # text block, re-segment with the LUMA barrier: saturated stripes stay
    # interior, true ink stays barrier.
    for key, e in list(entries.items()):
        if e["kind"] not in ("bubble", "open"):
            continue
        bx, by, bw, bh = e["block"]
        ex, ey, ew, eh = e["bbox"]
        m = e.get("mask")
        if m is not None:
            covered = sum(
                1 for yy in range(by, by + bh)
                if 0 <= yy - ey < m.shape[0]
                and m[yy - ey, max(0, bx - ex):bx - ex + bw].sum() >= 0.5 * bw)
            if covered >= 0.85 * bh:
                continue
        region = img[ey:ey + eh, ex:ex + ew]
        ink = region.min(axis=2) <= DARK_MAX
        fill = (m > 0) & ~ink if m is not None else ~ink
        tinted = fill & (region.max(axis=2).astype(int)
                         - region.min(axis=2).astype(int) > 18)
        if not fill.any():
            continue
        blkreg = region[max(0, by - ey):by - ey + bh,
                        max(0, bx - ex):bx - ex + bw]
        bink2 = blkreg.min(axis=2) <= DARK_MAX
        btint = (~bink2) & (blkreg.max(axis=2).astype(int)
                            - blkreg.min(axis=2).astype(int) > 18)
        if btint.sum() > 0.12 * max(1, (~bink2).sum()):
            res = strict_bubble(img, letters, bx, by, bw, bh, gutter,
                                luma=True)
        else:
            # not striped: fire only when the fill is BIMODAL — a chunk of
            # it clearly darker than the interior around the text block (the
            # mask leaked onto a pale wall/floor; cleaning would white it
            # out). Re-segment with the brightness-step retry forced.
            if m is None:
                continue
            fv = region.min(axis=2)[(m > 0) & ~ink]
            blk = region[max(0, by - ey):by - ey + bh,
                         max(0, bx - ex):bx - ex + bw]
            bink = blk.min(axis=2) <= DARK_MAX
            bv = blk.min(axis=2)[~bink]
            if not len(fv) or not len(bv):
                continue
            if (fv < np.median(bv) - 25).mean() <= 0.07:
                continue
            res = strict_bubble(img, letters, bx, by, bw, bh, gutter,
                                force_step=True)
        if res is None:
            continue
        sx, sy, interior = res
        rcov = sum(1 for yy in range(by, by + bh)
                   if 0 <= yy - sy < interior.shape[0]
                   and interior[yy - sy].any())
        if rcov < 0.85 * bh:
            continue
        e.update({"kind": "bubble", "strict": True, "mask": interior,
                  "bbox": [int(sx), int(sy), int(interior.shape[1]),
                           int(interior.shape[0])]})

    # white-on-black caption boxes: grow the block while surroundings stay
    # dark (bounded), require the region to really be a solid dark box
    wl = white_letter_mask(img)
    dark = (img.max(axis=2) <= 90)
    for (bx, by, bw, bh) in paragraph_blocks(wl):
        gx0, gy0, gx1, gy1 = bx, by, bx + bw, by + bh
        for _ in range(8):  # max ~32px growth per side
            grew = False
            if gx0 > 4 and dark[gy0:gy1, gx0 - 4:gx0].mean() > 0.9:
                gx0 -= 4; grew = True
            if gx1 < w - 4 and dark[gy0:gy1, gx1:gx1 + 4].mean() > 0.9:
                gx1 += 4; grew = True
            if gy0 > 4 and dark[gy0 - 4:gy0, gx0:gx1].mean() > 0.9:
                gy0 -= 4; grew = True
            if gy1 < h - 4 and dark[gy1:gy1 + 4, gx0:gx1].mean() > 0.9:
                gy1 += 4; grew = True
            if not grew:
                break
        box = img[gy0:gy1, gx0:gx1]
        boxdark = (box.max(axis=2) <= 90) | (wl[gy0:gy1, gx0:gx1] > 0)
        if boxdark.mean() < 0.85:
            continue  # white SFX on art, not a caption box
        entries[("dark", bx // 50, by // 50)] = {
            "kind": "dark", "block": [bx, by, bw, bh],
            "bbox": [int(gx0), int(gy0),
                     int(gx1 - gx0), int(gy1 - gy0)],
            "mask": None}

    # TINTED caption boxes (dark navy text on a blue rectangle): the
    # letters are dark in EVERY channel while the fill is
    # bright in its strongest channel yet below LIGHT_MIN in its weakest,
    # so both the light-ground letter pass and the white-on-black pass are
    # blind to them. The box is grown like a dark caption box; cleaning is
    # reletter_fit's 'tint' branch (row-median of the tinted fill) and the
    # new text stays black, like the original.
    tink = ((img.max(axis=2) <= 85) & ~(letters > 0)).astype(np.uint8)
    tintbg = (img.max(axis=2) >= 125) & (img.min(axis=2) < LIGHT_MIN)
    ntc, tclabels, tcstats, _ = cv2.connectedComponentsWithStats(tink, 8)
    tckeep = np.zeros(ntc, bool)
    for i in range(1, ntc):
        x, y, w2, h2, area = tcstats[i]
        if not (LETTER_H[0] <= h2 <= LETTER_H[1] and w2 <= LETTER_W_MAX
                and 8 <= area <= LETTER_AREA_MAX):
            continue
        y0, y1 = max(0, y - 3), min(h, y + h2 + 3)
        x0, x1 = max(0, x - 3), min(w, x + w2 + 3)
        ring = tintbg[y0:y1, x0:x1] & (tclabels[y0:y1, x0:x1] != i)
        if ring.mean() < 0.5:
            continue
        tckeep[i] = True
    tletters = tckeep[tclabels].astype(np.uint8)
    tb = tintbg | (tletters > 0)
    for (bx, by, bw, bh) in paragraph_blocks(tletters):
        if any(e["bbox"][0] <= bx and bx + bw <= e["bbox"][0] + e["bbox"][2]
               and e["bbox"][1] <= by and by + bh <= e["bbox"][1] + e["bbox"][3]
               for e in entries.values()):
            continue
        gx0, gy0, gx1, gy1 = bx, by, bx + bw, by + bh
        for _ in range(8):
            grew = False
            if gx0 > 4 and tb[gy0:gy1, gx0 - 4:gx0].mean() > 0.9:
                gx0 -= 4; grew = True
            if gx1 < w - 4 and tb[gy0:gy1, gx1:gx1 + 4].mean() > 0.9:
                gx1 += 4; grew = True
            if gy0 > 4 and tb[gy0 - 4:gy0, gx0:gx1].mean() > 0.9:
                gy0 -= 4; grew = True
            if gy1 < h - 4 and tb[gy1:gy1 + 4, gx0:gx1].mean() > 0.9:
                gy1 += 4; grew = True
            if not grew:
                break
        if tb[gy0:gy1, gx0:gx1].mean() < 0.85:
            continue  # dark blobs on tinted art, not a solid caption box
        entries[("tint", bx // 50, by // 50)] = {
            "kind": "tint", "block": [int(bx), int(by), int(bw), int(bh)],
            "bbox": [int(gx0), int(gy0),
                     int(gx1 - gx0), int(gy1 - gy0)],
            "mask": None}

    # merge entries whose masks overlap: the lobes of one compound balloon
    # (each keeps its own text block; the merged bubble holds N paragraphs)
    def bbox_overlap(a, b):
        ax, ay, aw, ah = a["bbox"]
        bx2, by2, bw2, bh2 = b["bbox"]
        ix = max(0, min(ax + aw, bx2 + bw2) - max(ax, bx2))
        iy = max(0, min(ay + ah, by2 + bh2) - max(ay, by2))
        inter = ix * iy
        return inter / min(aw * ah, bw2 * bh2)

    def stacked(a, b):
        """Lobes of one compound balloon: horizontally aligned, vertically
        touching or nearly so."""
        ax, ay, aw, ah = a["bbox"]
        bx2, by2, bw2, bh2 = b["bbox"]
        ix = max(0, min(ax + aw, bx2 + bw2) - max(ax, bx2))
        vgap = max(ay, by2) - min(ay + ah, by2 + bh2)
        return ix > 0.5 * min(aw, bw2) and vgap < 26

    def masks_overlap(a, b):
        """True when the two entries' masks share actual pixels (bbox
        overlap is not enough: a huge light-background entry engulfs every
        balloon bbox in its panel and must never trigger a waist split)."""
        ax, ay = a["bbox"][:2]
        bx2, by2 = b["bbox"][:2]
        x0, y0 = max(ax, bx2), max(ay, by2)
        x1 = min(ax + a["bbox"][2], bx2 + b["bbox"][2])
        y1 = min(ay + a["bbox"][3], by2 + b["bbox"][3])
        if x1 <= x0 or y1 <= y0:
            return False
        am = a["mask"][y0 - ay:y1 - ay, x0 - ax:x1 - ax]
        bm = b["mask"][y0 - by2:y1 - by2, x0 - bx2:x1 - bx2]
        return bool((am & bm).any())

    def _cut(e, at, axis, keep_before):
        """Zero one side of e's mask at page coordinate `at`.

        Skipped when it would leave the lobe with almost nothing: the clamp
        that keeps the cut clear of the upper/left lobe's own lettering can
        push it right past the other lobe's whole mask, and a lobe with an
        empty mask breaks everything downstream."""
        off = e["bbox"][axis]
        i = max(0, min(e["mask"].shape[axis], at - off))
        sl = [slice(None), slice(None)]
        sl[axis] = slice(i, None) if keep_before else slice(0, i)
        trial = e["mask"].copy()
        trial[tuple(sl)] = 0
        if trial.sum() >= 0.25 * e["mask"].sum():
            e["mask"] = trial

    def crop_to_mask(e):
        m = e["mask"]
        ys, xs = np.nonzero(m)
        if not len(ys):
            return
        x, y = e["bbox"][:2]
        e["mask"] = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        e["bbox"] = [x + int(xs.min()), y + int(ys.min()),
                     int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]

    # stacked compound balloons: two text blocks whose masks overlap share
    # one outline. Split the shared region at the waist midline so each
    # paragraph keeps its own lobe, and mark them as one group so the
    # typesetter gives both the same font size.
    merged = []
    ngroups = 0
    for e in sorted(entries.values(), key=lambda e: e["block"][1]):
        cands = [m for m in merged
                 if m["kind"] == "bubble" and e["kind"] == "bubble"
                 and m["mask"] is not None and e["mask"] is not None
                 # lobes pre-grouped by the short-utterance pass are
                 # already split along the right (vertical) axis — the
                 # horizontal waist split here would wreck them
                 and not (e.get("group") is not None
                          and m.get("group") == e.get("group"))
                 and (bbox_overlap(m, e) > 0.35 or stacked(m, e))]
        # a huge light-background entry engulfs every balloon bbox in its
        # panel: prefer the candidate whose MASK truly overlaps (the real
        # joined lobe) so it is the one that gets waist-split below
        host = next((m for m in cands if masks_overlap(m, e)),
                    cands[0] if cands else None)
        merged.append(e)
        if host is None:
            continue
        ngroups += 1
        gid = ngroups
        host["group"] = e["group"] = gid
        if not masks_overlap(host, e):
            continue  # separate masks already; shared size is all they need
        hbx, hby, hbw, hbh = host["block"]
        ebx, eby, ebw, ebh = e["block"]
        xov = min(hbx + hbw, ebx + ebw) - max(hbx, ebx)
        if xov < 0.25 * min(hbw, ebw):
            # SIDE-BY-SIDE lobes: split on the vertical midline between the
            # blocks — the horizontal waist split below would hand one lobe
            # the top band and the other the bottom (text pinned off-center)
            left, right = ((host, e) if hbx <= ebx else (e, host))
            midx = (left["block"][0] + left["block"][2]
                    + right["block"][0]) // 2
            # never inside a lobe's OWN text block: cutting there leaves that
            # lettering outside every mask, so it is never cleaned and the
            # new text lands on top of it (the rule cede() states below). A
            # no-op unless the blocks overlap, which inverts the midline.
            midx = max(midx, left["block"][0] + left["block"][2])
            _cut(left, midx, 1, True)
            _cut(right, midx, 1, False)
        else:
            upper, lower = ((host, e) if hby <= eby else (e, host))
            mid = (upper["block"][1] + upper["block"][3]
                   + lower["block"][1]) // 2
            mid = max(mid, upper["block"][1] + upper["block"][3])
            _cut(upper, mid, 0, True)
            _cut(lower, mid, 0, False)
        if host["mask"].any():
            crop_to_mask(host)
        if e["mask"].any():
            crop_to_mask(e)

    # REFINE grouped and multi-paragraph bubble masks: the union + waist /
    # midline splits above carve flat tops and hand one lobe pieces of its
    # sibling, so the text lands off the visible lobe center (user-reported
    # across comic 7's joined balloons). A per-block outline-bounded flood
    # (strict_bubble) recovers each lobe's honest drawn shape — accepted
    # only when it still covers the text block's rows.
    for e in merged:
        if e["kind"] != "bubble" or e.get("mask") is None:
            continue
        if e.get("group") is None and e.get("paragraphs", 1) == 1:
            continue
        ebx, eby, ebw, ebh = e["block"]
        s = strict_bubble(img, letters, ebx, eby, ebw, ebh, gutter)
        if s is None:
            continue
        sx, sy, m = s
        cov = sum(1 for yy in range(eby, eby + ebh)
                  if 0 <= yy - sy < m.shape[0]
                  and m[yy - sy, max(0, ebx - sx):ebx - sx + ebw].sum()
                  >= 0.5 * ebw)
        if cov < 0.85 * ebh:
            continue
        # ...and the 15% it may miss must not be a TRUNCATION at an end of
        # the block. p149 b08's flood stopped 12px above its block's bottom
        # — 89% covered, so it was accepted — and the mask lost the last
        # line, "SU PODER.", which the cleaner (confined to the mask) then
        # could not reach and the reader saw standing under the new text.
        # Give those rows back from the mask this one replaces, bounded to
        # the block: past the block the old mask has nothing to vouch for it.
        px, py = e["bbox"][:2]
        prev, pm = e["mask"], e["mask"].shape
        ux0, uy0 = min(sx, px), min(sy, py)
        ux1 = max(sx + m.shape[1], px + pm[1])
        uy1 = max(sy + m.shape[0], py + pm[0])
        keep = np.zeros((uy1 - uy0, ux1 - ux0), np.uint8)
        keep[sy - uy0:sy - uy0 + m.shape[0],
             sx - ux0:sx - ux0 + m.shape[1]] = m
        blk = own_block_mask(keep.shape, (ux0, uy0), e["block"],
                             [q["block"] for q in merged if q is not e
                              and q["kind"] == "bubble"],
                             letters=letters[uy0:uy1, ux0:ux1])
        # ...and ONLY the truncated end: block rows the new mask does not
        # reach at all, kept inside its own columns. Repairing every missing
        # pixel inside the block lets the old mask WIDEN the new one, and a
        # block runs a little past its own balloon — p294 b06's mask grew 9px
        # left into b04's balloon, which moved the notch cut between them and
        # cost b04 five size steps on a one-word line.
        band = np.zeros(keep.shape, bool)
        band[:sy - uy0] = True
        band[sy - uy0 + m.shape[0]:] = True
        band[:, :sx - ux0] = False
        band[:, sx - ux0 + m.shape[1]:] = False
        miss = blk & band & (keep == 0)
        if miss.any():
            old = np.zeros(keep.shape, np.uint8)
            old[py - uy0:py - uy0 + pm[0], px - ux0:px - ux0 + pm[1]] = prev
            keep[miss & (old > 0)] = 1
        ys_, xs_ = np.nonzero(keep)
        if not len(ys_):
            continue
        e["mask"] = keep[ys_.min():ys_.max() + 1, xs_.min():xs_.max() + 1]
        e["bbox"] = [int(ux0 + xs_.min()), int(uy0 + ys_.min()),
                     int(xs_.max() - xs_.min() + 1),
                     int(ys_.max() - ys_.min() + 1)]
        e["strict"] = True

    # where two refreshed lobes of a group overlap (one flood wrapped around
    # or through the other's balloon), assign each contested pixel to the
    # lobe whose EXCLUSIVE region is nearer — overlapping ovals split along
    # their natural equidistant seam, and a wrap-around gives the wrapped
    # balloon its own body back (a plain block-midline split hands one lobe
    # a third of its sibling's oval)
    bygroup = {}
    for e in merged:
        if (e.get("group") is not None and e["kind"] == "bubble"
                and e.get("mask") is not None):
            bygroup.setdefault(e["group"], []).append(e)
    for g, es in bygroup.items():
        if len(es) != 2:
            continue
        a, b2 = es
        ux0 = min(a["bbox"][0], b2["bbox"][0])
        uy0 = min(a["bbox"][1], b2["bbox"][1])
        ux1 = max(a["bbox"][0] + a["bbox"][2], b2["bbox"][0] + b2["bbox"][2])
        uy1 = max(a["bbox"][1] + a["bbox"][3], b2["bbox"][1] + b2["bbox"][3])
        can = np.zeros((2, uy1 - uy0, ux1 - ux0), np.uint8)
        for k, e in enumerate((a, b2)):
            x, y = e["bbox"][:2]
            mh, mw = e["mask"].shape
            can[k, y - uy0:y - uy0 + mh, x - ux0:x - ux0 + mw] = e["mask"]
        def cede(loser, take):
            """Zero `take` from loser's mask, but NEVER inside loser's own
            text block: that block is where this balloon's lettering has to
            go, so ceding it leaves the text outside every mask — it never
            gets cleaned (the old letters stay and the new text lands on
            top of them) and the fit has no room, so the size collapses.

            "Its own" excludes whatever the sibling's block claims too: a
            block is a blob cluster and p294 b07's reached over b12's first
            line, so this guard held its SIBLING's lettering inside b07 and
            with it the slice of b12's balloon b07's flood had filled."""
            other = (a if loser is b2 else b2)["block"]
            prot = own_block_mask(take.shape, (ux0, uy0), loser["block"],
                                  (other,),
                                  letters=letters[uy0:uy1, ux0:ux1])
            t = take & ~prot
            lx, ly = loser["bbox"][:2]
            lmh, lmw = loser["mask"].shape
            loser["mask"][t[ly - uy0:ly - uy0 + lmh,
                            lx - ux0:lx - ux0 + lmw]] = 0

        ov = (can[0] > 0) & (can[1] > 0)
        if not ov.any():
            continue
        only = [(can[k] > 0) & ~ov for k in (0, 1)]
        sums = [int((can[k] > 0).sum()) for k in (0, 1)]
        if (ov.sum() >= 0.85 * min(sums) and min(sums) <= 0.7 * max(sums)) \
                or not (only[0].any() and only[1].any()):
            # one flood swallowed a clearly SMALLER balloon whole (a lobe
            # drawn in FRONT of its sibling): the contained entry keeps its
            # body, the container cedes it. Near-equal full overlaps are a
            # joined pair whose floods both filled the whole shape — those
            # fall through to the notch split below.
            k_small = 0 if sums[0] <= sums[1] else 1
            loser = (a, b2)[1 - k_small]
            cede(loser, ov & (can[k_small] > 0))
            continue
        # OVERLAPPING (not joined) balloons: the FRONT one's outline runs
        # continuously through the overlap zone, so the interface between
        # the overlap and the BACK balloon's exclusive region crosses that
        # inked arc while the front's interface is clean interior (letters
        # excluded — the original text sits right next to the overlap and
        # pollutes the measure). The whole overlap goes to the clean side:
        # the front balloon keeps its full oval, where the original
        # letterer centers.
        dark_iface = (img[uy0:uy1, ux0:ux1].min(axis=2) <= DARK_MAX) \
            & ~(letters[uy0:uy1, ux0:ux1] > 0)
        band = cv2.dilate(ov.astype(np.uint8), np.ones((13, 13), np.uint8)) > 0
        iface = []
        for k in (0, 1):
            nb = (cv2.dilate(only[k].astype(np.uint8),
                             np.ones((13, 13), np.uint8)) > 0) & band
            iface.append(float(dark_iface[nb].mean()) if nb.any() else 1.0)
        if max(iface) > 0.04 and min(iface) < 0.5 * max(iface):
            loser = (a, b2)[iface.index(max(iface))]
            cede(loser, ov)
            continue
        # JOINED pair (ink on neither interface): one white shape with a
        # waist notch between the lobes. Split the UNION at the shallowest
        # column (or row) between the two text blocks — each lobe keeps its
        # whole side, so any rows one flood missed are repaired by the
        # sibling's coverage.
        union = ((can[0] > 0) | (can[1] > 0))
        axov = (min(a["block"][0] + a["block"][2],
                    b2["block"][0] + b2["block"][2])
                - max(a["block"][0], b2["block"][0]))
        ayov = (min(a["block"][1] + a["block"][3],
                    b2["block"][1] + b2["block"][3])
                - max(a["block"][1], b2["block"][1]))
        if (axov >= 0.25 * min(a["block"][2], b2["block"][2])
                and ayov >= 0.25 * min(a["block"][3], b2["block"][3])):
            # DIAGONAL lobes: the blocks overlap on BOTH axes, so neither a
            # row nor a column separates them and the window below runs
            # BACKWARDS — p294's cut fell at 1954, inside b07's own last
            # line, and the mask lost it (which the cleaner, bounded
            # to the mask, could not reach) while keeping the slice of b12's
            # balloon the flood had filled. Split on the seam equidistant
            # from each lobe's own lettering instead.
            seeds = []
            for e in (a, b2):
                sx, sy, sw, sh = e["block"]
                s = np.zeros(union.shape, np.uint8)
                s[max(0, sy - uy0):max(0, sy + sh - uy0),
                  max(0, sx - ux0):max(0, sx + sw - ux0)] = 1
                seeds.append(s & (letters[uy0:uy1, ux0:ux1] > 0))
            # a letter inside BOTH blocks says nothing about which lobe it
            # belongs to (b07's block swallowed b12's first line)
            both = (seeds[0] > 0) & (seeds[1] > 0)
            seeds = [s & ~both for s in seeds]
            pa, pb = seam_split(union, seeds)
            parts = {id(a): pa > 0, id(b2): pb > 0}
        elif axov < 0.25 * min(a["block"][2], b2["block"][2]):
            left, right = ((a, b2) if a["block"][0] <= b2["block"][0]
                           else (b2, a))
            lo = left["block"][0] + left["block"][2] - ux0 - 10
            hi = right["block"][0] - ux0 + 10
            lo, hi = max(0, min(lo, hi)), min(union.shape[1], max(lo, hi) + 1)
            prof = union[:, lo:hi].sum(axis=0)
            cut = lo + int(np.argmin(prof))
            pl, pr = union.copy(), union.copy()
            pl[:, cut:] = 0
            pr[:, :cut] = 0
            parts = {id(left): pl, id(right): pr}
        else:
            upper, lower = ((a, b2) if a["block"][1] <= b2["block"][1]
                            else (b2, a))
            lo = upper["block"][1] + upper["block"][3] - uy0 - 10
            hi = lower["block"][1] - uy0 + 10
            lo, hi = max(0, min(lo, hi)), min(union.shape[0], max(lo, hi) + 1)
            prof = union[lo:hi, :].sum(axis=1)
            cut = lo + int(np.argmin(prof))
            pu, pd = union.copy(), union.copy()
            pu[cut:, :] = 0
            pd[:cut, :] = 0
            parts = {id(upper): pu, id(lower): pd}
        for e in (a, b2):
            m = parts[id(e)].astype(np.uint8)
            if not m.any():
                continue
            e["mask"] = m
            e["bbox"] = [int(ux0), int(uy0),
                         int(m.shape[1]), int(m.shape[0])]

    # crop every mask to its nonzero extent: split lobes are created with
    # the whole component's bbox, which inflates the erosion pad below
    # (computed from bbox width) and biases the row centroid — the text then
    # sits off the visible lobe center
    for e in merged:
        if e.get("mask") is None:
            continue
        # ...and drop the DEBRIS the splits leave behind first, or the crop
        # is no crop at all: ceding its sibling's oval left p294 b07 with two
        # slivers of that balloon's rim, 1.4% and 0.4% of its mask, which
        # held its bbox 130px taller than the balloon it is for. A lobe is
        # never a two-percent fragment.
        nc, lc, sc, _ = cv2.connectedComponentsWithStats(e["mask"], 8)
        if nc > 2:
            tot = float(e["mask"].sum())
            drop = [i for i in range(1, nc) if sc[i, 4] < 0.02 * tot]
            if drop and len(drop) < nc - 1:
                for i in drop:
                    e["mask"][lc == i] = 0
        crop_to_mask(e)

    bubbles = []
    for e in merged:
        x, y, bw, bh = e["bbox"]
        if e["mask"] is None:
            interior = np.ones((bh, bw), np.uint8)
            pad = 0
        else:
            interior = e["mask"]
            pad = max(6, int(round(bw * PAD_FRAC)))
        inset = cv2.erode(interior, np.ones((pad, pad), np.uint8)) if pad \
            else interior
        rows = []
        for ry in range(bh):
            xs = np.flatnonzero(inset[ry])
            if len(xs):
                rows.append([int(ry), int(xs[0]), int(xs[-1]) + 1])
        if e["kind"] == "bubble":
            # every pixel the ORIGINAL text occupied is usable by the new
            # text: letters shredding the free space (text touching the
            # outline, or sitting in a pocket where the balloon pokes past
            # a panel border) must not cost the row
            # profile the width the letterer demonstrably used
            rowmap = {r[0]: r for r in rows}
            bx2, by2, bw2, bh2 = e["block"]
            sub = letters[by2:by2 + bh2, bx2:bx2 + bw2]
            for ly2 in range(bh2):
                xs2 = np.flatnonzero(sub[ly2])
                if not len(xs2):
                    continue
                ry = by2 + ly2 - y
                if not (0 <= ry < bh):
                    continue
                lx0 = int(bx2 + int(xs2[0]) - x)
                lx1 = int(bx2 + int(xs2[-1]) + 1 - x)
                r = rowmap.get(ry)
                if r is None:
                    r = [int(ry), lx0, lx1]
                    rowmap[ry] = r
                    rows.append(r)
                else:
                    r[1] = int(min(r[1], lx0))
                    r[2] = int(max(r[2], lx1))
            rows.sort()
            # bridge short PINCH runs sandwiched between wide rows (the
            # flood squeezing through a gap where a border crosses the
            # balloon): the letterer's text column flows straight across
            # them, and a single choked row vetoes every line box over it
            j = 1
            while j < len(rows) - 1:
                w_prev = rows[j - 1][2] - rows[j - 1][1]
                k = j
                while (k < len(rows) - 1 and k - j < 6
                       and rows[k][2] - rows[k][1] < 0.5 * w_prev):
                    k += 1
                w_next = rows[k][2] - rows[k][1]
                run_max = max((rows[m][2] - rows[m][1]
                               for m in range(j, k)), default=0)
                if k > j and w_next >= 2 * run_max:
                    x0 = max(rows[j - 1][1], rows[k][1])
                    x1 = min(rows[j - 1][2], rows[k][2])
                    if x1 > x0:
                        for m in range(j, k):
                            rows[m][1] = int(min(rows[m][1], x0))
                            rows[m][2] = int(max(rows[m][2], x1))
                    j = k
                else:
                    j += 1
        region = img[y:y + bh, x:x + bw]
        ink = region.min(axis=2) <= DARK_MAX
        fill = (interior > 0) & ~ink
        tinted = fill & (region.max(axis=2).astype(int)
                         - region.min(axis=2).astype(int) > 18)
        bubbles.append({
            "kind": e["kind"], "striped": bool(tinted.sum() > 0.12 * fill.sum()),
            "strict": bool(e.get("strict", False)),
            "group": e.get("group"),
            "paragraphs": e.get("paragraphs", 1),
            "bbox": [int(x), int(y), int(bw), int(bh)],
            "block": [int(v) for v in e["block"]],
            "rows": rows, "_mask": interior})
    # sort by the TEXT BLOCK position (stable across mask changes),
    # matching the order texts were transcribed in
    bubbles.sort(key=lambda b: (b["block"][1], b["block"][0]))
    return img, bubbles


def merge_boxes(a, b):
    x = min(a[0], b[0]); y = min(a[1], b[1])
    return [x, y, max(a[0] + a[2], b[0] + b[2]) - x,
            max(a[1] + a[3], b[1] + b[3]) - y]


# ported from process.py: classify mostly-paper pages (index/credits vs art)
INDEX_MIN_PAPER = 0.70
INDEX_MAX_INK_RUN = 0.15


def paper_page_kind(path):
    from PIL import Image
    im = Image.open(path)
    im.draft("RGB", (600, 1200))
    im = im.convert("RGB")
    im.thumbnail((600, 900))
    a = np.asarray(im, dtype=np.int16)
    h, w = a.shape[:2]
    ring = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3),
                           a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    paper = np.median(ring, axis=0)
    if (np.abs(a - paper).max(axis=2) <= 26).mean() < INDEX_MIN_PAPER:
        return None
    luma = a @ np.array([0.299, 0.587, 0.114])
    inky = (luma < 140).sum(axis=1) > 0.01 * w
    best = cur = 0
    for v in inky:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return "index" if best / h <= INDEX_MAX_INK_RUN else "art"


def is_story_page(stem):
    """Only comic story pages get re-lettered: no covers, index/credits,
    or art/extras pages."""
    import re as _re
    low = stem.lower()
    if "cover" in low or _re.search(r"\bart\b", low):
        return False
    src_stem = stem[:-2] if stem.endswith(("-1", "-2")) else stem
    src = next(iter(SRC_DIR.glob(src_stem + ".*")), None)
    if src is not None and paper_page_kind(src) in ("index", "art"):
        return False
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.png"):
        stale.unlink()
    for path in sorted(PAGES_DIR.glob("*.jpg")) + sorted(PAGES_DIR.glob("*.png")):
        if not is_story_page(path.stem):
            (OUT / f"{path.stem}.json").write_text("[]")
            print(f"{path.stem}: SKIPPED (cover/index/art)")
            continue
        img, bubbles = detect_page(path)
        for bi, b in enumerate(bubbles, 1):
            x, y, bw, bh = b["bbox"]
            mask = b.pop("_mask")
            cv2.imwrite(str(OUT / f"{path.stem}-b{bi:02d}-mask.png"),
                        (mask * 255).astype(np.uint8))
            crop = img[max(0, y - 8):y + bh + 8, max(0, x - 8):x + bw + 8]
            crop2 = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                               interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(str(OUT / f"{path.stem}-b{bi:02d}.png"),
                        cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR))
        (OUT / f"{path.stem}.json").write_text(json.dumps(bubbles, indent=1))
        print(f"{path.stem}: {len(bubbles)} bubbles "
              f"({sum(b['striped'] for b in bubbles)} striped, "
              f"{sum(b['kind'] == 'margin' for b in bubbles)} margin, "
              f"{sum(b['paragraphs'] > 1 for b in bubbles)} multi-par)")


if __name__ == "__main__":
    if COMIC is None:
        sys.exit('usage: reletter_detect.py "<comic folder name>"')
    sys.exit(main())
