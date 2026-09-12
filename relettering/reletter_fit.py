#!/usr/bin/env python3
"""Clean bubble interiors (wipe the old lettering, keep the fill: white or
stripes) and compute the new text layout: the largest font size whose
wrapped lines fit each bubble's inset row profile. Writes the cleaned
pages over pipeline/upscaled/... and a layout.json for the GIMP text step."""

import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

import sys

REPO = Path(__file__).resolve().parent.parent
if len(sys.argv) < 2:
    sys.exit('usage: reletter_fit.py "<comic folder name>"')
COMIC = sys.argv[1]
WORK = REPO / "relettering" / COMIC
PAGES_DIR = REPO / "upscaled" / COMIC
BUB = WORK / "bubbles"
# the comic font is USER-SUPPLIED (fonts are licensed material and are not
# in the repo): see relettering/fonts/README.md
FONT_DIR = Path(__file__).resolve().parent / "fonts"
FONTS = {"regular": FONT_DIR / "regular.ttf",
         "bolditalic": FONT_DIR / "bolditalic.ttf"}
for _style, _path in FONTS.items():
    if not _path.is_file():
        sys.exit("missing font file %s — supply your comic font as "
                 "described in relettering/fonts/README.md" % _path)
DARK_MAX = 110
LINE_SPACING = 1.10     # line height as a fraction of font size
MIN_SIZE, MAX_SIZE = 10, 80


def parse_runs(text: str) -> list:
    """'A *B C* D' -> [('regular','A'), ('bolditalic','B C'), ...] word-level:
    [(style, word), ...]"""
    words = []
    for m in re.finditer(r"\*([^*]+)\*|(\S+)", text):
        if m.group(1) is not None:
            words += [("bolditalic", w) for w in m.group(1).split()]
        else:
            words.append(("regular", m.group(2)))
    return words


class Fitter:
    def __init__(self):
        self._cache = {}

    def font(self, style, size):
        key = (style, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(FONTS[style]), size)
        return self._cache[key]

    def line_width(self, runs, size):
        space = self.font("regular", size).getlength(" ")
        return (sum(self.font(st, size).getlength(w) for st, w in runs)
                + space * (len(runs) - 1))

    def cap_height(self, size):
        f = self.font("regular", size)
        box = f.getbbox("H")
        return box[3] - box[1]


def rows_span(rows: dict, y0: int, y1: int):
    """Available [x0,x1] over the y-range, None if any row is missing."""
    x0, x1 = -1e9, 1e9
    for y in range(int(y0), int(y1)):
        r = rows.get(y)
        if r is None:
            return None
        x0 = max(x0, r[0])
        x1 = min(x1, r[1])
    return (x0, x1) if x1 > x0 else None


def span_avail(span, factor, anchor):
    """Usable width of a row span. With an anchor (the lobe's center axis),
    only the SYMMETRIC width about the anchor counts: every line renders
    centered on that axis, so a waist-transition row that sticks out to one
    side must not lure a line off-axis (it used to get clamped into its own
    lopsided span, jutting sideways in joint bubbles)."""
    if anchor is None:
        return (span[1] - span[0]) * factor
    return 2.0 * min(anchor - span[0], span[1] - anchor) * factor


def wrap_at(fit, words, rows, size, cy, hard_breaks, factor=0.96,
            para_breaks=(), anchor=None, lift=0.0, hold_cy=False):
    """Try to wrap words into lines centered vertically on cy. Returns
    [(y_top, x0, x1, line_runs)] or None. hard_breaks = word indices after
    which a line break is forced; para_breaks additionally add a visual gap
    (compound balloons hold several paragraphs)."""
    lh = max(1, round(size * LINE_SPACING))
    gap = round(0.55 * lh)
    breaks = set(hard_breaks) | set(para_breaks)
    for n in range(1, 14):
        total = n * lh + gap * len(para_breaks)
        top = cy - total / 2.0
        # geometry pass: available width per line (greedy fill drives the
        # paragraph-gap offsets, so compute spans with a rough greedy first)
        geo, i = [], 0
        extra = 0
        ok = True
        for li in range(n):
            y0 = top + li * lh + extra
            # glyphs fill ~70% of the line box, centered: the outer edge of
            # the first/last line may overhang the mask a little
            v0 = y0 + (0.16 * lh if li == 0 else 0)
            v1 = y0 + lh - (0.16 * lh if li == n - 1 else 0)
            span = rows_span(rows, v0, v1)
            if span is None:
                ok = False
                break
            # width factor: Pango-vs-PIL safety plus lateral breathing room
            avail = span_avail(span, factor, anchor)
            if avail <= 0:
                ok = False
                break
            line = []
            while i < len(words):
                cand = line + [words[i]]
                if fit.line_width(cand, size) <= avail:
                    line = cand
                    i += 1
                    if (i - 1) in breaks:
                        break
                else:
                    break
            if not line:
                ok = False
                break
            geo.append((y0, span[0], span[1], avail, line))
            if (i - 1) in para_breaks:
                extra += gap
        if not (ok and i == len(words)):
            continue
        # balanced pass: redistribute words over these n lines minimizing
        # squared slack against each line's width profile ("pyramid" look)
        split = balanced_split(fit, words, [g[3] for g in geo], size, breaks)
        lines = []
        if split is not None:
            j = 0
            fits = True
            for (y0, x0, x1, avail, _), line in zip(geo, split):
                if fit.line_width(line, size) > avail:
                    fits = False
                    break
                lines.append((y0, x0, x1, line))
                j += len(line)
            if not fits:
                lines = [(y0, x0, x1, line) for (y0, x0, x1, _, line) in geo]
        else:
            lines = [(y0, x0, x1, line) for (y0, x0, x1, _, line) in geo]
        if hold_cy:
            # the caller's cy is authoritative (the original letterer's
            # center) — no clearance re-balancing
            return lines
        return balance(fit, rows, lines, size, factor, anchor, lift)
    return None


def balanced_split(fit, words, avails, size, breaks):
    """Split words into len(avails) lines minimizing sum((avail-width)^2),
    honoring forced breaks. Returns list of word-lists or None."""
    n, m = len(avails), len(words)
    INF = float("inf")
    # cost[a][b): words a..b-1 on one line
    width = {}

    def line_cost(a, b, li):
        w = fit.line_width(words[a:b], size)
        if w > avails[li]:
            return INF, w
        # forced breaks may only occur at the line's end
        for k in range(a, b - 1):
            if k in breaks:
                return INF, w
        return (avails[li] - w) ** 2, w

    best = [[INF] * (m + 1) for _ in range(n + 1)]
    prev = [[-1] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for li in range(1, n + 1):
        for j in range(li, m + 1):
            for a in range(li - 1, j):
                if best[li - 1][a] == INF:
                    continue
                c, _ = line_cost(a, j, li - 1)
                if c == INF:
                    continue
                if best[li - 1][a] + c < best[li][j]:
                    best[li][j] = best[li - 1][a] + c
                    prev[li][j] = a
    if best[n][m] == INF:
        return None
    out = []
    j = m
    for li in range(n, 0, -1):
        a = prev[li][j]
        out.append(words[a:j])
        j = a
    return out[::-1]


def placement_ok(fit, rows, lines, size, factor, dy, anchor=None):
    lh = max(1, round(size * LINE_SPACING))
    n = len(lines)
    for li, (y0, _, _, runs) in enumerate(lines):
        v0 = y0 + dy + (0.16 * lh if li == 0 else 0)
        v1 = y0 + dy + lh - (0.16 * lh if li == n - 1 else 0)
        span = rows_span(rows, v0, v1)
        if span is None or span_avail(span, factor, anchor) < \
                fit.line_width(runs, size):
            return None
    return [(y0 + dy, x0, x1, runs) for (y0, x0, x1, runs) in lines]


def balance(fit, rows, lines, size, factor, anchor=None, lift=0.0):
    """Shift the fitted block vertically so the unused mask rows above the
    first line and below the last line even out (the wrap centers on the
    mask midpoint, which reads off-center in clipped/asymmetric bubbles)."""
    lh = max(1, round(size * LINE_SPACING))
    def wide_enough(yy, need):
        r = rows.get(yy)
        return r is not None and (r[1] - r[0]) >= need

    def clearances(ls):
        top = ls[0][0] + 0.16 * lh
        bot = ls[-1][0] + lh - 0.16 * lh
        need_t = 0.5 * fit.line_width(ls[0][3], size)
        need_b = 0.5 * fit.line_width(ls[-1][3], size)
        up = 0
        while wide_enough(int(top) - up - 1, need_t):
            up += 1
        dn = 0
        while wide_enough(int(bot) + dn + 1, need_b):
            dn += 1
        return up, dn
    up, dn = clearances(lines)
    # equalize clearances, biased upward by the optical lift (the block
    # reads centered when it sits slightly above the geometric middle —
    # single lines included: user-tuned on a short-shout lobe)
    want = int(round((dn - up) / 2.0 - lift))
    step = 1 if want > 0 else -1
    best = lines
    for dy in range(step, want + step, step):
        cand = placement_ok(fit, rows, lines, size, factor, dy, anchor)
        if cand is None:
            break
        best = cand
    return best


def centroid_y(rows: dict) -> float:
    wsum = sum(x1 - x0 for x0, x1 in rows.values())
    return sum(y * (x1 - x0) for y, (x0, x1) in rows.items()) / wsum


def centroid_x(rows: dict) -> float:
    wsum = sum(x1 - x0 for x0, x1 in rows.values())
    return sum((x0 + x1) / 2.0 * (x1 - x0)
               for x0, x1 in rows.values()) / wsum


def find_waist(rows: dict):
    """Split a two-lobe (figure-8) row profile at its deepest interior
    minimum. Returns (upper band, lower band) rows dicts, or None when the
    profile has no real waist (a single oval holding a text pause)."""
    ys = sorted(rows)
    if len(ys) < 24:
        return None
    wd = np.array([rows[y][1] - rows[y][0] for y in ys], float)
    sm = np.convolve(wd, np.ones(5) / 5.0, mode="same")
    lo, hi = 8, len(ys) - 8
    if hi <= lo:
        return None
    i = lo + int(np.argmin(sm[lo:hi]))
    if sm[i] > 0.72 * min(sm[:i].max(), sm[i:].max()):
        return None
    return ({y: rows[y] for y in ys[:i]}, {y: rows[y] for y in ys[i:]})


def visual_center(rows: dict):
    """The point the eye reads as a lobe's center: the distance-transform
    peak (center of the largest inscribed circle). Width-weighted centroids
    get dragged toward waist spill and wide bottoms — user-confirmed on the
    'MUY HONORABLE CONDE' lobe (pulled right) and several 'too low' blocks."""
    ys = sorted(rows)
    y0, h = ys[0], ys[-1] - ys[0] + 1
    x0 = min(r[0] for r in rows.values())
    x1 = max(r[1] for r in rows.values())
    m = np.zeros((h, x1 - x0), np.uint8)
    for y in ys:
        a, b = rows[y]
        m[y - y0, a - x0:b - x0] = 1
    # pad first: distanceTransform measures to the nearest ZERO, and a mask
    # touching the array edge has none there — the peak lands in the corner
    m = np.pad(m, 8)
    d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    # flat lobes give a PLATEAU of near-max distances; argmax alone picks
    # its first (top-left) pixel — take the plateau's centroid instead
    ys2, xs2 = np.nonzero(d >= 0.9 * d.max())
    return float(x0 + xs2.mean() - 8), float(y0 + ys2.mean() - 8)


def optical_lift(size):
    """Text blocks read as centered when they sit slightly ABOVE the
    geometric center (classic optical centering)."""
    return 0.30 * max(1, round(size * LINE_SPACING))


def block_centered(lines, rows, size, target_cy):
    """A wrap is only acceptable when the text block sits on the lobe's
    visual vertical center (with the optical lift): when the waist rows of
    a joint bubble can't host centered lines, the block gets pushed to the
    far end of the lobe and the eye reads it as misplaced — the original
    letterers drop the size until the block floats centered instead."""
    lh = max(1, round(size * LINE_SPACING))
    block_cy = (lines[0][0] + lines[-1][0] + lh) / 2.0
    tol = max(0.45 * lh, 0.05 * (max(rows) - min(rows)))
    return abs(block_cy - target_cy) <= tol


def fit_lines(fit, p, size):
    """Wrap p's words at `size`. Two-paragraph bubbles whose mask has a real
    waist get each paragraph wrapped INSIDE its own lobe, centered on that
    lobe's centroid — a whole-mask wrap centers the block on the union
    centroid and lets lines drift toward the waist. Returns
    [(y_top, x0, x1, runs, cx_anchor)] or None; cx_anchor is the lobe's
    width-weighted x-center each line should center on (clamped to its own
    row span when building the layout entry)."""
    rows = p["rows"]
    if p["b"]["kind"] == "bubble" and len(p["paras"]) == 1:
        bands = find_waist(rows)
        if bands is not None:
            pi = next(iter(p["paras"]))  # last word index of paragraph 1
            words1, words2 = p["words"][:pi + 1], p["words"][pi + 1:]
            hard1 = {h for h in p["hard"] if h < pi}
            hard2 = {h - (pi + 1) for h in p["hard"] if h > pi}
            lift = optical_lift(size)
            (a1x, a1y), (a2x, a2y) = visual_center(bands[0]), visual_center(bands[1])
            l1 = wrap_at(fit, words1, bands[0], size,
                         a1y - lift, hard1, anchor=a1x, lift=lift)
            l2 = wrap_at(fit, words2, bands[1], size,
                         a2y - lift, hard2, anchor=a2x, lift=lift)
            if l1 is not None and l2 is not None \
                    and block_centered(l1, bands[0], size, a1y - lift) \
                    and block_centered(l2, bands[1], size, a2y - lift):
                return ([(y0, x0, x1, r, a1x) for (y0, x0, x1, r) in l1]
                        + [(y0, x0, x1, r, a2x) for (y0, x0, x1, r) in l2])
            # banding failed at this size — fall THROUGH to the anchor
            # ladder below rather than giving up: whole-block placement with
            # per-paragraph axes (para_line_axes) handles the lobes too, and
            # returning None here silently capped dense two-lobe balloons
            # several px below what the anchors could place
    is_bubble = p["b"]["kind"] == "bubble"
    if is_bubble:
        # THE anchor: the ORIGINAL letterer's own text center (the detected
        # block box center). Geometry (visual_center) only approximates it
        # to +-6px and can never recover editorial placement (the original
        # can set a short shout 13px above every geometric center) — so center
        # the new text exactly where the original text was centered.
        bx, by, bw, bh = p["b"]["block"]
        cxa, cya = bx + bw / 2.0, by + bh / 2.0
        anchors = [(cxa, cya, True)]
        vx, vy = visual_center(rows)
        if abs(vx - cxa) > 4:
            # hybrid fallback: keep the letterer's VERTICAL center (the eye
            # forgives a sideways nudge onto the lobe axis far more than a
            # line-height drop) and center horizontally on the visible lobe
            anchors.append((float(vx), cya, True))
        if abs(vx - cxa) > 4 or abs(vy - cya) > 4:
            # last resort: the lobe's own visual center — still HELD, not
            # re-balanced: balance() re-centers geometrically on asymmetric
            # masks and dropped blocks a full line low (user-reported)
            anchors.append((float(vx), float(vy), True))
        if (p.get("ov") or {}).get("anchor") == "visual":
            anchors = [(float(vx), float(vy), True)]
        if (p.get("ov") or {}).get("anchor") == "lines":
            # per-line centering on each line's own row span: for balloons
            # whose shape shifts sideways along their height (e.g. a top
            # protrusion above a panel border) the original letterer follows
            # the shape — no single axis can reproduce that
            anchors = [(None, cya, True)]
        lh = max(1, round(size * LINE_SPACING))
        max_dy = max(9, int(round(0.9 * lh)))
        dys = [0]
        for d in range(3, max_dy + 1, 3):
            dys += [-d, d]
        for cxa, cya, hold in anchors:
            # minimal-deviation vertical search: an anchor that cannot host
            # this size EXACTLY at the letterer's center often can a few px
            # away (a protrusion or clipped arc chokes only the outermost
            # line) — a small slide beats falling through to the next
            # anchor, whose x can sit tens of px off the original axis
            for dy in dys:
                lines = wrap_at(fit, p["words"], rows, size, cya + dy,
                                p["hard"], para_breaks=p["paras"],
                                anchor=cxa, hold_cy=hold)
                if lines is None or not block_centered(lines, rows, size,
                                                       cya + dy):
                    continue
                if cxa is None and p.get("oaxis"):
                    # follow the original letterer's per-line axes
                    out = []
                    for (y0, x0, x1, r) in lines:
                        mids = [m for yy, m in p["oaxis"].items()
                                if y0 <= yy < y0 + lh]
                        ax = sum(mids) / len(mids) if mids else None
                        out.append((y0, x0, x1, r, ax))
                    return out
                if p["paras"]:
                    axes = para_line_axes(rows, lines, size, p["paras"],
                                          p.get("porig"))
                    return [(y0, x0, x1, r, ax)
                            for (y0, x0, x1, r), ax in zip(lines, axes)]
                return [(y0, x0, x1, r, cxa) for (y0, x0, x1, r) in lines]
        # LAST rung: per-line spans with the ORIGINAL letterer's per-line
        # axes. In a densely packed balloon no single axis can host the book
        # size — the original letterer places line by line against the
        # shape, and following those measured axes recovers 2-4px of font
        # size (dense balloons read "small" at the single-axis ceiling).
        # Confined to the original text's vertical envelope (± one line
        # height) so lines can never wander into a leak beyond the balloon.
        if p.get("oaxis"):
            bx2, by2, bw2, bh2 = p["b"]["block"]
            bcy = by2 + bh2 / 2.0
            # tight above (the original's first line marks the safe top —
            # rows above it can be a border/gutter zone), looser below; and
            # near-RAW mask rows: the assembly inset erodes 10-15px per side
            # that the original letterer demonstrably used (their lines hug
            # the outline and the sibling balloon's arc) — this rung packs
            # the way they did
            src = p.get("rows_raw") or rows
            rows2 = {yy: sp for yy, sp in src.items()
                     if by2 - 0.25 * lh <= yy <= by2 + bh2 + 0.75 * lh}
            if len(rows2) >= 3:
                for dy in dys:
                    lines = wrap_at(fit, p["words"], rows2, size, bcy + dy,
                                    p["hard"], para_breaks=p["paras"],
                                    anchor=None, hold_cy=True)
                    if lines is None:
                        continue
                    out = []
                    for (y0, x0, x1, r) in lines:
                        mids = [m for yy, m in p["oaxis"].items()
                                if y0 <= yy < y0 + lh]
                        ax = sum(mids) / len(mids) if mids else None
                        out.append((y0, x0, x1, r, ax))
                    return out
        return None
    cxa = centroid_x(rows)
    lines = wrap_at(fit, p["words"], rows, size, p["cys"][0], p["hard"],
                    para_breaks=p["paras"])
    if lines is None:
        return None
    return [(y0, x0, x1, r, cxa) for (y0, x0, x1, r) in lines]


def orig_para_axes(img, b, nparas):
    """Per-paragraph x-axes of the ORIGINAL lettering: letter-sized dark
    blobs inside the text block, rows clustered at the (nparas-1) largest
    vertical gaps. The letterer's own axis per paragraph — geometry only
    approximates it (one lobe can hang sideways of the other).
    Returns a list of axes or None when the blobs don't support a split."""
    bx, by, bw, bh = b["block"]
    reg = img[by:by + bh, bx:bx + bw]
    dark = (reg.min(axis=2) <= DARK_MAX).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    boxes = [tuple(int(v) for v in stats[i][:5]) for i in range(1, n)
             if 9 <= stats[i][3] <= 48 and stats[i][2] <= 130
             and 8 <= stats[i][4] <= 2600]
    if len(boxes) < 2 * nparas:
        return None
    boxes.sort(key=lambda s: (s[1], s[0]))
    gaps = []
    for i in range(1, len(boxes)):
        gaps.append((boxes[i][1] - max(bb2[1] + bb2[3]
                                       for bb2 in boxes[:i]), i))
    cuts = sorted(i for _, i in sorted(gaps, reverse=True)[:nparas - 1])
    groups, prev = [], 0
    for c in cuts + [len(boxes)]:
        groups.append(boxes[prev:c])
        prev = c
    axes = []
    for g in groups:
        if len(g) < 2:
            return None
        x0 = min(s[0] for s in g)
        x1 = max(s[0] + s[2] for s in g)
        axes.append(bx + (x0 + x1) / 2.0)
    return axes


def orig_axis_map(img, b):
    """Per-row x-axis of the ORIGINAL lettering (letter-blob extent middle
    per block row) — the 'lines' anchor override follows it line by line,
    for balloons whose shape shifts sideways along their height."""
    bx, by, bw, bh = b["block"]
    reg = img[by:by + bh, bx:bx + bw]
    dark = (reg.min(axis=2) <= DARK_MAX).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros(n, bool)
    for i in range(1, n):
        _, _, w2, h2, area = stats[i]
        if 9 <= h2 <= 48 and w2 <= 130 and 8 <= area <= 2600:
            keep[i] = True
    lm = keep[labels]
    out = {}
    for ry in range(bh):
        xs = np.flatnonzero(lm[ry])
        if len(xs):
            out[by + ry] = bx + (int(xs[0]) + int(xs[-1])) / 2.0
    return out


def para_line_axes(rows, lines, size, paras, porig=None):
    """A multi-paragraph bubble whose lobes have no row-profile waist wraps
    as ONE block on one axis — but each paragraph sits in its own lobe, and
    the original letterer centers it on that lobe's own axis (a diagonal
    figure-8: one lobe hangs sideways of the other). Give each paragraph's
    lines
    the width-weighted x-centroid of the mask rows IT occupies (line_cx
    clamps every line into its own row span, so a recentered line can never
    leave the mask). Returns one axis per line."""
    lh = max(1, round(size * LINE_SPACING))
    breaks = sorted(paras)
    groups, cur, consumed, bi = [], [], 0, 0
    for ln in lines:
        cur.append(ln)
        consumed += len(ln[3])
        if bi < len(breaks) and consumed - 1 >= breaks[bi]:
            groups.append(cur)
            cur, bi = [], bi + 1
    if cur:
        groups.append(cur)
    axes = []
    for k, g in enumerate(groups):
        if porig is not None and len(porig) == len(groups):
            # the original letterer's own axis for this paragraph
            ax = porig[k]
        else:
            sel = {y: rows[y] for y in rows
                   if g[0][0] <= y < g[-1][0] + lh}
            # inscribed-circle peak, not the centroid: a lobe bulging toward
            # its speaker drags the width-weighted centroid off the axis the
            # eye (and the original letterer) uses
            ax = visual_center(sel)[0] if len(sel) >= 3 \
                else (g[0][1] + g[0][2]) / 2.0
        axes += [ax] * len(g)
    return axes


def line_cx(fit, size, x0, x1, runs, cx_anchor):
    """Center a line on its lobe's x-centroid, clamped so it stays inside
    its own row span (falls back to the span midpoint when it can't fit)."""
    lw = fit.line_width(runs, size)
    lo, hi = x0 + lw / 2.0 + 2, x1 - lw / 2.0 - 2
    if lo > hi or cx_anchor is None:
        # no axis: each line centers on its own row span (the "lines"
        # anchor override, for balloons whose shape shifts sideways)
        return (x0 + x1) / 2.0
    return min(max(cx_anchor, lo), hi)


def relax_group_caps(fit, pending, group_cap):
    """Trade a little of a group's shared size for better wraps: one pixel
    whenever a member then needs FEWER lines, and up to three pixels when a
    SHORT utterance (<= 4 words, e.g. a repeated shout) reaches a SINGLE line —
    the original letterers set those as one line, and a stacked short text
    never reads centered."""
    for g, cap in list(group_cap.items()):
        best = cap
        for p in pending:
            if p["b"].get("group") != g:
                continue
            at_cap = fit_lines(fit, p, cap)
            if not at_cap:
                continue
            if cap - 1 > MIN_SIZE:
                down = fit_lines(fit, p, cap - 1)
                if down and len(down) < len(at_cap):
                    best = min(best, cap - 1)
        group_cap[g] = best


def clip_rows(rows: dict, cy0: int) -> dict:
    """Keep only the vertical run of rows that belongs to the bubble around
    cy0: walking away from the center, the ellipse width shrinks — a sharp
    re-expansion means the mask leaked through a gap (e.g. into the page
    margin where a panel border clips the bubble)."""
    ys = sorted(rows)
    cy0 = min(ys, key=lambda yy: abs(yy - cy0))
    keep = {cy0}
    for step in (1, -1):
        wmin = prev = rows[cy0][1] - rows[cy0][0]
        y = cy0 + step
        while y in rows:
            w = rows[y][1] - rows[y][0]
            if w > wmin * 1.3 and w > prev:
                break
            wmin, prev = min(wmin, w), w
            keep.add(y)
            y += step
    return {y: rows[y] for y in keep}


def clean_bubble(img, b, mask, allowed_ys=None):
    x, y, w, h = b["bbox"]
    region = img[y:y + h, x:x + w]
    if b["kind"] == "dark":
        # white-on-black caption: wipe the white letters with the row median
        # of the surrounding dark box. Dilate the selection so the letters'
        # anti-aliased gray rims go with them — wiping only the bright cores
        # leaves hollow letter outlines behind the new text.
        ink = (region.min(axis=2) >= 150).astype(np.uint8)
        ink = cv2.dilate(ink, np.ones((5, 5), np.uint8)) > 0
        for ry in range(h):
            sel = ink[ry]
            if not sel.any():
                continue
            good = ~sel
            med = (np.median(region[ry][good], axis=0) if good.any()
                   else np.array([0, 0, 0]))
            region[ry][sel] = med.astype(np.uint8)
        img[y:y + h, x:x + w] = region
        return
    if b["kind"] == "tint":
        # tinted caption box (dark navy letters on blue fill): wipe the
        # letters with the row median of the surrounding fill — the 'dark'
        # branch with the ink sense inverted
        ink = (region.max(axis=2) <= 85).astype(np.uint8)
        ink = cv2.dilate(ink, np.ones((5, 5), np.uint8)) > 0
        for ry in range(h):
            sel = ink[ry]
            if not sel.any():
                continue
            good = ~sel
            med = (np.median(region[ry][good], axis=0) if good.any()
                   else np.array([0, 0, 0]))
            region[ry][sel] = med.astype(np.uint8)
        img[y:y + h, x:x + w] = region
        return
    if b["kind"] == "bubble":
        # row-median fill keeps white AND stripes. Repaint ONLY the letters
        # plus a halo — flat-filling every mask pixel plateaus the fill and,
        # where a mask has a straight edge (waist split, bound clip), the
        # tonal step reads as a white RECTANGLE inside the bubble.
        keep_ring = cv2.erode(mask, np.ones((3, 3), np.uint8))
        ink = (region.min(axis=2) <= (DARK_MAX + 40)).astype(np.uint8)
        halo = cv2.dilate(ink, np.ones((9, 9), np.uint8)) > 0
        ink = ink > 0
        for ry in range(h):
            if allowed_ys is not None and (y + ry) not in allowed_ys:
                continue
            sel = (keep_ring[ry] > 0) & halo[ry]
            if not sel.any():
                continue
            good = (keep_ring[ry] > 0) & ~halo[ry]
            med = (np.median(region[ry][good], axis=0) if good.any()
                   else np.array([255, 255, 255]))
            region[ry][sel] = med.astype(np.uint8)
        img[y:y + h, x:x + w] = region
        # OVERFLOW lines: the original letterer sometimes spills a line past
        # the balloon outline onto the page margin — outside any honest
        # mask. Inpaint letter-sized dark blobs inside the TEXT BLOCK box
        # that the mask fill did not reach.
        bx2, by2, bw2, bh2 = b["block"]
        ox0, oy0 = max(0, bx2 - 8), max(0, by2 - 8)
        ox1 = min(img.shape[1], bx2 + bw2 + 8)
        oy1 = min(img.shape[0], by2 + bh2 + 8)
        oreg = img[oy0:oy1, ox0:ox1]
        covered = np.zeros(img.shape[:2], np.uint8)
        covered[y:y + h, x:x + w] = mask
        darkish = ((oreg.min(axis=2) <= (DARK_MAX + 40))
                   & (covered[oy0:oy1, ox0:ox1] == 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(darkish, 8)
        letters = np.zeros_like(darkish)
        wh, ww = darkish.shape
        for i in range(1, n):
            cx2, cy2, cw, ch, area = stats[i]
            if not (ch <= 52 and cw <= 130 and area <= 2600):
                continue
            # a blob touching the window border is a clipped wedge of the
            # SURROUNDINGS (dark panel between the window's straight edge
            # and the bubble's arc) — inpainting it paints the background
            # white up to the window edge: the "white rectangle behind the
            # bubble". Real overflow text never touches the border (the
            # window is the text block plus an 8px margin).
            if cx2 == 0 or cy2 == 0 or cx2 + cw >= ww or cy2 + ch >= wh:
                continue
            letters[labels == i] = 1
        if letters.any():
            letters = cv2.dilate(letters, np.ones((5, 5), np.uint8))
            img[oy0:oy1, ox0:ox1] = cv2.inpaint(
                oreg, letters, 5, cv2.INPAINT_TELEA)
        return
    else:
        # open/margin: inpaint ONLY letter-sized dark blobs (plus halo) so
        # bubble outlines, panel borders and art are never touched
        darkish = (region.min(axis=2) <= 175).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(darkish, 8)
        letters = np.zeros_like(darkish)
        for i in range(1, n):
            _, _, cw, ch, area = stats[i]
            if ch <= 52 and cw <= 130 and area <= 2600:
                letters[labels == i] = 1
        letters = cv2.dilate(letters, np.ones((5, 5), np.uint8))
        region[:] = cv2.inpaint(region, letters, 5, cv2.INPAINT_TELEA)
    img[y:y + h, x:x + w] = region


def old_cap_height(img, b):
    bx, by, bw, bh = b["block"]
    reg = img[by:by + bh, bx:bx + bw]
    if b["kind"] == "dark":
        dark = (reg.min(axis=2) >= 200).astype(np.uint8)
    elif b["kind"] == "tint":
        dark = (reg.max(axis=2) <= 85).astype(np.uint8)
    else:
        dark = (reg.min(axis=2) <= DARK_MAX).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    hs = [stats[i][3] for i in range(1, n)
          if 9 <= stats[i][3] <= 48 and stats[i][4] >= 8]
    return float(np.median(hs)) if hs else 0.0


# corrections to the original lettering's own errors, applied at typeset
# time (transcripts.json stays a verbatim record of the source). Per-comic
# data: relettering/<comic>/typo_fixes.json, a list of [wrong, fixed] pairs
_tf = WORK / "typo_fixes.json"
TYPO_FIXES = [tuple(x) for x in json.loads(_tf.read_text())] \
    if _tf.is_file() else []

# per-bubble editorial overrides (per-comic data, like typo_fixes):
# relettering/<comic>/layout_overrides.json = {"1-012 b03": {"size": 17}}
# pins that bubble's font size (bypassing group caps) — the escape hatch
# for taste calls no global rule should be bent around
_lo = WORK / "layout_overrides.json"
LAYOUT_OVERRIDES = json.loads(_lo.read_text()) if _lo.is_file() else {}


def override_for(stem: str, bi: int):
    import re as _re
    m = _re.match(r"^\D*(\d+)\b.*?(\d+(?:-\d)?)$", stem)
    key = f"{m.group(1)}-{m.group(2)} b{bi:02d}" if m else f"{stem} b{bi:02d}"
    return LAYOUT_OVERRIDES.get(key)


def apply_fixes(text):
    for a, b in TYPO_FIXES:
        if a in text:
            print(f"    typo fix: {a!r} -> {b!r}")
            text = text.replace(a, b)
    return text


BOOK_PCTL = 35  # percentile of per-bubble maxima used as the common size


def main():
    transcripts = json.loads((WORK / "transcripts.json").read_text())
    fit = Fitter()
    layout = {}
    report = []
    all_pending = []
    for path in sorted(PAGES_DIR.glob("*.jpg")) + sorted(PAGES_DIR.glob("*.png")):
        bpath = BUB / (path.stem + ".json")
        if not bpath.is_file():
            continue
        bubbles = json.loads(bpath.read_text())
        if not bubbles:
            continue
        texts = transcripts.get(path.stem)
        if texts is None:
            print(f"  !! {path.stem}: {len(bubbles)} bubbles but no "
                  f"transcripts — SKIPPED")
            continue
        assert len(texts) == len(bubbles), \
            f"{path.stem}: {len(texts)} texts vs {len(bubbles)} bubbles"
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        page_entries = []
        pending = []
        clean_jobs = []
        all_pending.append((path, pending, clean_jobs, img))
        tpos = 0
        for bi, b in enumerate(bubbles, 1):
            text = apply_fixes(texts[bi - 1])
            if not text.strip():
                continue  # "" = leave this region untouched (SFX, display)
            x, y, w, h = b["bbox"]
            mask = (cv2.imread(str(BUB / f"{path.stem}-b{bi:02d}-mask.png"),
                               cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            oldh = old_cap_height(img, b)

            rows = {y + r[0]: (x + r[1], x + r[2]) for r in b["rows"]}
            allowed = None
            rows_raw = None
            if b["kind"] == "bubble":
                # strict-path masks are already leak-free; clip_rows would
                # wrongly cut compound balloons at their waist
                if not b.get("strict"):
                    bx_, by_, bw_, bh_ = b["block"]
                    rows = clip_rows(rows, by_ + bh_ // 2)
                pad = max(6, int(round(w * 0.05)))
                allowed = set(range(min(rows) - pad, max(rows) + pad + 1))
                # near-raw rows for the fit's last rung (rim erode only)
                m2 = cv2.erode(mask, np.ones((3, 3), np.uint8))
                rows_raw = {}
                for ry2 in range(m2.shape[0]):
                    xs2 = np.flatnonzero(m2[ry2])
                    if len(xs2):
                        rows_raw[y + ry2] = (x + int(xs2[0]),
                                             x + int(xs2[-1]) + 1)
                for yy2, sp2 in rows.items():
                    old2 = rows_raw.get(yy2)
                    rows_raw[yy2] = (sp2 if old2 is None
                                     else (min(old2[0], sp2[0]),
                                           max(old2[1], sp2[1])))
            clean_jobs.append((bi, b, mask, allowed))

            # words with hard breaks (explicit \n) and paragraph breaks
            words, hard, paras = [], set(), set()
            for para in text.split("\n\n"):
                for part in para.split("\n"):
                    words += parse_runs(part)
                    hard.add(len(words) - 1)
                paras.add(len(words) - 1)
            hard -= paras
            paras.discard(len(words) - 1)
            hard.discard(len(words) - 1)
            # width-weighted centroid: narrow tail/arc rows barely count, so
            # the centering target is the bubble's visual center
            wsum = sum(x1 - x0 for x0, x1 in rows.values())
            cy = sum(yy * (x1 - x0)
                     for yy, (x0, x1) in rows.items()) / wsum
            pending.append({"bi": bi, "b": b, "words": words, "hard": hard,
                            "paras": paras, "rows": rows, "cys": [cy],
                            "oldh": oldh, "ov": override_for(path.stem, bi),
                            "porig": (orig_para_axes(img, b, len(paras) + 1)
                                      if paras else None),
                            "oaxis": (orig_axis_map(img, b)
                                      if b["kind"] == "bubble" else None),
                            "rows_raw": rows_raw})

        print(f"{path.stem}: {len(pending)} bubbles collected")

    # ---- global fitting ----
    def best_fit(p, cap):
        for size in range(cap, MIN_SIZE, -1):
            lines = fit_lines(fit, p, size)
            if lines:
                return size, lines
        return None, None

    # per-bubble maxima (margin/open notes stay near their original size)
    for _, pending, _, _ in all_pending:
        for p in pending:
            cap = MAX_SIZE
            if p["b"]["kind"] == "margin" and p["oldh"]:
                ratio = fit.cap_height(100) / 100.0
                cap = max(MIN_SIZE + 1,
                          int(round(p["oldh"] / ratio * 1.1)))
            p["max_size"], _ = best_fit(p, cap)

    # the book's common size: a low percentile of the dialogue maxima, so
    # nearly all bubbles carry the same size and only the densest shrink
    dialogue_max = [p["max_size"] for _, pen, _, _ in all_pending for p in pen
                    if p["b"]["kind"] == "bubble" and p["max_size"]]
    book = int(np.percentile(dialogue_max, BOOK_PCTL))
    print(f"\nbubble maxima: {sorted(dialogue_max)}")
    print(f"book size = p{BOOK_PCTL} = {book}px "
          f"({sum(m >= book for m in dialogue_max)}/{len(dialogue_max)} "
          f"bubbles at full size)")

    ratio = fit.cap_height(100) / 100.0
    for path, pending, clean_jobs, img in all_pending:
        page_entries = []
        fitted = set()
        # grouped lobes of a compound balloon share the smallest size
        group_cap = {}
        for p in pending:
            g = p["b"].get("group")
            if g is not None and p["max_size"]:
                group_cap[g] = min(group_cap.get(g, 999),
                                   p["max_size"], book)
        relax_group_caps(fit, pending, group_cap)
        for p in pending:
            bi, b, words, oldh = p["bi"], p["b"], p["words"], p["oldh"]
            cap = p["max_size"] or MAX_SIZE
            if b["kind"] in ("bubble", "dark", "open", "tint"):
                orig_px = int(round(oldh / ratio)) if oldh else 0
                cap = min(cap, max(book, orig_px))
            g = b.get("group")
            if g is not None and g in group_cap:
                cap = group_cap[g]
            ov = override_for(path.stem, bi)
            if ov and "size" in ov:
                cap = ov["size"]
            size, lines = best_fit(p, cap)
            if not size:
                print(f"  !! {path.stem} b{bi:02d}: no fit found")
                continue
            lh = max(1, round(size * LINE_SPACING))
            newh = fit.cap_height(size)
            styles = {st for st, _ in words}
            entry = {
                "index": bi, "kind": b["kind"],
                "color": "white" if b["kind"] == "dark" else "black",
                "font_px": size, "line_height": lh,
                "uniform": list(styles)[0] if len(styles) == 1 else None,
                "old_cap_h": oldh, "new_cap_h": newh,
                "lines": [{
                    "y_top": round(ly, 1),
                    "cx": round(line_cx(fit, size, lx0, lx1, runs, cxa), 1),
                    "width": round(fit.line_width(runs, size), 1),
                    "runs": consolidate(runs),
                } for (ly, lx0, lx1, runs, cxa) in lines],
            }
            page_entries.append(entry)
            fitted.add(bi)
            report.append((path.stem[-5:], bi, oldh, newh,
                           newh / oldh if oldh else 0))
        layout[path.stem] = page_entries
        # clean ONLY bubbles that got a layout entry (a fit failure must
        # never blank a bubble)
        changed = False
        for bi, b, mask, allowed in clean_jobs:
            if bi in fitted:
                clean_bubble(img, b, mask, allowed)
                changed = True
        if changed:
            out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(path), out,
                        [cv2.IMWRITE_JPEG_QUALITY, 95,
                         int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                         cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])

    (WORK / "layout.json").write_text(json.dumps(layout, indent=1))
    gains = [g for *_, g in report if g]
    print("\nsize gain per bubble (new/old cap height):")
    for stem, bi, oh, nh, g in report:
        print(f"  {stem} b{bi:02d}: {oh:.0f}px -> {nh:.0f}px  x{g:.2f}")
    print(f"median gain: x{np.median(gains):.2f}, "
          f"min x{min(gains):.2f}, max x{max(gains):.2f}")


def consolidate(word_runs):
    """[(style,word)...] -> [[style, 'joined words']...] merging neighbors."""
    out = []
    for st, w in word_runs:
        if out and out[-1][0] == st:
            out[-1][1] += " " + w
        else:
            out.append([st, w])
    return out


if __name__ == "__main__":
    main()
