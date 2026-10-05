#!/usr/bin/env python3
"""Clean bubble interiors (wipe the old lettering, keep the fill: white or
stripes) and compute the new text layout: the largest font size whose
wrapped lines fit each bubble's inset row profile. Writes the cleaned
pages over pipeline/upscaled/... and a layout.json for the GIMP text step."""

import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageFont

import sys

REPO = Path(__file__).resolve().parent.parent
COMIC = BOOK_PIN = WORK = PAGES_DIR = BUB = None
TYPO_FIXES, LAYOUT_OVERRIDES = [], {}
# caption boxes found from their drawn frame (`auto_caption_boxes`), per
# page: {"<short> bNN": {"box": ...}} and {stem: entries inside a box}
AUTO_OVERRIDES, AUTO_MEMBERS = {}, {}
_DEBUG_LOBES = False


def configure(comic, book_pin=None):
    """Point the module at one comic: its paths and its per-comic editorial
    data (typo_fixes.json, layout_overrides.json).

    Called automatically at import when the comic is on argv — which is how
    the pipeline and every qa-tool invoke it — so nothing about the command
    line changes. Importing this as a LIBRARY (the unit tests, anything that
    only wants the algorithms) calls it explicitly instead of faking
    sys.argv, which is what every caller used to have to do.

    `book_pin`: bug-fix rounds re-fit against an already-approved book, and
    pinning the common size keeps every page that was signed off at the size
    it shipped with, so a re-fit only moves pages whose layout changed.
    """
    global COMIC, BOOK_PIN, WORK, PAGES_DIR, BUB, TYPO_FIXES, LAYOUT_OVERRIDES
    COMIC, BOOK_PIN = comic, book_pin
    WORK = REPO / "relettering" / comic
    PAGES_DIR = REPO / "upscaled" / comic
    BUB = WORK / "bubbles"
    _tf = WORK / "typo_fixes.json"
    TYPO_FIXES = [tuple(x) for x in json.loads(_tf.read_text())] \
        if _tf.is_file() else []
    # per-bubble editorial overrides (per-comic data, like typo_fixes):
    # relettering/<comic>/layout_overrides.json = {"1-012 b03": {"size": 17}}
    # pins that bubble's font size (bypassing group caps) — the escape hatch
    # for taste calls no global rule should be bent around
    _lo = WORK / "layout_overrides.json"
    LAYOUT_OVERRIDES = json.loads(_lo.read_text()) if _lo.is_file() else {}


# Auto-configure only when THIS is the program being run — directly, or by
# a tool that announced itself the way the pipeline does
# (sys.argv = ["reletter_fit.py", comic]). Without that guard any host with
# a positional argument configures a comic of its own name: under
# `python -m unittest discover tests`, argv[1] is "discover".
_argv0 = Path(sys.argv[0]).name if sys.argv else ""
if ((__name__ == "__main__" or _argv0.startswith("reletter_"))
        and len(sys.argv) > 1 and not sys.argv[1].startswith("-")):
    configure(sys.argv[1],
              int(sys.argv[sys.argv.index("--book") + 1])
              if "--book" in sys.argv else None)
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
MAX_PIECES = 3          # most fragments one hyphenated word may become
# Clearance between the new lettering and the balloon outline. Detection's
# row profile runs to the mask edge, and the mask includes the outline
# stroke, so a line could be set right onto the drawn line — with the new
# text larger than the original (it grows to the book size) that reads as
# "text outside the bubble". Applied to the FIT only: the CLEANING still
# works the full mask, or letters at the rim would survive.
TEXT_INSET = 5
TRACK_MAX_FRAC = 0.05   # tightest tracking, as a fraction of the em
FRAME_MAX = 90          # a drawn frame is dark in EVERY channel
# two strips of ONE caption box are cut from the same rectangle, so their x
# ranges agree this closely (intersection over union). Measured book-wide:
# the real pairs run 0.893-0.998, the false ones 0.716 and 0.769
X_AGREE = 0.85
# how far a pixel's colour may stray from the balloon's own fill and still
# count as that fill: the wall where no dark stroke separates balloon from page
COLOUR_WALL = 55
# a word shorter than this is never hyphenated: breaking it to buy a size or
# two reads as a typo, and hyphenation exists only so one LONG word cannot
# collapse a bubble
MIN_HYPHEN_LETTERS = 7
MIN_SIZE, MAX_SIZE = 10, 80


def parse_runs(text: str) -> list:
    """'A *B C* D' -> [('regular','A'), ('bolditalic','B C'), ...] word-level:
    [(style, word), ...]

    Style is resolved PER CHARACTER and only then split on whitespace. A
    regex alternation ('\\*([^*]+)\\*|(\\S+)') cannot do this: it is tried
    left to right at each position, so in "¡--*VENTANA*" the emphasis branch
    fails for want of a leading '*' and the greedy \\S+ branch swallows the
    token whole — the book printed a literal "¡--*VENTANA*".

    Runs are word level and the renderer sets a space between them, so one
    word cannot carry two styles: a token that is emphasised anywhere keeps
    its attached punctuation and takes the emphasis style."""
    marked, emph, i = [], False, 0
    while i < len(text):
        ch = text[i]
        if ch == "*" and (emph or text.find("*", i + 1) != -1):
            emph = not emph          # a lone, unmatched '*' stays literal
        else:
            marked.append((ch, emph))
        i += 1
    words, cur, cur_emph = [], [], False
    for ch, st in marked:
        if ch.isspace():
            if cur:
                words.append(("bolditalic" if cur_emph else "regular",
                              "".join(cur)))
                cur, cur_emph = [], False
        else:
            cur.append(ch)
            cur_emph = cur_emph or st
    if cur:
        words.append(("bolditalic" if cur_emph else "regular", "".join(cur)))
    return words


class Fitter:
    def __init__(self):
        self._cache = {}
        self._wcache = {}
        self._capr = None
        # current tracking (letter spacing) in px, <= 0. Set by the size
        # search and carried into every width it measures, so the wrap sees
        # exactly the line the GIMP text layer will render.
        self.track = 0.0

    def word_width(self, style, word, size):
        """Memoized single-word width, with the current tracking applied.
        hyphenate() measures every word on every wrap attempt, and a bubble
        makes thousands of those."""
        key = (style, word, size, self.track)
        w = self._wcache.get(key)
        if w is None:
            w = (self.font(style, size).getlength(word)
                 + self.track * max(0, len(word) - 1))
            self._wcache[key] = w
        return w

    def font(self, style, size):
        key = (style, size)
        if key not in self._cache:
            self._cache[key] = ImageFont.truetype(str(FONTS[style]), size)
        return self._cache[key]

    def line_width(self, runs, size):
        """Width of a laid-out line, tracking included.

        The runs are rendered as one GIMP text layer each, with the space
        between them added by the script, so tracking closes the gaps INSIDE
        a run and not the space between two styles — which is exactly what
        this sums."""
        space = self.font("regular", size).getlength(" ")
        return (sum(self.word_width(st, w, size) for st, w in runs)
                + space * (len(runs) - 1))

    def cap_ratio(self):
        """Cap ink height as a fraction of the em, measured from the font.

        Never hardcoded: the book is set in whatever font the user supplies,
        and a 0.86 that is right for one is wrong for the next.
        """
        if self._capr is None:
            from fontTools.pens.boundsPen import BoundsPen
            from fontTools.ttLib import TTFont
            t = TTFont(str(FONTS["regular"]))
            gs = t.getGlyphSet()
            pen = BoundsPen(gs)
            gs[t.getBestCmap()[ord("H")]].draw(pen)
            y0, y1 = pen.bounds[1], pen.bounds[3]
            self._capr = (y1 - y0) / float(t["head"].unitsPerEm)
        return self._capr

    def cap_height(self, size):
        """The height the cap glyph ACTUALLY renders at, `size` px of em.

        This is what `new_cap_h` reports and what `orig_px` divides by, and
        both are compared against `old_cap_h` — which is MEASURED off the
        original letterer's pixels. So it has to be a measured height too.

        It used to be PIL's `getbbox("H")`, which is the layout box from the
        ascender origin rounded out to whole pixels: at 23px it said 22
        where GIMP renders 19, and it moved in 3px jumps between adjacent
        sizes. Against the printed PDFs the ratio was 0.864 book-wide, all
        of it in the same direction — so "is the new text at least as big as
        the original?" could not fail, and passed on 56 of 56 sampled
        bubbles while 5 of them printed smaller than the original. Four
        rounds of "verified fixed" reached a reader who could still see it.
        """
        return round(self.cap_ratio() * size, 1)


def widest_run(row, min_frac: float = 0.25) -> tuple:
    """The usable span of a boolean row: first to last, MINUS any run too
    narrow to be part of the same shape.

    Low in a balloon the fill breaks into the body and the mouth of the tail,
    with drawn outline between them. A span taken plainly first-to-last spans
    that gap and the wrap sets its last lines out into the tail, hard against
    the outline (p127). But taking only the single widest run is wrong too:
    on a row crossing LETTERING the fill is broken into the gaps between
    words, and the widest of those is a few px — p101's caption then had a
    word gap for a profile and could not be typeset at all.

    Dropping runs narrower than `min_frac` of the widest separates the two:
    a tail mouth is a small fraction of the body, while the gaps a row of
    letters leaves are all of a size."""
    xs = np.flatnonzero(row)
    if not len(xs):
        return None
    cuts = np.flatnonzero(np.diff(xs) > 1)
    starts = np.concatenate(([0], cuts + 1))
    ends = np.concatenate((cuts, [len(xs) - 1]))
    widths = xs[ends] - xs[starts]
    keep = widths >= min_frac * widths.max()
    return int(xs[starts[keep][0]]), int(xs[ends[keep][-1]]) + 1


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


# Latin-script vowels (the pipeline is language-agnostic for LTR,
# space-separated, Latin-script-like languages), used only to PREFER an
# open-syllable break; any position is legal if none is available.
VOWELS = set("AEIOUYÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛÄËÏÖÜÃÕÅØÆ"
             "aeiouyáéíóúàèìòùâêîôûäëïöüãõåøæ")


def split_word(fit, style, word, size, limit):
    """Longest hyphenated head of `word` that fits `limit`, plus the rest.

    Returns None when no break leaves at least two characters on each side
    (breaking off a single letter reads as a typo, not a hyphenation), and
    for words too SHORT to be worth breaking: hyphenation is here so that one
    long word cannot collapse a bubble, and splitting a short one to buy a
    size or two reads as a typo instead — p88's two-word balloon came back as
    "SI / PUE- / DO..." at 18px, worse than the words whole at 16px."""
    if sum(c.isalpha() for c in word) < MIN_HYPHEN_LETTERS:
        return None
    fits = []
    for i in range(2, len(word) - 1):
        if fit.word_width(style, word[:i] + "-", size) > limit:
            break
        fits.append(i)
    if not fits:
        return None
    # break at an existing hyphen when there is one inside the head
    at_hyphen = [i for i in fits if word[i - 1] == "-"]
    if at_hyphen:
        return word[:at_hyphen[-1]], word[at_hyphen[-1]:]
    # otherwise prefer vowel|consonant — the open-syllable break that reads
    # acceptably across Latin-script languages
    good = [i for i in fits
            if word[i - 1] in VOWELS and word[i] not in VOWELS]
    i = good[-1] if good else fits[-1]
    return word[:i] + "-", word[i:]


def hyphenate(fit, words, size, limit, hard_breaks, para_breaks):
    """Break any word too wide for the widest line the mask offers.

    Words used to be unbreakable, so one over-long word made every wrap at
    that size fail and the size search walked all the way down until it
    fitted on one line — a single long shout dropped its balloon far below
    the book size while its neighbours stayed at full size. Returns
    (words, hard_breaks, para_breaks) with a forced break after each piece.
    """
    if limit <= 0:
        return words, hard_breaks, para_breaks
    out, nhard, npara = [], set(), set()
    changed = False
    for i, (style, word) in enumerate(words):
        pieces = [(style, word)]
        while fit.word_width(pieces[-1][0], pieces[-1][1], size) > limit:
            if len(pieces) >= MAX_PIECES:
                # this size is hopeless for this word — hand the caller the
                # ORIGINAL words so the wrap fails fast and the size search
                # steps down, instead of shattering every word into
                # fragments and running the line-break DP over them
                return words, hard_breaks, para_breaks
            sp = split_word(fit, pieces[-1][0], pieces[-1][1], size, limit)
            if sp is None:
                break
            head, tail = sp
            pieces[-1] = (style, head)
            pieces.append((style, tail))
            changed = True
        for k, piece in enumerate(pieces):
            out.append(piece)
            if k < len(pieces) - 1:
                nhard.add(len(out) - 1)
        if i in hard_breaks:
            nhard.add(len(out) - 1)
        if i in para_breaks:
            npara.add(len(out) - 1)
    if not changed:
        return words, hard_breaks, para_breaks
    return out, nhard, npara


def wrap_at(fit, words, rows, size, cy, hard_breaks, factor=0.96,
            para_breaks=(), anchor=None, lift=0.0, hold_cy=False,
            tight=False):
    """Try to wrap words into lines centered vertically on cy. Returns
    [(y_top, x0, x1, line_runs)] or None. hard_breaks = word indices after
    which a line break is forced; para_breaks additionally add a visual gap
    (compound balloons hold several paragraphs)."""
    lh = max(1, round(size * LINE_SPACING))
    gap = round(0.55 * lh)
    # a word wider than the widest line this mask can offer fits nowhere:
    # hyphenate it rather than let the size search shrink the whole bubble
    if rows:
        if anchor is None:
            limit = max(x1 - x0 for x0, x1 in rows.values()) * factor
        else:
            limit = max(2.0 * min(anchor - x0, x1 - anchor)
                        for x0, x1 in rows.values()) * factor
        words, hard_breaks, para_breaks = hyphenate(
            fit, words, size, limit, set(hard_breaks), set(para_breaks))
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
            # the first/last line may overhang the mask a little — an arc
            # forgives that. `tight` refuses it: a drawn caption FRAME
            # forgives nothing (the fill IS the writing area), and on p214's
            # 47px of box interior the allowance let two 24px line boxes
            # "fit" and the type printed across the top and bottom rules.
            over = 0.0 if tight else 0.16 * lh
            v0 = y0 + (over if li == 0 else 0)
            v1 = y0 + lh - (over if li == n - 1 else 0)
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


def find_bands(rows: dict, nbands: int):
    """Split an n-lobe row profile into `nbands` bands, deepest waist first.

    A compound balloon carries one paragraph per lobe. Only the two-lobe
    case was handled, so a three-lobe balloon fell straight through to a
    whole-mask wrap: all three paragraphs were crammed into the union and
    the size search collapsed to fit them — one lobe typeset tiny and the
    other two left empty. Returns a top-to-bottom list, or None when the
    profile has no run of real waists."""
    if nbands <= 1:
        return [rows]
    bands = [rows]
    while len(bands) < nbands:
        # always split the tallest band that still has a real waist
        target, best = None, -1
        for i, band in enumerate(bands):
            if len(band) > best and find_waist(band) is not None:
                target, best = i, len(band)
        if target is None:
            return None
        bands[target:target + 1] = list(find_waist(bands[target]))
    return bands


def mask_lobes(mask, ox, oy, nlobes):
    """Decompose a compound balloon into `nlobes` lobes in 2D.

    find_bands cuts the row-width profile, which only works when the lobes
    are stacked: a row span is one x-range, so lobes sitting side by side
    (a diagonal run of balloons) collapse into a single wide span and no
    waist is visible. Here the lobe centres are the peaks of the distance
    transform and every mask pixel joins its nearest peak. Returns a list
    of page-coordinate rows dicts in reading order, or None."""
    if nlobes < 2:
        return None
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return None
    for frac in (0.55, 0.5, 0.45, 0.6, 0.4, 0.65):
        seeds = (dist > frac * peak).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(seeds, 8)
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 200]
        if len(keep) != nlobes:
            continue
        sel = np.zeros_like(seeds)
        for i in keep:
            sel[lab == i] = 1
        # nearest-seed assignment: distances are measured to the ZEROS, so
        # invert and let each pixel inherit its nearest seed's component
        _, near = cv2.distanceTransformWithLabels(
            (sel == 0).astype(np.uint8), cv2.DIST_L2, 5,
            labelType=cv2.DIST_LABEL_CCOMP)
        out = []
        for k in sorted(set(near[sel > 0].tolist())):
            lobe = (near == k) & (mask > 0)
            rows = {}
            for ry in range(lobe.shape[0]):
                xs = np.flatnonzero(lobe[ry])
                if len(xs):
                    rows[oy + ry] = (ox + int(xs[0]), ox + int(xs[-1]) + 1)
            if len(rows) < 12:
                break
            out.append(rows)
        if len(out) == nlobes:
            # reading order: left to right where two lobes share most of
            # their height, top to bottom otherwise (reading_order)
            ex = [(min(v[0] for v in r.values()), min(r),
                   max(v[1] for v in r.values()), max(r)) for r in out]
            return [out[i] for i in reading_order(ex)]
    return None


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


def block_anchor(block, rows, bbox=None):
    """Centre of the letterer's text block, clipped to the balloon.

    The block comes from clustering letter blobs, and a stroke of artwork
    beside the balloon (fur, a panel line, a highlight) joins the cluster
    now and then. Since the block centre is THE primary anchor, an
    over-wide block slid the new text clean out of the lobe. Clip it to
    what the balloon actually offers — its row spans, and its detected box
    where the mask itself leaked sideways — while leaving an honest block
    exactly where it was."""
    bx, by, bw, bh = block
    span = [rows[y] for y in range(int(by), int(by + bh)) if y in rows]
    if not span:
        span = list(rows.values())
    if not span:
        return bx + bw / 2.0, by + bh / 2.0
    lo = min(x0 for x0, _ in span)
    hi = max(x1 for _, x1 in span)
    if bbox is not None:
        lo, hi = max(lo, bbox[0]), min(hi, bbox[0] + bbox[2])
        if hi <= lo:                   # mask and box disagree — trust rows
            lo = min(x0 for x0, _ in span)
            hi = max(x1 for _, x1 in span)
    x0, x1 = max(bx, lo), min(bx + bw, hi)
    if x1 <= x0:                       # block misses the balloon entirely
        x0, x1 = lo, hi
    return (x0 + x1) / 2.0, by + bh / 2.0


def lobe_bands(p):
    """The per-paragraph bands fit_lines wraps into, or None.

    A compound balloon carries ONE PARAGRAPH PER LOBE: stacked lobes are cut
    from the row-width profile's waist, lobes side by side have no waist and
    come from the distance-transform peaks instead.
    """
    nparas = len(p["paras"]) + 1
    if p["b"]["kind"] != "bubble" or nparas < 2:
        return None, False
    # balloons found from their own lettering already ARE the lobes, in
    # reading order; a waist cut of their union would re-order them by row
    if p.get("auto_lobes") and p.get("lobes") and len(p["lobes"]) == nparas:
        return p["lobes"], True
    bands = find_bands(p["rows"], nparas)
    if bands is not None:
        return bands, False
    lobes = p.get("lobes")
    if lobes and len(lobes) == nparas:
        return lobes, True
    return None, False


def fit_one_lobe(fit, p, bands, k, size):
    """Wrap paragraph k inside its own lobe, centered on that lobe.

    Split out of fit_lines so a lobe can also be sized ON ITS OWN: three
    balloons that detection returned as a single entry shared one size, set
    by the tightest of them, and the roomy ones then did not fill their own
    balloon (p123, reported three rounds running).
    """
    pis = sorted(p["paras"])
    bounds = [0] + [i + 1 for i in pis] + [len(p["words"])]
    a, z = bounds[k], bounds[k + 1]
    seg = p["words"][a:z]
    hardk = {hh - a for hh in p["hard"] if a <= hh < z - 1}
    lift = optical_lift(size)
    ax, ay = visual_center(bands[k])
    lk = wrap_at(fit, seg, bands[k], size, ay - lift, hardk,
                 anchor=ax, lift=lift)
    if lk is None or not block_centered(lk, bands[k], size, ay - lift):
        return None
    return [(y0, x0, x1, r, ax) for (y0, x0, x1, r) in lk]


def grow_lobes(fit, p, size, cap):
    """Raise each lobe to the largest size IT can host, never below `size`.

    Returns [(size, track, lines)] per lobe, or None when the bubble is not
    banded or nothing grew. The user's call (Sep 16 2026): each lobe fills
    its own balloon, even where that means two sizes in one joined shape.
    """
    bands, _ = lobe_bands(p)
    if bands is None:
        return None
    out = []
    for k in range(len(bands)):
        best = None
        for s2 in range(cap, size - 1, -1):
            for tr2 in track_steps(s2):
                fit.track = tr2
                lk = fit_one_lobe(fit, p, bands, k, s2)
                if lk:
                    best = (s2, tr2, lk)
                    break
            if best:
                break
        fit.track = 0.0
        if best is None:
            return None
        out.append(best)
    return out if any(g[0] > size for g in out) else None


def fit_lines(fit, p, size):
    """Wrap p's words at `size`. Two-paragraph bubbles whose mask has a real
    waist get each paragraph wrapped INSIDE its own lobe, centered on that
    lobe's centroid — a whole-mask wrap centers the block on the union
    centroid and lets lines drift toward the waist. Returns
    [(y_top, x0, x1, runs, cx_anchor)] or None; cx_anchor is the lobe's
    width-weighted x-center each line should center on (clamped to its own
    row span when building the layout entry)."""
    rows = p["rows"]
    nparas = len(p["paras"]) + 1
    if p["b"]["kind"] == "bubble" and nparas >= 2:
        bands, disjoint = lobe_bands(p)
        if bands is not None:
            # one paragraph per lobe, each wrapped INSIDE its own lobe and
            # centered on it — a whole-mask wrap centers the block on the
            # union centroid and lets lines drift toward the waist
            out, ok = [], True
            for k in range(nparas):
                lk = fit_one_lobe(fit, p, bands, k, size)
                if lk is None:
                    ok = False
                    break
                out += lk
            if ok:
                return out
            if disjoint:
                # separate balloons sharing one detection entry: a
                # whole-mask wrap runs lines straight across the gap
                # between them and over both outlines. Step the size down
                # instead — a smaller size inside the right balloon always
                # beats a bigger one spanning two.
                return None
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
        cxa, cya = block_anchor(p["b"]["block"], rows, p["b"]["bbox"])
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
            # FIT WINS OVER SIZE (the user's call, round 7). This rung
            # used to pack the way the original letterer did — near-RAW mask
            # rows, and three quarters of a line height of slack below the
            # original's own text — which is how it recovered 2-4px in a
            # dense balloon. But our font is ~1.5x wider per unit of cap
            # height than the original edition's, so at the book size a
            # dense balloon needs an extra LINE, and that slack is what let
            # the extra line land on the bottom outline: p211's "MUERTOS!"
            # and p294's "MUERTO." crossed it, p286's block sat a line low.
            # The block must stay inside the original's own envelope and
            # keep the outline clearance instead, and a balloon that cannot
            # hold the size drops a step — which is what the user chose.
            src = rows
            rows2 = {yy: sp for yy, sp in src.items()
                     if by2 - 0.25 * lh <= yy <= by2 + bh2 + 0.25 * lh}
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
    # HOLD the target, as every balloon anchor does. Without it wrap_at
    # re-balances and centres the block geometrically in the rows, which
    # threw away the letterer's own centre we just measured: p42's caption
    # was set at 175.5 (its box's middle) where the letterer had it at 181.5.
    # a drawn caption FRAME has no slack — the line boxes must fit inside it,
    # overhang allowance and all (p214's box is 47px of interior and two
    # 24px line boxes printed across its rules)
    tight = bool(p.get("in_box"))
    lines = wrap_at(fit, p["words"], rows, size, p["cys"][0], p["hard"],
                    para_breaks=p["paras"], hold_cy=True, tight=tight)
    if lines is None:
        # the letterer's centre cannot host this size in this box — fall back
        # to the geometric centre rather than losing the bubble entirely
        lines = wrap_at(fit, p["words"], rows, size, p["cys"][0], p["hard"],
                        para_breaks=p["paras"], tight=tight)
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


def track_steps(size):
    """Tracking levels to try for a size, loosest (none) first.

    Capped at 5% of the em — measured in GIMP, that is about an 8% narrower
    line on a typical word, which buys one to two size steps in a balloon
    the type only just misses. Proportional to the font, never a fixed
    pixel count: the book may be re-set in any font the user supplies.
    """
    worst = -TRACK_MAX_FRAC * size
    return [0.0, round(worst / 3.0, 3), round(2 * worst / 3.0, 3),
            round(worst, 3)]


def lobe_size_caps(maxes, origs, book):
    """Size cap per lobe of a joined balloon — each fills its OWN balloon.

    These lobes used to share the group's smallest maximum, because the
    original letterer used one size per joined pair. That was faithful while
    the original ran uniformly at ~16px cap down a whole page; ours runs at
    the book size, so the pair took the tightest lobe's size and sat at 18px
    next to neighbours at 23px. The user reported exactly that on p24, p79
    and p101 in three consecutive rounds and chose (Sep 16 2026) to let each
    lobe fill its own balloon, accepting two sizes in one joined shape.

    maxes/origs are keyed by bubble index: the lobe's own maximum, and the
    original's own size there in our font's px.
    """
    return {i: min(maxes[i], max(book, origs.get(i, 0))) for i in maxes}


def inset_rows(rows: dict, inset: int = TEXT_INSET) -> dict:
    """Pull a row profile in from the balloon edge by `inset` px each way.

    Keeps the new lettering clear of the drawn outline. Rows that become
    too narrow are dropped, and the top/bottom rows go too, so the first
    and last line cannot ride the arc. The clearance scales with the
    balloon: a flat 5px is a big bite out of a small one and cost it a
    whole size step."""
    if inset <= 0 or not rows:
        return rows
    ys = sorted(rows)
    extent = min(ys[-1] - ys[0] + 1,
                 max(x1 - x0 for x0, x1 in rows.values()))
    inset = max(2, min(inset, int(round(0.035 * extent))))
    lo, hi = ys[0] + inset, ys[-1] - inset
    out = {}
    for y in ys:
        if y < lo or y > hi:
            continue
        x0, x1 = rows[y]
        if x1 - x0 > 2 * inset + 8:
            out[y] = (x0 + inset, x1 - inset)
    return out or rows


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


def fill_holes(m):
    """Boolean mask with its enclosed holes filled.

    Floods the background inward from OUTSIDE a 1px pad: OpenCV treats
    out-of-border as foreground, so an unpadded flood would leave any
    region that touches the crop edge unreached and report it as a hole."""
    pad = np.pad(m.astype(np.uint8), 1)
    ff = pad.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8),
                  (0, 0), 1)
    return ((pad > 0) | (ff == 0))[1:-1, 1:-1]


def bubble_interior(dark, keep_ring, block_box, img=None):
    """The balloon's own fill: the light region inside the mask that holds
    the letterer's text block, walled in by the balloon outline.

    Detection sometimes hands back a mask that spills past the outline — a
    tail that leaves the crop window floods the white page around it, and a
    balloon on a light background (pale sky) has nothing to stop the fill.
    Everything dark in the mask then looked like old lettering, so the
    repaint erased the outline itself and the balloon vanished from the
    page. Anchoring on the text block keeps us inside the real balloon.

    Dark strokes are not the only wall. Where the outline has a gap — a tail
    that leaves the crop, a stroke the mask clipped — the fill escapes onto
    the page even though the page is plainly a different COLOUR (white fill
    against pale sky). Clipping the fit's row profile to this interior does
    nothing when the interior has leaked too, so pass `img` and the flood
    also stops where the colour leaves the fill's own."""
    def _grow(extra=None):
        light = ((dark == 0) & (keep_ring > 0))
        if extra is not None:
            light &= extra
        _, lab = cv2.connectedComponents(light.astype(np.uint8), 8)
        sx0, sy0, sx1, sy1 = block_box
        if sx1 > sx0 and sy1 > sy0:
            sub = lab[sy0:sy1, sx0:sx1]
            vals, counts = np.unique(sub[sub > 0], return_counts=True)
            if len(vals):
                # the component the text block sits in, not merely the
                # largest: past a leak the surrounding page is bigger
                return lab == vals[counts.argmax()]
        return None
    first = _grow()
    if first is None:
        return keep_ring > 0
    if img is None:
        return first
    sx0, sy0, sx1, sy1 = block_box
    inblock = np.zeros(first.shape, bool)
    inblock[max(0, sy0):sy1, max(0, sx0):sx1] = True
    sample = first & inblock
    if sample.sum() < 50:
        return first
    fill = np.median(img[sample], axis=0)
    # a colour SHIFT, not a brightness one: the old lettering's own grey rims
    # are the fill's hue, only darker, and an absolute channel difference
    # carves them out of the interior — everything gated on the fill then
    # skips them and they survive as hollow ghost letters. Dropping the
    # common luminance offset leaves only the hue change that marks the page.
    d = img.astype(np.int16) - fill.astype(np.int16)
    near = (d.max(axis=2) - d.min(axis=2)) <= COLOUR_WALL
    walled = _grow(near)
    return first if walled is None else walled


def lobes_from_rects(img, rects, bbox):
    """A bubble's mask and lobes rebuilt from hand-measured rectangles.

    The escape hatch for a balloon whose detection leaked past all
    recognition. p276 b04 is TWO balloons drawn on pale sky, returned as one
    1269x528 entry whose mask covers 79% of the panel: nothing downstream
    can find either balloon — mask_lobes sees a single blob, the fit places
    nothing, and cleaning it would repaint the sky — and splitting the entry
    in detection would renumber the page and invalidate its positional
    transcripts. A `lobes` override in layout_overrides.json names one
    rectangle per balloon and each balloon's own interior is flooded inside
    its rectangle with the walled flood used everywhere else, so the mask
    becomes the balloons and nothing else.

    `rects` = [[x0, y0, x1, y1], ...] in page coords, one per lobe, IN
    PARAGRAPH ORDER, each roughly centred on its balloon (the middle of the
    rectangle seeds the flood). Returns (mask in the bbox frame, lobes as
    page-coordinate row dicts, per-lobe CLEANING jobs) or None if any
    rectangle finds no plausible interior — a rectangle that misses is
    better ignored than trusted.

    The cleaning jobs are per lobe — a `(bbox, block, mask)` triple each —
    because the cleaner reads the letterer's block and this entry's spans
    every balloon and the artwork between them: cleaned as one, p276's
    strict ink term reached 80px past the left balloon and repainted the
    green sphere and the darts behind it. They are two balloons; clean them
    as two. The block per lobe is the original lettering's own ink inside
    that interior, which is what a block is everywhere else.
    """
    x, y, w, h = bbox
    mask = np.zeros((h, w), np.uint8)
    lobes, jobs = [], []
    for rect in rects:
        rx0, ry0, rx1, ry1 = (int(v) for v in rect)
        rx0, ry0 = max(0, rx0), max(0, ry0)
        rx1, ry1 = min(img.shape[1], rx1), min(img.shape[0], ry1)
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < 40 or rh < 40:
            return None
        reg = img[ry0:ry1, rx0:rx1]
        dark = (reg.min(axis=2) <= DARK_MAX - 20).astype(np.uint8)
        # the middle of the rectangle stands in for the letterer's block:
        # a rectangle measured around a balloon has its interior at its
        # centre, and the flood needs no more than that to start
        seed = (rw // 4, rh // 4, rw - rw // 4, rh - rh // 4)
        inter = fill_holes(bubble_interior(
            dark, np.ones(dark.shape, np.uint8), seed, img=reg))
        if not (0.10 <= inter.mean() <= 0.92):
            return None             # the flood escaped, or found nothing
        got = interior_lobe(reg, inter, rx0, ry0, bbox, mask)
        if got is None:
            return None
        lobes.append(got[0])
        jobs.append(got[1])
    if len(lobes) != len(rects):
        return None
    return mask, lobes, jobs


def _box_gap(a, c):
    dx = max(0, max(a[0], c[0]) - min(a[0] + a[2], c[0] + c[2]))
    dy = max(0, max(a[1], c[1]) - min(a[1] + a[3], c[1] + c[3]))
    return (dx * dx + dy * dy) ** 0.5


def letter_groups(boxes, n):
    """Split letter boxes into `n` paragraphs at the widest gaps.

    A minimum spanning tree over the gaps between letters, cut at its
    longest edges — but only where both sides keep real lettering: a lone
    speck of ink far from the text is the longest edge of all, and cutting
    it off "splits" 2-030-2's two balloons into the whole text and one dot.
    Specks then join the nearest group. Returns n lists of boxes, or None."""
    m = len(boxes)
    minsz = max(4, int(0.08 * m))
    if m < n * minsz:
        return None
    edges = sorted((_box_gap(boxes[i], boxes[j]), i, j)
                   for i in range(m) for j in range(i + 1, m))
    par = list(range(m))

    def find(i):
        while par[i] != i:
            par[i] = par[par[i]]
            i = par[i]
        return i
    adj = {i: set() for i in range(m)}
    mst = []
    for g, i, j in edges:
        a, c = find(i), find(j)
        if a != c:
            par[a] = c
            mst.append((g, i, j))
            adj[i].add(j)
            adj[j].add(i)

    def comp(s0):
        seen, st = {s0}, [s0]
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    st.append(v)
        return seen
    for _ in range(n - 1):
        for g, i, j in sorted(mst, reverse=True):
            if j not in adj[i]:
                continue
            adj[i].discard(j)
            adj[j].discard(i)
            if len(comp(i)) >= minsz and len(comp(j)) >= minsz:
                break
            adj[i].add(j)
            adj[j].add(i)
        else:
            return None
    groups, seen = [], set()
    for i in range(m):
        if i not in seen:
            c = comp(i)
            seen |= c
            groups.append([boxes[t] for t in sorted(c)])
    big = [g for g in groups if len(g) >= minsz]
    if len(big) != n:
        return None
    for g in groups:
        if len(g) < minsz:
            for bb in g:
                min(big, key=lambda G: min(_box_gap(bb, o)
                                           for o in G)).append(bb)
    return big


def _extent(g):
    return (min(b[0] for b in g), min(b[1] for b in g),
            max(b[0] + b[2] for b in g), max(b[1] + b[3] for b in g))


def side_by_side(a, c):
    """Two text extents (x0, y0, x1, y1) that share most of their height —
    read left to right, not top to bottom."""
    ov = min(a[3], c[3]) - max(a[1], c[1])
    return ov > 0.3 * min(a[3] - a[1], c[3] - c[1])


def reading_order(exts):
    """Indices of text extents in reading order: left before right where
    two sit side by side, top before bottom otherwise.

    Sorting on the top row first (mask_lobes used to) swaps two balloons
    side by side whenever the right one's top is a few px higher: p2-042,
    p2-052, p3-031 and p3-032 all printed each balloon's text in the
    other."""
    import functools

    def cmp(i, j):
        a, c = exts[i], exts[j]
        if side_by_side(a, c):
            return -1 if a[0] < c[0] else 1
        return -1 if a[1] < c[1] else 1
    return sorted(range(len(exts)), key=functools.cmp_to_key(cmp))


def block_letters(img, block, within=None, origin=(0, 0)):
    """Letter-sized ink pieces lying WHOLLY inside `block`, as page boxes.

    Wholly inside: clipped to the block, a balloon's outline breaks into
    short arcs that pass for letters, and 2-030-2's block corner took one —
    the cleaner then repainted the outline as a white tab. `within` (in the
    frame starting at `origin`) bounds the ink to a mask."""
    bx, by, bw, bh = block
    X0, Y0 = max(0, bx - 60), max(0, by - 60)
    reg = img[Y0:by + bh + 60, X0:bx + bw + 60]
    ink = reg.min(axis=2) <= DARK_MAX
    if within is not None:
        ox, oy = origin
        m = np.zeros(ink.shape, bool)
        ys0, ys1 = max(Y0, oy), min(Y0 + ink.shape[0], oy + within.shape[0])
        xs0, xs1 = max(X0, ox), min(X0 + ink.shape[1], ox + within.shape[1])
        if ys1 > ys0 and xs1 > xs0:
            m[ys0 - Y0:ys1 - Y0, xs0 - X0:xs1 - X0] = \
                within[ys0 - oy:ys1 - oy, xs0 - ox:xs1 - ox]
        ink &= m
    k, _, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
    return [(X0 + int(st[i, 0]), Y0 + int(st[i, 1]),
             int(st[i, 2]), int(st[i, 3])) for i in range(1, k)
            if 9 <= st[i, 3] <= 48 and st[i, 2] <= 130 and st[i, 4] >= 8
            and X0 + st[i, 0] >= bx - 2 and Y0 + st[i, 1] >= by - 2
            and X0 + st[i, 0] + st[i, 2] <= bx + bw + 2
            and Y0 + st[i, 1] + st[i, 3] <= by + bh + 2]


def geodesic_owner(region, seeds, nlab):
    """Label every pixel of `region` with the seed label nearest to it
    WITHIN the region: the labels grow outward from their seeds one ring
    at a time, never leaving the region, so two balloons joined by a neck
    meet at the neck instead of along a straight line across one of them.
    Pixels no seed can reach stay 0."""
    owner = np.where(region, seeds, 0).astype(np.int32)
    k3 = np.ones((3, 3), np.uint8)
    reg = region.astype(bool)
    while True:
        free = reg & (owner == 0)
        if not free.any():
            break
        grew = False
        for t in range(1, nlab + 1):
            g = cv2.dilate((owner == t).astype(np.uint8), k3) > 0
            g &= free
            if g.any():
                owner[g] = t
                free &= ~g
                grew = True
        if not grew:
            break
    return owner


def letter_lobes(img, b, mask, n, others=(), paras=None):
    """The balloon(s) of an entry, found from the ORIGINAL LETTERING.

    Detection returns two balloons side by side as one entry, and from the
    mask alone the fit cannot always tell them apart: the distance-
    transform peaks miss a balloon much smaller than its neighbour (4 of
    11 such entries in book 3 did not split at all and printed one block of
    text across both), the lobes came back in the wrong order whenever the
    right balloon's top sat a few px higher, and a mask that had leaked
    across the panel (2-030-2) was cleaned as one region — both outlines
    and the panel's colour went with it. And an entry can lose half its
    own balloon to a neighbour it is joined to (4-024 b05: a rectangle over
    the last two lines, the first line left standing under the new text).

    The letterer says which balloon is which. His letters split at the
    widest gaps into one group per paragraph (`letter_groups`), the groups
    read in comic order (`reading_order`), and each group's balloon is the
    walled flood out from its own letters — the same flood the `lobes`
    override seeds from a hand-measured rectangle. Where floods meet —
    balloons joined with no outline between them, this entry's or a
    neighbour's (`others`, their blocks) — every pixel goes to the
    lettering nearest it, detection's seam_split rule.

    Used only where the existing path demonstrably fails: paragraphs side
    by side, or a detection mask well past (leaked) or well under
    (truncated) the balloons the lettering is in. Stacked lobes are found
    reliably from the waist of the row profile, and the books already
    shipped were approved on that. Returns what lobes_from_rects returns,
    or None."""
    x, y, w, h = b["bbox"]
    H, W = img.shape[:2]
    bx, by, bw, bh = b["block"]
    mfill = fill_holes(mask > 0)
    boxes = block_letters(img, b["block"], mfill, (x, y))
    groups = letter_groups(boxes, n)
    if groups is None:
        return None
    exts = [_extent(g) for g in groups]
    beside = any(side_by_side(exts[i], exts[j])
                 for i in range(n) for j in range(i + 1, n))
    groups = [groups[i] for i in reading_order(exts)]
    exts = [_extent(g) for g in groups]
    # each group must be about as long as the paragraph it will carry: a
    # transcript whose paragraphs are not this balloon's (1-100 b03 repeats
    # its neighbour's "ADELANTE:") would otherwise split one text in two
    if paras is not None and len(paras) == n and n > 1:
        dens = [len(g) / max(1, len(re.sub(r"[\s*]", "", t)))
                for g, t in zip(groups, paras)]
        if max(dens) > 2.5 * min(dens):
            return None
    # the flood window: the entry AND its block (4-024 b05's block starts
    # 35px above a mask that lost its first line), and any neighbour whose
    # block reaches into it, plus room for a balloon the mask cut (a
    # balloon's arc can rise 30px past its first line: 4-024 b05's did)
    P = 60
    X0, Y0 = max(0, min(x, bx) - P), max(0, min(y, by) - P)
    X1 = min(W, max(x + w, bx + bw) + P)
    Y1 = min(H, max(y + h, by + bh) + P)
    rivals = [o for o in others
              if o[0] < X1 and o[0] + o[2] > X0
              and o[1] < Y1 and o[1] + o[3] > Y0]
    for (ox, oy, ow, oh) in rivals:
        X0, Y0 = max(0, min(X0, ox - P)), max(0, min(Y0, oy - P))
        X1, Y1 = min(W, max(X1, ox + ow + P)), min(H, max(Y1, oy + oh + P))
    win = img[Y0:Y1, X0:X1]
    dark = (win.min(axis=2) <= DARK_MAX - 20).astype(np.uint8)
    floods = []
    for (ex0, ey0, ex1, ey1) in exts:
        f = fill_holes(bubble_interior(
            dark, np.ones(dark.shape, np.uint8),
            (ex0 - X0, ey0 - Y0, ex1 - X0, ey1 - Y0), img=win))
        if not f.any():
            return None
        floods.append(f)
    # the flood's wall is the raw ink, so a letter within a few px of the
    # outline is welded to it and left OUT of the interior — and the
    # cleaner is confined to the mask, so 2-030-2's first and last lines
    # survived as fragments under the new text. Each lobe owns its
    # letters, bounded by what detection's mask holds.
    mwin = np.zeros(dark.shape, bool)
    mwin[y - Y0:y - Y0 + h, x - X0:x - X0 + w] = mfill
    for f, g in zip(floods, groups):
        for (lx, ly, lw, lh) in g:
            sl = (slice(max(0, ly - Y0 - 2), ly - Y0 + lh + 2),
                  slice(max(0, lx - X0 - 2), lx - X0 + lw + 2))
            f[sl] |= mwin[sl]
    # where floods meet, the nearest LETTERING owns the pixel — nearest
    # INSIDE the fill (geodesic_owner): straight-line distance gave the big
    # balloon's top corner to the small one's text beside it (4-024), and
    # a watershed over the distance-to-edge handed the strip under
    # 2-030-2's left text to the right balloon. ANY overlap counts — a
    # flood the colour wall stopped partway into the next balloon shares
    # only a sliver, and that sliver (the neighbour's first line) went to
    # the wrong lobe and was never cleaned.
    uni = np.zeros(dark.shape, bool)
    for f in floods:
        uni |= f
    seeds = np.zeros(dark.shape, np.int32)
    for t, g in enumerate(groups, 1):
        for (lx, ly, lw, lh) in g:
            seeds[max(0, ly - Y0):ly - Y0 + lh,
                  max(0, lx - X0):lx - X0 + lw] = t
    for (ox, oy, ow, oh) in rivals:
        for (lx, ly, lw, lh) in block_letters(img, (ox, oy, ow, oh)):
            if uni[min(dark.shape[0] - 1, max(0, ly - Y0 + lh // 2)),
                   min(dark.shape[1] - 1, max(0, lx - X0 + lw // 2))]:
                seeds[max(0, ly - Y0):ly - Y0 + lh,
                      max(0, lx - X0):lx - X0 + lw] = n + 1
    if n == 1 and not (seeds == 2).any():
        inters = floods             # one balloon, no one else's letters
    else:
        owner = geodesic_owner(uni, seeds, n + 1)
        inters = [f & (owner == t) for t, f in enumerate(floods, 1)]
    found = np.zeros(dark.shape, bool)
    for inter in inters:
        ys, xs = np.nonzero(inter)
        if not len(ys):
            return None
        if ((xs.min() == 0 and X0 > 0) or (ys.min() == 0 and Y0 > 0)
                or (xs.max() == inter.shape[1] - 1 and X1 < W)
                or (ys.max() == inter.shape[0] - 1 and Y1 < H)):
            return None             # escaped: this is no balloon
        found |= inter
    # stacked lobes keep the waist cut — unless detection's mask LEAKED
    # well past the balloons the lettering is in (2-030-2's covered the
    # whole panel, and cleaning it painted out both outlines and the
    # panel's colour), or covers well UNDER them (4-024 b05's)
    inside = found[y - Y0:y - Y0 + h, x - X0:x - X0 + w].sum()
    # (for a COMPOUND entry only: a single balloon's mask commonly runs
    # 1.6-2.9x its walled interior and the ordinary path copes — re-finding
    # those moved 14 approved balloons in book 2 for no gain)
    leaked = n >= 2 and mfill.sum() > 1.4 * max(1, inside)
    # ...and a mask that covers too little only counts when it cost the
    # entry its OWN LETTERS (4-024 b05's first line lay outside it): a mask
    # that merely stops short of the outline already holds the text, and
    # re-finding those balloons moved approved type (6-133 b13 crowded the
    # joined balloon below)
    allb = block_letters(img, b["block"])
    lost = sum(1 for (lx, ly, lw, lh) in allb
               if not (0 <= ly + lh // 2 - y < h and 0 <= lx + lw // 2 - x < w
                       and mfill[ly + lh // 2 - y, lx + lw // 2 - x]))
    short = ((found & mwin).sum() < 0.6 * found.sum()
             and lost >= max(3, 0.15 * len(allb)))
    if _DEBUG_LOBES:
        print(f"    letter_lobes n={n} beside={beside} "
              f"mask/found={mfill.sum() / max(1, inside):.2f} "
              f"lost={lost}/{len(allb)} short={short}")
    if not (beside or leaked or short):
        return None
    out = np.zeros((h, w), np.uint8)
    lobes, jobs = [], []
    for inter in inters:
        got = interior_lobe(win, inter, X0, Y0, b["bbox"], out)
        if got is None:
            return None
        lobes.append(got[0])
        jobs.append(got[1])
    return out, lobes, jobs


def gate_mask(img, stem, bi, b, default, pristine=None, texts=None,
              bubbles=None):
    """The mask a verification gate must measure for entry `bi`, in its
    bbox frame: the region the fit actually used, not detection's PNG.

    A `lobes` override and the balloons `letter_lobes` re-finds both
    REPLACE detection's mask for the fit and the cleaner, and a gate that
    reads the PNG measures a region nothing ever cleaned (p276 b04's covers
    79% of a panel; letter_mask found 16 letter-shaped pieces of artwork in
    it). The one home for that rule — leftover, welded and faint_scan all
    call it. `pristine`, `texts` and `bubbles` let it re-find the balloons;
    without them only the override is applied."""
    ov = override_for(stem, bi)
    if ov and "lobes" in ov:
        got = lobes_from_rects(img, ov["lobes"], b["bbox"])
        if got is not None:
            return got[0].astype(bool)
    if (pristine is not None and texts is not None and bubbles is not None
            and b["kind"] == "bubble"):
        npar = apply_fixes(texts[bi - 1]).count("\n\n") + 1
        got = letter_lobes(pristine, b, default.astype(np.uint8), npar,
                           [o["block"] for j, o in enumerate(bubbles, 1)
                            if j != bi],
                           apply_fixes(texts[bi - 1]).split("\n\n"))
        if got is not None:
            return got[0].astype(bool)
    return default


def interior_lobe(reg, inter, rx0, ry0, bbox, mask):
    """One balloon's interior (`inter`, a boolean in the frame of `reg`,
    whose top-left is page (rx0, ry0)) as a lobe: its page-coordinate row
    dict, and its own cleaning job — the interior's box, with the original
    ink inside the interior as its block. Paints the interior into `mask`
    (the entry's bbox frame). None when it has under 12 usable rows."""
    x, y, w, h = bbox
    rh = inter.shape[0]
    rows = {}
    for ry in range(rh):
        run = widest_run(inter[ry])
        if run is None:
            continue
        gx0, gx1 = rx0 + run[0], rx0 + run[1]
        rows[ry0 + ry] = (gx0, gx1)
        a, b = max(gx0, x) - x, min(gx1, x + w) - x
        if 0 <= ry0 + ry - y < h and b > a:
            mask[ry0 + ry - y, a:b] = 1
    if len(rows) < 12:
        return None
    iy, ix = np.nonzero(inter)
    lx0, ly0 = rx0 + int(ix.min()), ry0 + int(iy.min())
    lx1, ly1 = rx0 + int(ix.max()) + 1, ry0 + int(iy.max()) + 1
    sub = inter[ly0 - ry0:ly1 - ry0, lx0 - rx0:lx1 - rx0]
    ink = (reg[ly0 - ry0:ly1 - ry0,
               lx0 - rx0:lx1 - rx0].min(axis=2) <= DARK_MAX) & (sub > 0)
    nk, _, stk, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), 8)
    sel = [i for i in range(1, nk)
           if 9 <= stk[i, 3] <= 48 and stk[i, 4] >= 8]
    if sel:
        bx0 = lx0 + min(stk[i, 0] for i in sel)
        by0 = ly0 + min(stk[i, 1] for i in sel)
        bx1 = lx0 + max(stk[i, 0] + stk[i, 2] for i in sel)
        by1 = ly0 + max(stk[i, 1] + stk[i, 3] for i in sel)
    else:                           # no lettering found: the interior itself
        bx0, by0, bx1, by1 = lx0, ly0, lx1, ly1
    return rows, ([lx0, ly0, lx1 - lx0, ly1 - ly0],
                  [bx0, by0, bx1 - bx0, by1 - by0],
                  sub.astype(np.uint8))


def repair_letter_bites(mask, page, origin, block_box, PAD=14):
    """Give a balloon back the interior its own lettering cost the mask.

    Detection walls its flood with the drawn ink CLOSED by a 7x7 kernel, so
    that a gap in the outline cannot leak. Wherever a letter sits within that
    distance of the outline the close WELDS the two: the letter is no longer
    an ENCLOSED hole, the hole-fill leaves a bite out of the mask, and a line
    of text that all but spans the balloon seals whole regions off the
    interior. The mask then under-reports the balloon both ways that matter —
    the fit sees a fraction of the width the balloon offers (p65 b08: 127px
    of a 252px balloon) and drops several size steps, and the cleaning, which
    is confined to the mask, never reaches the letters inside the bite, so
    they survive as the pale remnants users read as small white objects.

    Re-flood with the RAW ink as the only wall: the 2-3px the letterer left
    between letter and outline survives, the interior stays ONE component and
    the letters are enclosed holes again.

    That flood has no seal against a real break in the outline, so what it
    recovers is only trusted where the mask and the drawn ink ALREADY enclose
    it. A bite is a POCKET: the welded letter is its wall on one side and the
    mask is its wall on the others. A leak is enclosed by nothing — it runs
    on until it meets the crop. Keying on that tells the two apart even when
    the leak lands inside the letterer's block, which is where it actually
    lands: p104's balloon sits on pale sky with its outline broken where a
    sword crosses it, and the flood ran out onto the sky BEHIND the lettering
    — the cleaner then read the fill off sky, painted blue over the artwork
    and left the old lettering standing. A block bound cannot see that (the
    blob cluster reaches past the balloon's own arc), and neither can an area
    ratio: the escape was 1.7% of the mask.
    """
    h, w = mask.shape
    ox, oy = int(origin[0]), int(origin[1])
    x0, y0, x1, y1 = (int(v) for v in block_box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return mask
    # Work on a window PADDED past the bubble's box. The box is the mask's own
    # extent and the mask is eroded off the outline, so the drawn outline
    # usually falls just OUTSIDE it — and a pocket whose wall is that outline
    # then opens at the crop edge and reads as a leak. The pad brings the wall
    # into the frame.
    py0, px0 = min(PAD, oy), min(PAD, ox)
    win = page[oy - py0:min(page.shape[0], oy + h + PAD),
               ox - px0:min(page.shape[1], ox + w + PAD)]
    big = np.zeros(win.shape[:2], np.uint8)
    big[py0:py0 + h, px0:px0 + w] = mask
    stroke = (win.min(axis=2) <= DARK_MAX - 20).astype(np.uint8)
    free = bubble_interior(stroke, np.ones(big.shape, np.uint8),
                           (x0 + px0, y0 + py0, x1 + px0, y1 + py0), img=win)
    # what the mask and the drawn ink already close in (the ink threshold is
    # the generous one: a pocket wall with a hole in it is not a pocket)
    sealed = fill_holes((big > 0) | (win.min(axis=2) <= DARK_MAX))
    add = np.zeros(big.shape, bool)
    add[y0 + py0:y1 + py0, x0 + px0:x1 + px0] = (
        fill_holes(free) & sealed)[y0 + py0:y1 + py0, x0 + px0:x1 + px0]
    out = mask.copy()
    out[add[py0:py0 + h, px0:px0 + w]] = mask.max() if mask.max() else 1
    return out


def resolve_rows(rows, fill_rows, block_rows, pad):
    """Row profile resolved against the balloon's own fill.

    The fill is the authority in BOTH directions. Where the profile reaches
    past it the profile has leaked — a mask covering balloon AND page gives a
    RECTANGLE, and lines get set where the oval long since narrowed. Where
    the profile sits further INSIDE it than the profile's own clearance
    (`pad`), that is not clearance, it is a letter bite: `rows` is the mask
    eroded, so it carries every bite the mask does, detection's letter-union
    repair only reaches rows that actually hold letter pixels, and the
    erosion smears a bite three rows past the letter that caused it. Those
    neighbouring rows are what a line box lands on, and rows_span takes the
    NARROWEST row of a band, so one chewed row vetoes the whole line.

    Widening is confined to the letterer's block rows for the same reason the
    mask repair is: beyond them the fill has nothing to vouch for it.
    """
    by0, by1 = block_rows
    out = {}
    for y, (lo, hi) in rows.items():
        span = fill_rows.get(y)
        if span is None:
            continue
        if by0 <= y < by1:
            if lo - span[0] > pad:
                lo = span[0] + pad
            if span[1] - hi > pad:
                hi = span[1] - pad
        a, b = max(lo, span[0]), min(hi, span[1])
        if b - a > 8:
            out[y] = (a, b)
    return out


def caption_boxes(bubs, fitted):
    """The coloured caption boxes on a page, as (x0, y0, x1, y1).

    A gradient-filled box comes back from detection as a `tint` strip plus a
    `bubble` strip directly above or below it, sharing its x range — that is
    the split, and it is a tight enough signature to key on. Anything looser
    (a rectangular mask, a drawn frame) also matches ordinary oval balloons
    and leaked masks, which must never be wiped as a box. "Sharing its x
    range" means the two extents AGREE (X_AGREE), not that one overlaps the
    other: an oval of similar width sitting a little to one side passes an
    overlap test easily.

    Only entries the fit actually placed count: a bubble left unplaced keeps
    its original lettering, so its box must not be wiped.

    Returns [((x0, y0, x1, y1), {member indices})]. The members matter: the
    box must be cleaned INSTEAD of its entries, never after them. A per-entry
    pass repaints its own strip near-white, and the box pass would then read
    that as the fill level and paint a white band right across the box."""
    out = []
    for i, a in enumerate(bubs, 1):
        if a["kind"] != "tint" or i not in fitted:
            continue
        ax, ay, aw, ah = a["bbox"]
        x0, y0, x1, y1 = ax, ay, ax + aw, ay + ah
        members = {i}
        for j, c in enumerate(bubs, 1):
            if j == i or c["kind"] != "bubble" or j not in fitted:
                continue
            cx, cy, cw, ch = c["bbox"]
            # the two strips may OVERLAP, not just abut: detection does not
            # always cut the box cleanly at the seam (p219's bubble strip
            # starts 26px above the tint strip's bottom). The count is stable
            # from -30 to -60, so the tolerance is not tuned to one page.
            gap = cy - (ay + ah) if cy > ay else ay - (cy + ch)
            if not -40 <= gap <= 30:
                continue
            # The two halves are cut from ONE rectangle by a horizontal fill
            # threshold, so their x extents AGREE — measure that, not mere
            # overlap. Bounding the intersection against the wider member
            # let a balloon of similar width but offset edges in: p164's
            # 257x86 oval two panels up joined a caption box,
            # which grew it from 233x73 to 277x179 (the fit then had 265px
            # of writing width in a 230px box and both lines printed past
            # the frame) and left the box with a member nothing typesets, so
            # the wipe skipped the whole box and its strips were cleaned
            # separately — a flat slab with a white band at the seam. Over
            # the book the 31 real pairs run 0.893-0.998 and the two false
            # ones sit at 0.716 and 0.769.
            inter = min(ax + aw, cx + cw) - max(ax, cx)
            union = max(ax + aw, cx + cw) - min(ax, cx)
            if inter < X_AGREE * union:
                continue
            x0, y0 = min(x0, cx), min(y0, cy)
            x1, y1 = max(x1, cx + cw), max(y1, cy + ch)
            members.add(j)
        out.append(((x0, y0, x1, y1), members))
    return out


def stack_box_lines(members, lh, cy):
    """y_top for every line of a caption box laid out as ONE block of type.

    `members` = [(bi, n_lines)] in reading order; returns {bi: [y_top...]}.

    Detection cuts a gradient box into a `tint` strip and a `bubble` strip,
    and each strip's lines were then centred in the strip — so the box
    printed with a blank line at the seam: "too much space between
    paragraphs, as if it were 2 bubbles not one" (p147's bottom-left box,
    p219). The letterer set the box as one piece of type on one line grid,
    centred on his own block; the strips are an artefact of the fill
    crossing LIGHT_MIN and carry no meaning for the layout."""
    n = sum(k for _, k in members)
    top = cy - n * lh / 2.0
    out, i = {}, 0
    for bi, k in members:
        out[bi] = [top + (i + j) * lh for j in range(k)]
        i += k
    return out


def frame_bounds(region):
    """The drawn frame inside a caption box, as (x0, y0, x1, y1) to keep.

    The box rectangle detection hands back does not coincide with the frame
    the artist drew — it runs a few px wide of it. The repaint then covers a
    band OUTSIDE the frame (a tab of fill colour on the artwork) and wipes
    the frame itself where it happens to fall inside the box.

    The frame is not reliably recovered as a "long thin curve": the fill
    estimate is a horizontal close, which a rule spanning the full width
    survives, so the rule only registers as ink where the gradient pushes it
    past the threshold and the surviving pieces are too short to look like a
    curve. Its geometry is unmistakable though — a near-solid dark row or
    column across the box — so find it as that, and let the caller keep the
    repaint strictly inside it. Returns None when the box has no drawn frame.
    """
    h2, w2 = region.shape[:2]
    if h2 < 8 or w2 < 8:
        return None
    # on the MAX channel, not the min: a yellow fill has a low BLUE channel,
    # so min() calls the whole box dark (93% of p42's box) and no frame can
    # be picked out of it. Only true black ink is dark in every channel.
    solid = region.max(axis=2) <= FRAME_MAX
    if solid.mean() > 0.5:
        # a white-on-black caption: every row is "solid" and there is no
        # frame to find. Reading one here would shrink the box to nothing.
        return None
    rowf, colf = solid.mean(axis=1), solid.mean(axis=0)
    # the frame sits AT the edge — a rule found well inside the box is a
    # line of artwork or a stroke of lettering, not the border
    band_y, band_x = max(2, min(8, h2 // 6)), max(2, min(8, w2 // 6))

    def edge(frac, lo, hi, thresh=0.75):
        hits = [i for i in range(lo, hi) if frac[i] >= thresh]
        return hits or None
    top = edge(rowf, 0, band_y)
    bot = edge(rowf, h2 - band_y, h2)
    left = edge(colf, 0, band_x)
    right = edge(colf, w2 - band_x, w2)
    if not (top or bot or left or right):
        return None
    return (max(left) + 1 if left else 0,
            max(top) + 1 if top else 0,
            min(right) if right else w2,
            min(bot) if bot else h2)


def clean_caption_box(img, box):
    """Wipe the old lettering from a whole coloured caption box.

    letter_mask is a "dark text on light paper" ring test, and on a
    yellow->white gradient fill the bright half fails it. The box then comes
    back as TWO stacked entries — a `tint` strip and a `bubble` strip, split
    where the fill crosses LIGHT_MIN — each carrying a RAGGED mask that
    follows the gradient instead of the box. Letters straddle the mask edge,
    so the band at the junction is never cleaned and the new line is set on
    top of the old one.

    Both entries are typeset, so the whole box may be wiped; their bboxes
    together already span every line of it. Repaint with the ROW median so
    the box keeps its vertical gradient (a flat fill reads as a coloured
    slab), and spare the long thin curves so the drawn frame survives."""
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(img.shape[1], x1), min(img.shape[0], y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    region = img[y0:y1, x0:x1]
    minch = region.min(axis=2)
    h2, w2 = minch.shape
    # The fill varies BOTH ways across these boxes: a vertical gradient, and
    # often a highlight along the row. A single fill level per row then reads
    # the bright part as the fill and everything yellow as ink, and paints
    # the row WHITE — a band straight across the box. So estimate the fill
    # LOCALLY: a horizontal grayscale close removes anything as narrow as
    # lettering and leaves the fill behind.
    k = int(max(21, min(w2 // 2, 121)) // 2 * 2 + 1)
    kern = np.ones((1, k), np.uint8)

    def background(plane):
        return cv2.blur(cv2.morphologyEx(plane, cv2.MORPH_CLOSE, kern),
                        (25, 5))
    bg = background(minch).astype(np.int16)
    ink = minch.astype(np.int16) <= bg - 45
    n, lab, st, _ = cv2.connectedComponentsWithStats(
        ink.astype(np.uint8), 8)
    span = max(w2, h2)
    keep = np.zeros(ink.shape, bool)
    for ci in range(1, n):
        cw, ch, ca = st[ci, 2], st[ci, 3], st[ci, 4]
        if max(cw, ch) >= 0.45 * span and ca <= 0.28 * max(1, cw * ch):
            keep[lab == ci] = True              # the drawn frame / a rule
    # the drawn frame, found by its geometry rather than by looking like a
    # curve, and everything outside it left alone: the box rectangle runs
    # wide of the frame, so repainting the whole of it wiped the frame and
    # put a tab of fill colour on the artwork beside it (p42).
    inner = frame_bounds(region)
    # the frame's home is the box RIM and the lettering lies inside it, so
    # the same weld happens here: a line-end that touches the rule is one
    # component with it and the curve test spared the lot. p147's boxes
    # printed "NTRA", "OLA" and a stray "E" over the new text (1554px of
    # old lettering on one box). Re-ask the question inside the frame.
    ins = np.zeros(ink.shape, bool)
    fx0, fy0, fx1, fy1 = inner if inner is not None else (0, 0, w2, h2)
    ins[fy0:fy1, fx0:fx1] = True
    welded = keep & ins
    if welded.any():
        # the box's own letter height, measured
        nk, _, stk, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), 8)
        hsk = [stk[i, 3] for i in range(1, nk)
               if 9 <= stk[i, 3] <= 48 and stk[i, 4] >= 8]
        capk = float(np.median(hsk)) if hsk else 0.0
        if capk:
            # ...and the same depth test as a balloon's outline needs: the
            # rule hugs the box rim, a letter body stands a third of a
            # letter height inside it. Intersecting with the frame bounds
            # alone is not enough — the rule's own inner sliver is inside
            # them too, and it runs the height of the box, so the piece
            # stays "too tall to be a letter" and nothing is reclaimed.
            deep = max(4.0, 0.35 * capk)
            dist = cv2.distanceTransform(ins.astype(np.uint8), cv2.DIST_L2, 3)
            bodies = welded & (dist >= deep)
            nw, lw, stw, _ = cv2.connectedComponentsWithStats(
                bodies.astype(np.uint8), 8)
            reclaim = np.zeros(ink.shape, bool)
            for ci in range(1, nw):
                cw, ch, ca = stw[ci, 2], stw[ci, 3], stw[ci, 4]
                if ca < 8 or (max(cw, ch) >= 0.45 * span
                              and ca <= 0.28 * max(1, cw * ch)):
                    continue
                if not 0.5 * capk <= ch <= 1.8 * capk:
                    continue
                reclaim |= lw == ci
            if reclaim.any():
                r = int(np.ceil(deep))
                keep &= ~(cv2.dilate(reclaim.astype(np.uint8),
                                     np.ones((2 * r + 1, 2 * r + 1),
                                             np.uint8)) > 0)
    if inner is not None:
        outside = ~ins
        keep |= outside
    drawn = ink.copy()                  # ink INCLUDING the spared curves
    ink &= ~keep
    halo = cv2.dilate(ink.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    sel = (ink | halo) & ~keep
    if not sel.any():
        return
    # what may be MEASURED as fill: never any drawn ink, whether or not it is
    # being repainted. The frame is spared from the repaint, so it is not in
    # `sel`, and taking it for fill dragged the estimate beside it 100 levels
    # dark — a dark rim inside the box's own edge.
    known = ~(drawn | halo)
    if inner is not None:
        # never read fill from the artwork outside the frame either
        known &= ~outside

    def under_the_ink(plane):
        """The fill the lettering covers, interpolated ACROSS it from the
        fill on either side, per row.

        The close above is the right tool for FINDING the ink — it only has
        to be a threshold — but not for replacing it: a close takes the
        BRIGHTEST value in its window, and the window has to be as wide as a
        whole word (a narrower one leaves the word's interior dark and the
        estimate tracks the ink itself). On a box whose fill grades along the
        row that reads the bright end of the gradient as the fill everywhere,
        and the line is repainted as a flat, too-bright slab with straight
        edges — p42's caption came out 20-45 levels bright across its whole
        line. Interpolation reproduces the gradient instead.
        """
        src = plane.astype(np.float32)
        out = np.empty_like(src)
        xs = np.arange(w2, dtype=np.float32)
        ok = np.zeros(h2, bool)
        for r in range(h2):
            idx = np.flatnonzero(known[r])
            if len(idx) < max(4, int(0.02 * w2)):
                continue
            out[r] = np.interp(xs, idx.astype(np.float32), src[r][idx])
            ok[r] = True
        if not ok.any():
            return None
        # a row the lettering fills end to end has no fill of its own left to
        # read: interpolate it DOWN the column instead, between the nearest
        # rows that do. Copying the nearest row flat loses the box's vertical
        # gradient exactly where a line of type sits on it.
        good, bad = np.flatnonzero(ok), np.flatnonzero(~ok)
        if len(bad):
            for c in range(w2):
                out[bad, c] = np.interp(bad, good, out[good, c])
        return cv2.blur(out, (15, 3))
    est = [under_the_ink(region[:, :, c]) for c in range(3)]
    if any(e is None for e in est):     # nothing to interpolate from at all
        bgc = np.stack([background(region[:, :, c]) for c in range(3)], axis=2)
    else:
        bgc = np.clip(np.stack(est, axis=2), 0, 255).astype(np.uint8)
    region[sel] = bgc[sel]
    img[y0:y1, x0:x1] = region


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
        minch = region.min(axis=2)
        # the WALL is drawn ink only — a strong threshold, so that a tinted
        # fill (yellow caption, blue box) whose own luminance sits near the
        # ink cut-off cannot fragment the interior into bands
        stroke = (minch <= DARK_MAX - 20).astype(np.uint8)
        bx2, by2, bw2, bh2 = b["block"]
        interior = bubble_interior(stroke, keep_ring, (
            max(0, bx2 - x), max(0, by2 - y),
            min(w, bx2 + bw2 - x), min(h, by2 + bh2 - y)), img=region)
        # ...and what counts as old lettering adapts to THIS balloon's fill,
        # capped so that a striped or grey-shaded balloon keeps its shading
        ref = float(np.median(minch[interior])) if interior.any() else 255.0
        cut = min(DARK_MAX + 40, max(60.0, ref - 60.0))
        dark = (minch <= cut).astype(np.uint8)
        # What to wipe. Enclosure alone is not enough: a joined-balloon
        # split can truncate the mask mid-text, and letters straddling that
        # edge are not enclosed, so they survived and the new text landed on
        # top of them. So wipe every dark pixel the mask covers EXCEPT the
        # long thin curves — a balloon outline or a panel line spans the
        # balloon and encloses almost nothing, while lettering (even several
        # letters merged into one blob) is compact. That protects the
        # outline even when the mask has leaked well past it.
        solid = fill_holes(interior)
        # where the letterer WROTE — the bound both the weld repair below
        # and the strict ink term further down are held to
        sx0, sy0 = max(0, bx2 - x), max(0, by2 - y)
        sx1, sy1 = min(w, bx2 + bw2 - x), min(h, by2 + bh2 - y)
        inblock = np.zeros(dark.shape, bool)
        inblock[sy0:sy1, sx0:sx1] = True
        # this balloon's own letter height, measured (never a fixed pixel
        # count): it bounds what may be called lettering below, and how far
        # past the block the strict term may reach — detection's paragraph
        # cluster can leave an accent or a stray letter a little outside it.
        nl, _, stl, _ = cv2.connectedComponentsWithStats(
            (dark & inblock).astype(np.uint8), 8)
        hs = [stl[i, 3] for i in range(1, nl)
              if 9 <= stl[i, 3] <= 48 and stl[i, 4] >= 8]
        cap = float(np.median(hs)) if hs else 0.0
        pad = int(max(8, round(2 * cap)))
        # two different bounds, and they must not be confused. The strict
        # ink term may reach a little PAST the block (a stray letter the
        # cluster left out); the weld repair may not — padding there pulls
        # the outline itself into the region it examines, and the welded
        # letter stops being a piece of its own.
        home = solid | (cv2.dilate(inblock.astype(np.uint8),
                                   np.ones((2 * pad + 1, 2 * pad + 1),
                                           np.uint8)) > 0)
        letterland = inblock | solid
        # classify on the WHOLE region, never on the mask-clipped copy: the
        # mask boundary chops the outline into short arcs that no longer
        # look like long curves, and each arc was then wiped as lettering
        nc, lab, stc, _ = cv2.connectedComponentsWithStats(dark, 8)
        span = max(w, h)

        def _curve(cw, ch, ca):
            return (max(cw, ch) >= 0.45 * span
                    and ca <= 0.28 * max(1, cw * ch))

        protect = np.zeros(dark.shape, bool)
        for ci in range(1, nc):
            cw, ch, ca = stc[ci, 2], stc[ci, 3], stc[ci, 4]
            if _curve(cw, ch, ca):
                protect[lab == ci] = True
        # ...but the lettering runs right up to the outline, and at the ink
        # threshold a letter that TOUCHES it is ONE component with it — so
        # the curve test spared the letter too, and it printed beside the
        # new text: p215 shipped a stray "L" (the L of "EL", fused to the
        # right arc: 111x152, sparse 0.26). Neither gate can see that — the
        # leftover scan is detection's ring test and a fused letter has no
        # ring, and qa_scan only sees pixels that CHANGED.
        #
        # Nothing separates the two by component: they are one, for the raw
        # ink as much as for the threshold, so repair_letter_bites cannot
        # find this letter either (it is not an enclosed hole even with the
        # ink as the only wall, and 82 of 4323 px is all its re-flood
        # recovers here). Nor by distance from the fill: the bite makes the
        # fill's own boundary wrap around the letter, so the letter lies
        # ON that boundary.
        #
        # What does separate them is DEPTH into the balloon. Fill the
        # letterer's ink into the fill and the result is the balloon's own
        # interior: the outline then hugs its rim, while a letter BODY
        # stands a third of a letter height further in. Reclaim what is
        # that deep, is letter-sized, and is not itself a rule.
        if cap and protect.any():
            hull = fill_holes(solid | ((dark > 0) & inblock))
            dist = cv2.distanceTransform(hull.astype(np.uint8), cv2.DIST_L2, 3)
            deep = max(4.0, 0.35 * cap)
            bodies = protect & (dist >= deep)
            nb2, lb2, stb2, _ = cv2.connectedComponentsWithStats(
                bodies.astype(np.uint8), 8)
            reclaim = np.zeros(dark.shape, bool)
            for ci in range(1, nb2):
                cw, ch, ca = stb2[ci, 2], stb2[ci, 3], stb2[ci, 4]
                if ca < 8 or _curve(cw, ch, ca):
                    continue            # a speck, or a rule that spans
                # ...and it has to be the size of THIS balloon's lettering.
                # Without that the repair reclaimed the artwork a block
                # swallows — a block corner is often a slab of art, and
                # wiping it painted 115x261 of p149 and 92x47 of p198 with
                # the fill colour.
                if not 0.5 * cap <= ch <= 1.8 * cap:
                    continue
                reclaim |= lb2 == ci
            if reclaim.any():
                # give the body back the rim the depth test cut off it —
                # which is exactly `deep` px thick — or a letter-shaped
                # hook is left hugging the outline. This does nick the arc
                # where the letter was fused to it: a few px of drawn line
                # traded for a stray letter standing on the artwork.
                r = int(np.ceil(deep))
                protect &= ~(cv2.dilate(reclaim.astype(np.uint8),
                                        np.ones((2 * r + 1, 2 * r + 1),
                                                np.uint8)) > 0)
        # That strict cut-off is what lets a striped or grey-shaded balloon
        # keep its shading, but the lettering's PALE strokes never reach it:
        # the upscaler leaves thin strokes and antialiased rims at 150-220 on
        # a 250 fill, so they were neither ink nor within the 4px halo of any
        # (measured up to 51px from the nearest wiped pixel) and survived as
        # faint letter-shaped specks in 1428 bubbles on 263 pages. Inside the
        # letterer's TEXT BLOCK a softer cut-off is safe — it is a text area,
        # not somewhere shading lives — so bound it to the block, and keep
        # clear of the protected outline so its rim is not thinned.
        soft = np.zeros(dark.shape, bool)
        if sx1 > sx0 and sy1 > sy0:
            soft[sy0:sy1, sx0:sx1] = (minch[sy0:sy1, sx0:sx1]
                                      <= max(cut, ref - 30.0))
        # ...and inside the balloon's own FILL, never merely inside the mask.
        # The block swallows artwork beside the balloon and the mask leaks
        # past the outline, so a block-bounded soft threshold caught the art
        # as ink and the row median painted it with the fill colour — white
        # rectangles across the page.
        soft &= solid
        soft &= ~(cv2.dilate(protect.astype(np.uint8),
                             np.ones((5, 5), np.uint8)) > 0)
        # The strict term belongs inside the text block or inside the fill,
        # never merely inside the mask. Where a mask spilled past the
        # outline (p149: 558px of mask for a 254px text block) the artwork's
        # own line work was read as old lettering and painted with the
        # balloon's fill colour — white streaks and blobs across the ice.
        # Beyond the fill and the block the mask is evidence of nothing —
        # that is where a leak puts it. But the letterer does leave text out
        # there: a line the ceded lobe's mask cut off (p294), an accent the
        # paragraph cluster missed (p42, p104). Lettering out there is
        # LETTER-SHAPED — a body about as tall as the letter height measured
        # in this very balloon — while the artwork's line work is a hairline
        # or a sweep much taller than a letter, which is what got painted
        # with the fill colour before.
        far = (dark > 0) & ~home
        if cap and far.any():
            nf, lf, stf, _ = cv2.connectedComponentsWithStats(
                far.astype(np.uint8), 8)
            letterish = np.zeros(dark.shape, bool)
            for ci in range(1, nf):
                cw, ch = stf[ci, 2], stf[ci, 3]
                if 0.5 * cap <= ch <= 1.8 * cap and cw <= 8 * cap:
                    letterish[lf == ci] = True
            far = letterish
        ink = ((((dark > 0) & home) | far | soft)
               & (keep_ring > 0) & ~protect)
        ink |= solid & (dark > 0)        # enclosed letters, always
        ink = ink.astype(np.uint8)
        halo = cv2.dilate(ink, np.ones((9, 9), np.uint8)) > 0
        # the row median that replaces the letters is measured on the fill
        # alone — any dark structure the mask covers would drag it grey,
        # and past a leak the spilled-onto background (sky, art) would tint
        # it: the wiped letters came back as coloured ghost blocks
        structure = cv2.dilate(((dark > 0) | soft).astype(np.uint8),
                               np.ones((5, 5), np.uint8)) > 0
        paint = interior | (ink > 0)
        for ry in range(h):
            if allowed_ys is not None and (y + ry) not in allowed_ys:
                continue
            sel = paint[ry] & halo[ry]
            if not sel.any():
                continue
            good = interior[ry] & ~structure[ry]
            med = (np.median(region[ry][good], axis=0) if good.any()
                   else np.array([255, 255, 255]))
            region[ry][sel] = med.astype(np.uint8)
        img[y:y + h, x:x + w] = region
        # OVERFLOW lines: the original letterer sometimes spills a line past
        # the balloon outline onto the page margin — outside any honest
        # mask. Inpaint letter-sized dark blobs inside the TEXT BLOCK box
        # that the mask fill did not reach.
        bx2, by2, bw2, bh2 = b["block"]
        # ...but NEVER outside this balloon's own bbox. The block comes from a
        # blob cluster and happily swallows the balloon next door, and the
        # step then erased a neighbour's lettering that nothing types back —
        # two balloons shipped empty. Measured over this book, the window
        # reaching past the bbox bought 3 stray blobs (<=13px out, 276px) and
        # cost 45 blobs of a neighbour's lettering and line art; every one of
        # the 1704 blobs it legitimately cleans lies INSIDE the bbox. No
        # margin can help: the balloons are adjacent.
        ox0, oy0 = max(0, bx2 - 8, x), max(0, by2 - 8, y)
        ox1 = min(img.shape[1], bx2 + bw2 + 8, x + w)
        oy1 = min(img.shape[0], by2 + bh2 + 8, y + h)
        if ox1 - ox0 < 8 or oy1 - oy0 < 8:
            return
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
        # bubble outlines, panel borders and art are never touched.
        #
        # `open` means "lettering on light ground with no balloon to mask",
        # so the mask is the letterer's block — a RECTANGLE. Where detection
        # reads a real balloon as `open` (4 entries in this book) that
        # rectangle crosses the drawn outline, and the arcs inside it are
        # letter-sized: inpainting them floods the window from its own white
        # surroundings, so the page printed a white RECTANGLE with the
        # balloon's sides erased (p182, p242). The bubble branch already
        # refuses a blob that touches its window border for exactly this
        # reason; do the same here, which needs a window with a MARGIN (the
        # block's own edge is where this entry's letters sit), and spare the
        # long thin strokes of the drawn art as everywhere else.
        # An 8px margin, as the bubble branch uses: wider does not help
        # (p182's "S" is welded to the outline, so it goes with the arcs
        # whatever the window) and it lets the inpaint reach the artwork.
        ox0, oy0 = max(0, x - 8), max(0, y - 8)
        ox1 = min(img.shape[1], x + w + 8)
        oy1 = min(img.shape[0], y + h + 8)
        oreg = img[oy0:oy1, ox0:ox1]
        darkish = (oreg.min(axis=2) <= 175).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(darkish, 8)
        letters = np.zeros_like(darkish)
        wh, ww = darkish.shape
        span = max(ww, wh)
        for i in range(1, n):
            cx2, cy2, cw, ch, area = stats[i]
            if not (ch <= 52 and cw <= 130 and area <= 2600):
                continue
            if cx2 == 0 or cy2 == 0 or cx2 + cw >= ww or cy2 + ch >= wh:
                continue
            if max(cw, ch) >= 0.45 * span and area <= 0.28 * max(1, cw * ch):
                continue
            letters[labels == i] = 1
        if letters.any():
            letters = cv2.dilate(letters, np.ones((5, 5), np.uint8))
            img[oy0:oy1, ox0:ox1] = cv2.inpaint(
                oreg, letters, 5, cv2.INPAINT_TELEA)
        return
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


def unfound_balloon(img, b):
    """The balloon an `open`/`margin` entry is really in, as (x0, x1) in page
    coords — or None when there is none to find.

    `open` and `margin` have no balloon by definition: the block is a bare
    rectangle over the artwork and the cleaning is a letter-only inpaint. But
    detection also falls back to `open` when its walled flood fails on a
    balloon that IS drawn, and then nothing bounds the new type: p242's
    shout was set 141px wide in a 132px oval and printed across the
    outline on both sides, and its block — the blob cluster, which had
    swallowed both arcs of that oval — was 148px, so the block could not
    catch it either. The oval's own cap height is ~25px and the arcs carried
    the measured one to 36px, so the size ceiling could not catch it either.

    Flooding out from the block finds the balloon when there is one. The
    test that it IS one is that it comes back NARROWER than the block: the
    block runs the width of the lettering, so a real interior contains it
    and is wider only where the flood escaped onto the artwork (p222 b04's
    flood ran down the tail and came back 194px for a 132px block). Returns
    None then, and the entry keeps the block as its writing area.
    """
    bx, by, bw, bh = b["block"]
    pad = max(30, int(round(0.6 * max(bw, bh))))
    x0, y0 = max(0, bx - pad), max(0, by - pad)
    x1, y1 = min(img.shape[1], bx + bw + pad), min(img.shape[0], by + bh + pad)
    reg = img[y0:y1, x0:x1]
    dark = (reg.min(axis=2) <= DARK_MAX - 20).astype(np.uint8)
    inter = fill_holes(bubble_interior(
        dark, np.ones(dark.shape, np.uint8),
        (bx - x0, by - y0, bx + bw - x0, by + bh - y0), img=reg))
    runs = [r for r in (widest_run(inter[ry]) for ry in range(y1 - y0)) if r]
    if not runs:
        return None
    lo = x0 + min(r[0] for r in runs)
    hi = x0 + max(r[1] for r in runs)
    # narrower than the block = a real interior; wider = the flood escaped
    if hi - lo >= bw or hi - lo < 16:
        return None
    return (lo, hi)


# corrections to the original lettering's own errors, applied at typeset
# time (transcripts.json stays a verbatim record of the source). Per-comic
# data: relettering/<comic>/typo_fixes.json, a list of [wrong, fixed] pairs
def override_key(stem: str, bi: int) -> str:
    import re as _re
    m = _re.match(r"^\D*(\d+)\b.*?(\d+(?:-\d)?)$", stem)
    return f"{m.group(1)}-{m.group(2)} b{bi:02d}" if m else f"{stem} b{bi:02d}"


def override_for(stem: str, bi: int):
    # a hand-written override always wins over a box the frame finder
    # derived (auto_caption_boxes skips any box with one)
    key = override_key(stem, bi)
    return LAYOUT_OVERRIDES.get(key) or AUTO_OVERRIDES.get(key)


def apply_fixes(text):
    for a, b in TYPO_FIXES:
        if a in text:
            print(f"    typo fix: {a!r} -> {b!r}")
            text = text.replace(a, b)
    return text


BOOK_PCTL = 35  # percentile of per-bubble maxima used as the common size


def is_strip(bb):
    """A caption strip is WIDE and short — a whole box is at most a
    few lines tall and runs the width of a panel. The tint+bubble
    signature alone false-pairs an ordinary oval balloon with an
    unplaced tint region just below it (p232 b05 + b07, gap 4px,
    x-overlap 81%), and keying the box on `fitted` used to hide that
    because the tint was never typeset. Wiping that "box" painted a
    white slab over the balloon and the artwork, and laying the text
    out in it set the lines across both."""
    return bb["bbox"][2] >= 3.0 * max(1, bb["bbox"][3])


def page_caption_boxes(bubbles, stem):
    """The page's coloured caption boxes: {entry: box}, {entry:
    members} and the set of entries that belong to one.

    Shared by the fit and by anything that has to reproduce the
    cleaning (reclean_all). The membership rule is subtle enough that a
    second, hand-written copy of it diverges silently.
    """
    # entries that form a coloured caption box. Their fill legitimately
    # shifts hue across the box (yellow -> white), so the colour wall —
    # right for a balloon against the page — must not be applied to
    # them: it cut p101's caption profile from 60 rows to 45 and the
    # fit then found no size at all.
    cap_box, cap_mem = {}, {}
    framed = AUTO_MEMBERS.get(stem, set())
    for _bx, _mem in caption_boxes(bubbles,
                                   set(range(1, len(bubbles) + 1))):
        if _mem & framed:
            continue                # its frame was found: one whole box
        for _i in _mem:
            cap_box[_i] = _bx
            cap_mem[_i] = set(_mem)
    for _bi in range(1, len(bubbles) + 1):
        _ov = override_for(stem, _bi)
        if _ov and "box" in _ov:
            cap_box[_bi] = tuple(_ov["box"])
            cap_mem[_bi] = {_bi}
    cap_members = set(cap_box)
    return cap_box, cap_mem, cap_members


def boxes_to_wipe(bubbles, stem, fitted):
    """The caption boxes to clean as ONE box, for a page whose typeset
    entries are `fitted`. Shared with reclean_all for the same reason
    prepare_bubble is.

    The box to WIPE is the box the fit typeset into, which is built from
    every entry: a box whose text was transcribed as one string on one
    strip leaves its sibling unplaced, and keying the wipe on `fitted` then
    shrank the box back to that strip — the rest of the box kept the
    original's lines with the new text over them. A box with no typeset
    member at all is still never wiped (it keeps its own lettering), which
    is what the rule was there for.
    """
    boxes = []
    framed = AUTO_MEMBERS.get(stem, set())
    for _bx, _mem in caption_boxes(bubbles,
                                   set(range(1, len(bubbles) + 1))):
        if _mem & framed:
            continue                # wiped whole, as its framed box
        if not _mem & fitted:
            continue                # no member typeset: it keeps its own text
        if _mem <= fitted or all(is_strip(bubbles[i - 1]) for i in _mem):
            boxes.append((_bx, _mem))
    for bi in fitted:
        ov = override_for(stem, bi)
        if ov and "box" in ov:
            boxes.append((tuple(ov["box"]), {bi}))
    return boxes


def frame_box(img, block, bbox):
    """The INNER edge (x0, y0, x1, y1) of the drawn rectangular frame
    holding `block`, or None when the block sits in no such frame.

    A gradient caption box defeats detection in more ways than the
    tint+bubble split `caption_boxes` knows: the fill crosses LIGHT_MIN
    wherever it likes, so a box comes back as one strip that misses its
    pale lines (p2-004's "FORJADAS."), as three slivers side by side with
    the top two lines in none of them (p2-066), or with one line cut in
    two. The FRAME does not care about the fill. Flood the box's fill
    from the lettering with only true black (dark in every channel, the
    frame and the letters) as the wall: a yellow-to-white fill is one
    region, and inside a drawn frame it comes back a near-perfect
    rectangle. An oval balloon does not (0.55-0.78 against >=0.93), and
    a box drawn on open art fails the ink and frame tests below.

    The window grows once when the flood reaches it, because a sliver's
    block can be a small fraction of its box (p3-020: 79px of a 446px
    box); a flood that still reaches it has escaped the box."""
    for mx, my in ((260, 200), (700, 400)):
        got = _frame_box(img, block, bbox, mx, my)
        if got != "escaped":
            return got
    return None


def _frame_box(img, block, bbox, mx, my):
    H, W = img.shape[:2]
    bx, by, bw, bh = block
    X0, Y0 = max(0, min(bx, bbox[0]) - mx), max(0, min(by, bbox[1]) - my)
    X1 = min(W, max(bx + bw, bbox[0] + bbox[2]) + mx)
    Y1 = min(H, max(by + bh, bbox[1] + bbox[3]) + my)
    if bw <= 0 or bh <= 0:
        return None
    reg = img[Y0:Y1, X0:X1]
    dark = reg.max(axis=2) <= FRAME_MAX
    _, lab = cv2.connectedComponents((~dark).astype(np.uint8), 4)
    sub = lab[by - Y0:by + bh - Y0, bx - X0:bx + bw - X0]
    vals, cnt = np.unique(sub[sub > 0], return_counts=True)
    if not len(vals):
        return None
    comp = lab == vals[cnt.argmax()]
    ys, xs = np.nonzero(comp)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if ((x0 == 0 and X0 > 0) or (y0 == 0 and Y0 > 0)
            or (x1 == reg.shape[1] and X1 < W)
            or (y1 == reg.shape[0] and Y1 < H)):
        return "escaped"
    if x0 == 0 or y0 == 0 or x1 == reg.shape[1] or y1 == reg.shape[0]:
        return None                 # runs off the page: no frame there
    if x1 - x0 < 60 or y1 - y0 < 25:
        return None
    solid = fill_holes(comp)
    if solid[y0:y1, x0:x1].mean() < 0.93:
        return None                 # not a rectangle: a balloon
    # everything dark inside the fill must be LETTERING: a drawn object
    # taller than any line of type means this "box" is a panel of art
    holes = (solid & ~comp)[y0:y1, x0:x1].astype(np.uint8)
    k, _, st, _ = cv2.connectedComponentsWithStats(holes, 8)
    if any(st[i, 3] > 70 for i in range(1, k)):
        return None
    # and a frame really is drawn round it, on all four sides
    sides = (dark[max(0, y0 - 6):y0, x0:x1].any(axis=0).mean(),
             dark[y1:y1 + 6, x0:x1].any(axis=0).mean(),
             dark[y0:y1, max(0, x0 - 6):x0].any(axis=1).mean(),
             dark[y0:y1, x1:x1 + 6].any(axis=1).mean())
    if min(sides) < 0.85:
        return None
    return (X0 + int(x0), Y0 + int(y0), X0 + int(x1), Y0 + int(y1))


def auto_caption_boxes(stem, img, bubbles, texts):
    """Find the page's caption boxes detection only part-found, and set
    each as ONE block of type. Returns the page's texts as the fit must
    read them.

    A box qualifies when its frame is found (`frame_box`) AND detection
    demonstrably lost part of it — at least 4 of the original letters
    inside the frame are outside every member's mask and outside any
    strip pair `caption_boxes` already wipes whole, or one line of it was
    cut into pieces side by side. Anything less is left to the existing
    machinery, which the shipped books were approved on: across both of
    them this finds nothing that a hand-measured `box` override does not
    already cover (and those overrides match the frames it finds to 2px).

    The box is then exactly what a `box` override makes of it: the first
    member with text carries the WHOLE box's text (the members' strings
    in order — one string per strip, or the whole caption on one strip
    and "" on the rest, read the same) and the box as its writing area,
    the other members typeset nothing, and the box is wiped whole."""
    out = list(texts)
    AUTO_MEMBERS[stem] = set()
    for key in [k for k in AUTO_OVERRIDES
                if k.startswith(override_key(stem, 0)[:-2])]:
        del AUTO_OVERRIDES[key]
    found = {}
    for bi, e in enumerate(bubbles, 1):
        if e["kind"] == "dark":
            continue
        r = frame_box(img, e["block"], e["bbox"])
        if r is not None:
            found.setdefault(r, None)
    pairs = [m for m in caption_boxes(bubbles,
                                      set(range(1, len(bubbles) + 1)))
             if len(m[1]) > 1]
    for (x0, y0, x1, y1) in found:
        mem = sorted(bi for bi, e in enumerate(bubbles, 1)
                     if x0 <= e["block"][0] + e["block"][2] / 2 < x1
                     and y0 <= e["block"][1] + e["block"][3] / 2 < y1)
        carriers = [i for i in mem if apply_fixes(texts[i - 1]).strip()]
        if not carriers or any(LAYOUT_OVERRIDES.get(override_key(stem, i))
                               for i in mem):
            continue
        # what detection covered: the members' masks, and any strip pair
        # the box pass wipes whole
        cov = np.zeros((y1 - y0, x1 - x0), bool)
        for i in mem:
            x, y, w, h = bubbles[i - 1]["bbox"]
            _mp = BUB / f"{stem}-b{i:02d}-mask.png"
            if not _mp.is_file():
                continue
            mk = cv2.imread(str(_mp), cv2.IMREAD_GRAYSCALE) > 127
            a0, b0 = max(x, x0), max(y, y0)
            a1, b1 = min(x + w, x1), min(y + h, y1)
            if a1 > a0 and b1 > b0:
                cov[b0 - y0:b1 - y0, a0 - x0:a1 - x0] |= \
                    mk[b0 - y:b1 - y, a0 - x:a1 - x]
        for (p0, q0, p1, q1), pm in pairs:
            if pm & set(mem):
                a0, b0 = max(p0, x0), max(q0, y0)
                a1, b1 = min(p1, x1), min(q1, y1)
                if a1 > a0 and b1 > b0:
                    cov[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = True
        cov = cv2.dilate(cov.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        ink = (img[y0:y1, x0:x1].max(axis=2) <= DARK_MAX).astype(np.uint8)
        k, lab, st, _ = cv2.connectedComponentsWithStats(ink, 8)
        lost = sum(1 for j in range(1, k)
                   if 9 <= st[j, 3] <= 48 and st[j, 4] >= 8
                   and cov[lab == j].mean() < 0.5)

        def _vov(a, c):
            ay, ah = bubbles[a - 1]["block"][1], bubbles[a - 1]["block"][3]
            cy, ch = bubbles[c - 1]["block"][1], bubbles[c - 1]["block"][3]
            return min(ay + ah, cy + ch) - max(ay, cy) > 0.5 * min(ah, ch)
        side = any(_vov(a, c) for a in mem for c in mem if a < c)
        # several entries in one frame that `caption_boxes` did not pair
        # (p3-013: the last line alone in a strip of its own, set apart by
        # a white band) are the same failure; a recognised strip pair is
        # left to the box pass it already gets
        loose = len(mem) > 1 and not any(set(mem) == pm for _, pm in pairs)
        if lost < 4 and not side and not loose:
            continue
        # the letterer's own cap height, off every letter in the frame:
        # read off a strip, or with the tint threshold, it comes out a size
        # or two short (p2-014 read 19px for letters 21-25px tall)
        caps = [st[j, 3] for j in range(1, k)
                if 9 <= st[j, 3] <= 48 and st[j, 4] >= 8]
        lead = carriers[0]
        parts = [apply_fixes(texts[i - 1]).strip() for i in carriers]
        # members that are each a whole sentence are separate lines of the
        # box ("LA CAPITAL DE KIROS." / "19 ROTACIONES MÁS TARDE.", 5-006),
        # and run together they read as one; strips cut mid-sentence are
        # one paragraph
        sep = ("\n" if all(re.search(r"[.!?…]\W*$", t.replace("*", ""))
                           for t in parts[:-1]) else " ")
        out[lead - 1] = sep.join(parts)
        for i in mem:
            if i != lead:
                out[i - 1] = ""
        AUTO_OVERRIDES[override_key(stem, lead)] = {
            "box": [x0, y0, x1, y1],
            "cap_h": float(np.median(caps)) if caps else 0.0}
        AUTO_MEMBERS[stem] |= set(mem)
        print(f"  {stem}: caption box {[x0, y0, x1, y1]} found from its "
              f"frame — b{lead:02d} carries members {mem} "
              f"({lost} letters outside detection)")
    return out


def prepare_bubble(stem, bi, b, img, texts, bubbles, cap_box, cap_mem):
    """Turn one detection entry into everything the fit AND the cleaning
    read: the mask as repaired, the entry as the TYPE sees it, the entry
    as the CLEANER sees it, the three row profiles, the cleaning band.

    THE single source of truth for that preparation. It used to live
    inline in main(), which meant reclean_all.py — the tool that proves
    a cleaning change is safe BOOK-WIDE — carried a hand-written copy,
    four blocks deep by the time this was extracted. Every new cleaning
    behaviour (the `lobes` override, sole_box, unfound_balloon) had to
    be mirrored there by hand, or the verification quietly stopped
    reproducing what the pipeline actually does.

    Returns None when the entry has no mask on disk. Otherwise a dict:
      mask        the mask as the CLEANING must see it
      b           the entry as the TYPE sees it (sole_box redirects it
                  to the box)
      b_clean     the entry as the CLEANER sees it — always the real
                  bbox, because the mask is bbox-sized
      rows / rows_fit / rows_raw / allowed / oldh / forced_lobes /
      sole_box / bbox
      jobs        the (bi, b, mask, allowed) cleaning jobs this entry
                  contributes — more than one when a `lobes` override
                  splits it into separate balloons
    """
    cap_members = set(cap_box)
    jobs = []
    x, y, w, h = b["bbox"]
    _mp = BUB / f"{stem}-b{bi:02d}-mask.png"
    if not _mp.is_file():
        return None
    mask = (cv2.imread(str(_mp),
                       cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
    # A mask that leaked past all recognition, rescued per-comic:
    # `lobes` names one hand-measured rectangle per balloon and each
    # balloon's interior is re-flooded inside it. Everything that
    # follows — the rows, the lobes, the CLEANING — then sees the
    # balloons and nothing else, so none of the repairs and clips
    # below (which all read the letterer's block, and this entry's
    # block spans both balloons) may touch it.
    forced_lobes = lobe_jobs = None
    _ovm = override_for(stem, bi)
    if _ovm and "lobes" in _ovm:
        got = lobes_from_rects(img, _ovm["lobes"], b["bbox"])
        if got is None:
            print(f"  !! {stem} b{bi:02d}: `lobes` override "
                  f"found no interior — ignored")
        else:
            mask, forced_lobes, lobe_jobs = got
    # two balloons side by side returned as one entry: find each from its
    # own lettering (letter_lobes), which also fixes their ORDER and
    # replaces a mask that leaked across the panel
    auto_lobes = False
    _npar = apply_fixes(texts[bi - 1]).count("\n\n") + 1
    if (forced_lobes is None and b["kind"] == "bubble"
            and bi not in cap_members):
        got = letter_lobes(img, b, mask, _npar,
                           [o["block"] for j, o in enumerate(bubbles, 1)
                            if j != bi],
                           apply_fixes(texts[bi - 1]).split("\n\n"))
        if got is not None:
            mask, forced_lobes, lobe_jobs = got
            auto_lobes = True
            print(f"  {stem} b{bi:02d}: balloon(s) re-found from the "
                  f"lettering ({len(forced_lobes)} lobe(s))")
    # give the balloon back what its own lettering cost the mask:
    # detection's barrier close welds a letter to an outline it comes
    # within 7px of, and the hole-fill then leaves a bite. Do it here,
    # before anything reads the mask, so the row profile, the lobes
    # and the CLEANING all see the same repaired balloon — text set
    # over a bite the cleaner never reached is the worst of all.
    # Caption boxes are exempt: their fill legitimately shifts hue, so
    # the flood's colour wall has nothing to hold on to, and the box
    # path drives its own profile anyway.
    if (b["kind"] == "bubble" and b.get("strict")
            and bi not in cap_members and forced_lobes is None):
        bxr, byr, bwr, bhr = b["block"]
        mask = repair_letter_bites(
            mask, img, (x, y),
            (max(0, bxr - x), max(0, byr - y),
             min(w, bxr + bwr - x), min(h, byr + bhr - y)))
    # A box whose text sits on THIS entry alone is one piece of
    # type: the whole box is its writing area, and the letterer
    # centred his type in the box, not in the strip detection
    # happened to find. Make the box this entry's block before
    # anything measures it — the cap height, the anchor ladder and
    # the per-line axes all read it. Anchored on the strip, p287
    # b03 set a 4-line caption at 14px in a box with room for 23,
    # and b08's three lines sat in the bottom two thirds of its box.
    _mem = cap_mem.get(bi, {bi})
    sole_box = (bi in cap_box
                and all(not apply_fixes(texts[j - 1]).strip()
                        for j in _mem if j != bi)
                and (len(_mem) == 1
                     or all(is_strip(bubbles[j - 1]) for j in _mem)))
    # the CLEANER must keep the real bbox: its mask is bbox-sized,
    # and a box whose sibling strip was never typeset is not re-formed
    # as a box at cleaning time, so the entry does get its own pass —
    # with a box-sized bbox that crashed on p231 (shapes (198,264) vs
    # (108,233)).
    b_clean = b
    if sole_box:
        _cx0, _cy0, _cx1, _cy1 = cap_box[bi]
        _ii = max(3, int(round(0.02 * (_cx1 - _cx0)))) + 3
        if (_cx1 - _cx0 > 2 * _ii + 20
                and _cy1 - _cy0 > 2 * _ii + 12):
            # the BOX is this entry's box in every sense — the
            # anchor ladder clips the block to the bbox, so leaving
            # the strip's bbox in place pulled the block's centre
            # back down into the strip (p287 b08 sat 17px low). The
            # mask and the row profile were read above, off the real
            # bbox, so this only redirects what measures the TYPE.
            b = dict(b, block=[_cx0 + _ii, _cy0 + _ii,
                               _cx1 - _cx0 - 2 * _ii,
                               _cy1 - _cy0 - 2 * _ii],
                     bbox=[_cx0, _cy0, _cx1 - _cx0, _cy1 - _cy0])
    oldh = old_cap_height(img, b)
    if sole_box and _ovm and _ovm.get("cap_h"):
        oldh = _ovm["cap_h"]        # measured on the whole framed box

    # clip the row profile to the bubble's own box: detection can
    # follow a row run out past the bbox (the same artwork that
    # pollutes the text block — see block_anchor), and the MASK is
    # bbox-sized, so a line placed out there lands on art that was
    # never cleaned
    rows = {}
    for r in b["rows"]:
        rx0, rx1 = max(x + r[1], x), min(x + r[2], x + w)
        if rx1 > rx0:
            rows[y + r[0]] = (rx0, rx1)
    if forced_lobes is not None:
        # detection's own profile is the leak we are replacing. The
        # per-row UNION spans both balloons, which is wrong for a
        # wrap and right for the cleaning band — and the wrap never
        # sees it: lobe_bands reports these lobes as DISJOINT, so
        # fit_lines steps the size down rather than falling through
        # to a whole-mask wrap across the gap.
        rows = {}
        for L in forced_lobes:
            for yy, sp in L.items():
                old = rows.get(yy)
                rows[yy] = sp if old is None else (min(old[0], sp[0]),
                                                   max(old[1], sp[1]))
    if b["kind"] in ("open", "margin"):
        # detection falls back to `open` when its flood fails on a
        # balloon that IS drawn, and then nothing bounds the new
        # type: p242's shout was set 141px wide in a 132px oval and
        # printed across the outline both sides. Where the balloon
        # can be found, it is the writing area.
        _bal = unfound_balloon(img, b)
        if _bal is not None:
            _clip = {yy: (max(sp[0], _bal[0]), min(sp[1], _bal[1]))
                     for yy, sp in rows.items()
                     if min(sp[1], _bal[1]) - max(sp[0], _bal[0]) > 8}
            if _clip:
                rows = _clip
    cap_rows = None
    if bi in cap_box:
        # A caption box's usable width is the BOX, not the ragged
        # mask letter_mask managed to find on a gradient fill — the
        # mask covered 250px of p101's 361px box, so the fit had a
        # fraction of the room and set the caption small and low.
        # Keep each entry's own vertical band so the two strips of a
        # split box do not land on top of each other.
        cbx0, cby0, cbx1, cby1 = cap_box[bi]
        ins = max(3, int(round(0.02 * (cbx1 - cbx0))))
        lo = max(cby0, min(rows) if rows else cby0)
        hi = min(cby1, (max(rows) + 1) if rows else cby1)
        # ...unless this entry carries the box's text ALONE — a box
        # half the detector never found (a `box` override), or a
        # split box whose text was transcribed as one string on the
        # first strip. Then the writing area is the whole box, and
        # holding the entry to its own band crammed the caption into
        # a 36px strip at 14px where the box had room for 17px
        # (p287 b03, b08).
        if sole_box:
            lo, hi = cby0 + ins, cby1 - ins
        if hi - lo >= 8 and cbx1 - cbx0 > 2 * ins + 20:
            rows = {yy: (cbx0 + ins, cbx1 - ins)
                    for yy in range(lo, hi)}
            cap_rows = dict(rows)
    allowed = None
    rows_raw = None
    rows_fit = None
    if b["kind"] == "bubble" and forced_lobes is not None:
        # a hand-bounded mask needs none of the repairs below, and
        # they would all read this entry's block — which spans both
        # balloons, so the fill flood would pick one of them and
        # resolve_rows would drop the other's rows entirely
        pad = max(6, int(round(w * 0.05)))
        allowed = set(range(min(rows) - pad, max(rows) + pad + 1))
    elif b["kind"] == "bubble":
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

        # The profile the FIT uses must follow the balloon's OWN
        # fill, in BOTH directions. Detection sometimes hands back a
        # mask that covers the balloon AND the page around it (a tail
        # that leaves the crop window), and its row profile is then a
        # RECTANGLE: lines get set where the oval has long since
        # narrowed, straight over the outline. And `rows` is the mask
        # ERODED, so it also carries every letter bite the mask does —
        # inside the letterer's block the fill gives those rows their
        # width back. Resolve BOTH profiles (the wrap's and the last
        # rung's) against it.
        bxf, byf, bwf, bhf = b["block"]
        regf = img[y:y + h, x:x + w]
        keepf = cv2.erode(mask, np.ones((3, 3), np.uint8))
        interf = bubble_interior(
            (regf.min(axis=2) <= DARK_MAX - 20).astype(np.uint8),
            keepf, (max(0, bxf - x), max(0, byf - y),
                    min(w, bxf + bwf - x), min(h, byf + bhf - y)),
            img=None if bi in cap_members else regf)
        solidf = fill_holes(interf)
        srows = {}
        for ry3 in range(h):
            run = widest_run(solidf[ry3])
            if run:
                srows[y + ry3] = (x + run[0], x + run[1])
        # detection's own profile clearance (PAD_FRAC of the width):
        # anything further in than that is a bite, not clearance
        pad_in = max(6, int(round(w * 0.03)))
        blk_rows = (byf, byf + bhf)

        def _resolve(src):
            return resolve_rows(src, srows, blk_rows, pad_in)
        cand = _resolve(rows)
        # only when the fill is a genuinely usable profile
        if len(cand) >= 12 and len(cand) >= 0.5 * len(rows):
            rows_fit = cand
            rr = _resolve(rows_raw)
            if len(rr) >= 12:
                rows_raw = rr
        if bi in cap_box and cap_rows:
            # the last rung sets per-line spans on the near-raw
            # profile. For a caption box that profile is the plain
            # RECTANGLE — clipping it to the fill (which the old
            # lettering breaks up) left the rung unable to place the
            # lines and held p101's caption at 16px with room for 19.
            rows_raw = dict(cap_rows)
            # ...and so is the profile the WRAP uses. The fill here
            # is flooded inside the MASK, which for a box detection
            # only half-found is the strip — resolving the box rows
            # against it dropped every row the strip does not cover,
            # so the box collapsed back to the strip and p287 b08's
            # three lines sat 17px low in it.
            rows_fit = dict(cap_rows)

    if lobe_jobs:
        # two balloons are cleaned as two: this entry's block spans
        # both and the artwork between them
        for _bb, _blk, _m in lobe_jobs:
            jobs.append(
                (bi, dict(b, bbox=_bb, block=_blk), _m,
                 set(range(_bb[1] - 6, _bb[1] + _bb[3] + 7))))
    else:
        jobs.append((bi, b_clean, mask, allowed))
    return {"mask": mask, "b": b, "b_clean": b_clean,
            "oldh": oldh, "rows": rows, "rows_fit": rows_fit,
            "rows_raw": rows_raw, "allowed": allowed,
            "forced_lobes": forced_lobes, "lobe_jobs": lobe_jobs,
            "auto_lobes": auto_lobes,
            "sole_box": sole_box, "bbox": (x, y, w, h),
            "jobs": jobs}


def collect_page(path, transcripts, all_pending):
    """Read one page's bubbles, prepare every entry with text and push
    the page onto `all_pending`. Returns early when the page has no
    bubbles file, no bubbles, or no transcripts yet.
    """
    bpath = BUB / (path.stem + ".json")
    if not bpath.is_file():
        return
    bubbles = json.loads(bpath.read_text())
    if not bubbles:
        return
    texts = transcripts.get(path.stem)
    if texts is None:
        print(f"  !! {path.stem}: {len(bubbles)} bubbles but no "
              f"transcripts — SKIPPED")
        return
    assert len(texts) == len(bubbles), \
        f"{path.stem}: {len(texts)} texts vs {len(bubbles)} bubbles"
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    texts = auto_caption_boxes(path.stem, img, bubbles, texts)
    page_entries = []
    pending = []
    clean_jobs = []
    all_pending.append((path, pending, clean_jobs, img, bubbles))
    cap_box, cap_mem, cap_members = page_caption_boxes(bubbles,
                                                       path.stem)
    for bi, b in enumerate(bubbles, 1):
        text = apply_fixes(texts[bi - 1])
        if not text.strip():
            continue  # "" = leave this region untouched (SFX, display)
        prep = prepare_bubble(path.stem, bi, b, img, texts, bubbles,
                              cap_box, cap_mem)
        mask, b, oldh = prep["mask"], prep["b"], prep["oldh"]
        rows, rows_fit = prep["rows"], prep["rows_fit"]
        rows_raw, sole_box = prep["rows_raw"], prep["sole_box"]
        forced_lobes = prep["forced_lobes"]
        x, y, w, h = prep["bbox"]
        clean_jobs.extend(prep["jobs"])

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
        if b["kind"] != "bubble":
            # A balloon is centred on the ORIGINAL LETTERER'S block (the
            # anchor ladder); a caption used the centroid of its own
            # rectangle instead. For a graded caption box those are not
            # the same place: `caption_boxes` finds the box by its FILL,
            # so the rectangle is only the coloured part and its centre
            # sits ABOVE where the letterer actually set the type — p42's
            # box centre is 175.5 against the letterer's 181.5, the
            # printed caption landed on rows 167-183 where the original
            # is on 174-189, and the 7px of flat colour left under it
            # read as a "yellow rectangle" for three rounds. The letterer
            # is the gold standard here as everywhere else.
            _, _bcy = block_anchor(b["block"], rows, b["bbox"])
            if _bcy is not None:
                cy = _bcy
        pending.append({"bi": bi, "b": b, "words": words, "hard": hard,
                        "paras": paras, "sole_box": sole_box,
                        "auto_lobes": prep["auto_lobes"],
                        "in_box": bi in cap_box,
                        "rows": (inset_rows(rows_fit or rows)
                                 if b["kind"] == "bubble" else rows),
                        "cys": [cy],
                        "oldh": oldh, "ov": override_for(path.stem, bi),
                        "porig": (orig_para_axes(img, b, len(paras) + 1)
                                  if paras else None),
                        "oaxis": (orig_axis_map(img, b)
                                  if b["kind"] == "bubble" else None),
                        "rows_raw": (inset_rows(rows_raw)
                                     if (b["kind"] == "bubble"
                                         and rows_raw) else rows_raw),
                        "lobes": ([inset_rows(L) for L in _lob]
                                  if (b["kind"] == "bubble" and paras
                                      and (_lob := (
                                          forced_lobes if forced_lobes
                                          and len(forced_lobes)
                                          == len(paras) + 1
                                          else mask_lobes(
                                              mask, x, y,
                                              len(paras) + 1))))
                                  else None)})

    print(f"{path.stem}: {len(pending)} bubbles collected")


def best_fit(fit, p, cap):
    """Largest size that fits, tightening tracking only if it has to.

    Sizes are tried from the top down and, at each, the LOOSEST tracking
    first — so a balloon with room is set untouched and only one the
    type just misses is tightened. A bigger size slightly tightened
    beats a smaller size set loose: that gap is the whole of "text
    small, not occupying the whole bubble".
    """
    for size in range(cap, MIN_SIZE, -1):
        for tr in track_steps(size):
            fit.track = tr
            lines = fit_lines(fit, p, size)
            if lines:
                fit.track = 0.0
                return size, lines, tr
    fit.track = 0.0
    return None, None, 0.0


def typeset_page(fit, path, pending, clean_jobs, img, bubbles, book,
                 ratio, layout, report):
    """Typeset and clean one page: per-lobe size caps, the fit, the
    layout entries, then the cleaning. Returns whether the page's
    pixels changed (a page whose bubbles all failed to fit is left
    alone rather than blanked).
    """
    page_entries = []
    fitted = set()
    # joined lobes size for their OWN balloon (lobe_size_caps): they
    # used to share the group's smallest maximum, which sat a pair at
    # 18px beside neighbours at 23px — the user's "text small, not
    # occupying the whole bubble" on p24/p79/p101.
    bubble_cap = {}
    gmax, gorig = {}, {}
    for p in pending:
        g = p["b"].get("group")
        if g is not None and p["max_size"]:
            gmax.setdefault(g, {})[p["bi"]] = p["max_size"]
            gorig.setdefault(g, {})[p["bi"]] = (
                int(round(p["oldh"] / ratio)) if p["oldh"] else 0)
    for g, maxes in gmax.items():
        bubble_cap.update(lobe_size_caps(maxes, gorig[g], book))
    # A CAPTION BOX IS ONE PIECE OF TYPE. A gradient-filled box comes
    # back as two strips, and sizing each on its own printed one box in
    # two sizes (p147: 21px over 19px, with the seam showing) — the
    # original letterer set the whole box at one size, and at the
    # ORIGINAL's size: a box has no slack at all (the fill IS the
    # writing area), so raising it to the book size is what pushed the
    # type to the frame. Share the tightest strip's maximum, ceilinged
    # at what the letterer himself used.
    box_cap = {}
    for _, members in caption_boxes(bubbles, {q["bi"] for q in pending}):
        mem = [q for q in pending if q["bi"] in members and q["max_size"]]
        if len(mem) < 2:
            continue
        orig = max((int(round(q["oldh"] / ratio)) if q["oldh"] else 0)
                   for q in mem)
        shared = min(min(q["max_size"] for q in mem), orig or MAX_SIZE)
        for q in mem:
            box_cap[q["bi"]] = shared
    for p in pending:
        bi, b, words, oldh = p["bi"], p["b"], p["words"], p["oldh"]
        cap = p["max_size"] or MAX_SIZE
        # the size ceiling for this KIND, before the whole-bubble
        # maximum narrows it. A compound balloon's maximum is the
        # tightest lobe's, so growing a roomier lobe has to measure
        # against the ceiling, not against that.
        ceiling = MAX_SIZE
        if b["kind"] in ("bubble", "dark", "open", "tint"):
            orig_px = int(round(oldh / ratio)) if oldh else 0
            ceiling = max(book, orig_px)
            cap = min(cap, ceiling)
        if bi in bubble_cap:
            cap = bubble_cap[bi]
        if bi in box_cap:
            cap = ceiling = box_cap[bi]
        ov = override_for(path.stem, bi)
        if ov and "size" in ov:
            cap = ceiling = ov["size"]
        size, lines, track = best_fit(fit, p, cap)
        if not size:
            print(f"  !! {path.stem} b{bi:02d}: no fit found")
            continue
        # each lobe of a compound balloon fills its OWN balloon: one
        # shared size is set by the tightest lobe and leaves the roomy
        # ones short of their own outline (p123's three bottom
        # balloons, one detection entry).
        per_line = None
        grown = grow_lobes(fit, p, size, max(cap, ceiling))
        if grown:
            lines = [ln for _, _, lk in grown for ln in lk]
            per_line = [(gs, gt) for gs, gt, lk in grown for _ in lk]
            size = max(gs for gs, _, _ in grown)
            track = next(gt for gs, gt, _ in grown if gs == size)
        lh = max(1, round(size * LINE_SPACING))
        sizes = [s for s, _ in per_line] if per_line else [size]
        newh = fit.cap_height(min(sizes))
        styles = {st for st, _ in words}
        out_lines = []
        for li, (ly, lx0, lx1, runs, cxa) in enumerate(lines):
            lsize, ltrack = per_line[li] if per_line else (size, track)
            fit.track = ltrack
            ln = {
                "y_top": round(ly, 1),
                "cx": round(line_cx(fit, lsize, lx0, lx1, runs, cxa), 1),
                "width": round(fit.line_width(runs, lsize), 1),
                "runs": consolidate(runs),
            }
            if per_line and (lsize != size or ltrack != track):
                # GIMP falls back to the entry's values without these
                ln["font_px"] = lsize
                ln["line_height"] = max(1, round(lsize * LINE_SPACING))
                ln["track"] = round(ltrack, 3)
            out_lines.append(ln)
        fit.track = 0.0
        entry = {
            "index": bi, "kind": b["kind"],
            "color": "white" if b["kind"] == "dark" else "black",
            "font_px": size, "line_height": lh,
            "uniform": list(styles)[0] if len(styles) == 1 else None,
            "old_cap_h": oldh, "new_cap_h": newh,
            "track": round(track, 3),
            "lines": out_lines,
        }
        page_entries.append(entry)
        fitted.add(bi)
        report.append((path.stem[-5:], bi, oldh, newh,
                       newh / oldh if oldh else 0))
    # A CAPTION BOX IS ONE BLOCK OF TYPE. Its strips are an artefact of
    # the fill crossing LIGHT_MIN, and centring each one in its own strip
    # printed a blank line at the seam — "too much space between
    # paragraphs, as if it were 2 bubbles not one" (p147's bottom-left
    # box, p219). Put every line of the box on ONE grid, centred on the
    # letterer's own block, exactly as he set it. A `sole_box` entry is
    # already laid out over the whole box, so it is left alone.
    _sole = {q["bi"] for q in pending if q.get("sole_box")}
    byi = {e["index"]: e for e in page_entries}
    for _, members in caption_boxes(bubbles, fitted):
        mem = sorted((m for m in members if m in byi and m not in _sole),
                     key=lambda m: (bubbles[m - 1]["block"][1],
                                    bubbles[m - 1]["block"][0]))
        if len(mem) < 2:
            continue
        lhs = {byi[m]["line_height"] for m in mem}
        if len(lhs) != 1:
            continue        # not one piece of type after all
        blocks = [bubbles[m - 1]["block"] for m in mem]
        cyb = (min(bl[1] for bl in blocks)
               + max(bl[1] + bl[3] for bl in blocks)) / 2.0
        ys = stack_box_lines([(m, len(byi[m]["lines"])) for m in mem],
                             lhs.pop(), cyb)
        for m in mem:
            for ln, yy in zip(byi[m]["lines"], ys[m]):
                ln["y_top"] = round(yy, 1)
    layout[path.stem] = page_entries
    # clean ONLY bubbles that got a layout entry (a fit failure must
    # never blank a bubble)
    changed = False
    boxes = boxes_to_wipe(bubbles, path.stem, fitted)
    in_box = {bi for _, mem in boxes for bi in mem}
    for bi, b, mask, allowed in clean_jobs:
        if bi in fitted and bi not in in_box:
            clean_bubble(img, b, mask, allowed)
            changed = True
    # A coloured caption box the gradient split into two entries is
    # cleaned as ONE box, in place of its members' per-entry passes.
    # Each strip's mask is ragged (it follows the fill, not the box), so
    # letters straddle the mask edge and the band at the junction is left
    # under the new line. Both strips are typeset, so wiping the whole
    # box is safe. `box` in an override covers a box detection did not
    # split but only half-masked.
    for box, _ in boxes:
        clean_caption_box(img, box)
        changed = True
    if changed:
        out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 95,
                     int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                     cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    return changed, page_entries


def main():
    transcripts = json.loads((WORK / "transcripts.json").read_text())
    fit = Fitter()
    layout = {}
    report = []
    all_pending = []
    for path in sorted(PAGES_DIR.glob("*.jpg")) + sorted(PAGES_DIR.glob("*.png")):
        collect_page(path, transcripts, all_pending)

    # ---- global fitting ----
    # per-bubble maxima (margin/open notes stay near their original size).
    # This pass is the long one: every bubble searches down until a size
    # fits, and the doomed large sizes cost the most, so it reports progress
    # rather than going silent for a very long time.
    #
    # With the book size PINNED, every dialogue size is capped below at
    # max(book, the original's own size) anyway, so probing above that is
    # pure waste — it is what made a subset re-fit take 37 minutes for 84
    # pages. Unpinned the search must still run free: the book size IS the
    # percentile of these maxima, and capping them would bias it.
    ratio0 = fit.cap_height(100) / 100.0

    def search_cap(p):
        if p["b"]["kind"] == "margin" and p["oldh"]:
            return max(MIN_SIZE + 1, int(round(p["oldh"] / ratio0 * 1.1)))
        if BOOK_PIN is not None and p["b"]["kind"] in (
                "bubble", "dark", "open", "tint"):
            orig = int(round(p["oldh"] / ratio0)) if p["oldh"] else 0
            return min(MAX_SIZE, max(BOOK_PIN, orig))
        return MAX_SIZE
    n_total = sum(len(pen) for _, pen, _, _, _ in all_pending)
    n_done, t_start = 0, time.time()
    print(f"\nsizing {n_total} bubbles (searching down from {MAX_SIZE}px"
          f"{', capped at the pinned book size' if BOOK_PIN else ''})",
          flush=True)
    for _, pending, _, _, _ in all_pending:
        for p in pending:
            p["max_size"], _, _ = best_fit(fit, p,
                                            search_cap(p))
            n_done += 1
            if n_done % 25 == 0 or n_done == n_total:
                el = time.time() - t_start
                eta = el / n_done * (n_total - n_done)
                print(f"  sizing {n_done}/{n_total} "
                      f"({100 * n_done // n_total}%) — {el / 60:.0f} min "
                      f"elapsed, ~{eta / 60:.0f} min left", flush=True)

    # the book's common size: a low percentile of the dialogue maxima, so
    # nearly all bubbles carry the same size and only the densest shrink
    dialogue_max = [p["max_size"] for _, pen, _, _, _ in all_pending for p in pen
                    if p["b"]["kind"] == "bubble" and p["max_size"]]
    book = (BOOK_PIN if BOOK_PIN is not None
            else int(np.percentile(dialogue_max, BOOK_PCTL)))
    print(f"\nbubble maxima: {sorted(dialogue_max)}")
    print(f"book size = {'pinned' if BOOK_PIN is not None else 'p%d' % BOOK_PCTL}"
          f" = {book}px "
          f"({sum(m >= book for m in dialogue_max)}/{len(dialogue_max)} "
          f"bubbles at full size)")

    ratio = fit.cap_height(100) / 100.0
    n_pages = len(all_pending)
    t_start = time.time()
    print(f"\nfitting and cleaning {n_pages} pages", flush=True)
    for n_page, (path, pending, clean_jobs, img, bubbles) in enumerate(
            all_pending, 1):
        changed, page_entries = typeset_page(fit, path, pending,
                                             clean_jobs, img, bubbles,
                                             book, ratio, layout, report)
        el = time.time() - t_start
        eta = el / n_page * (n_pages - n_page)
        print(f"  [{n_page}/{n_pages}] {path.stem}: "
              f"{len(page_entries)} bubbles typeset"
              f"{'' if changed else ' (page unchanged)'}"
              f" — ~{eta / 60:.0f} min left", flush=True)

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
    if COMIC is None:
        sys.exit('usage: reletter_fit.py "<comic folder name>" [--book N]')
    main()
