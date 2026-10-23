#!/usr/bin/env python3
"""Keep the original lettering or re-letter it: measure every issue after
detection, BEFORE transcription, and write relettering/<comic>/
issue_decisions.json once (the rule and the files: issue_plan.py).

Two gates, both read off the original's letter blobs, so no transcript is
needed:
  * SIZE — the original's dialogue cap height as it PRINTS: pages are
    placed 1:1 on the template, so its pixels convert to mm at the
    template's resolution.
  * FONT — letter-shape statistics (width, stroke, slant, lowercase)
    against the same statistics of our font rendered at the same height.
Reported beside them, deciding nothing: how much taller OUR letters would
come out. The words are unknown, but their SHAPE is not: the blobs give
the lines, the word breaks and roughly how many letters each word has.
Placeholder text of that shape goes through OUR REAL FIT (the same
prepare_bubble, row profiles, tracking and hyphenation), so our font's
width and the balloon's shape are counted exactly as the fit will.

    reletter_decide.py "<comic folder name>"
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reletter_fit as F  # noqa: E402  (auto-configures from argv)
from make_sheets import issue_of  # noqa: E402

WORD_GAP = 0.30     # a gap wider than this x cap height starts a word
LINE_TOL = 0.6      # a box joins a line within this x the line's height
# a word whose strokes run this much heavier than its balloon's median is
# lettered bold (measured: bold words 1.12-1.14x, regular 0.98-0.99x)
BOLD_STROKE = 1.08
# The fit's book size is the 35th percentile (BOOK_PCTL) of the maxima of
# the REAL text; placeholder text fits a little more easily (fewer legal
# break points to fight, no punctuation), so its maxima run a hair high.
# Measured on six fitted comics: at 35 the estimated book was 1px high on 3
# of 6; at 30 on 1 (and high = the safe side: re-letter); at 28 a book came
# out low and an issue that must be re-lettered was kept.
BOOK_ESTIMATE_PCTL = 30
# cycled through for the placeholder words: an alphabet's average advance,
# no language's letter frequencies
PLACEHOLDER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ---- the text's shape, from the original's letter blobs --------------------

def blob_lines(boxes):
    """Letter boxes (x, y, w, h) -> lines top to bottom, each left to
    right. A box joins the line whose centre it shares (within LINE_TOL x
    that line's median height)."""
    lines = []          # [centre, [boxes]]
    for b in sorted(boxes, key=lambda b: b[1] + b[3] / 2):
        cy = b[1] + b[3] / 2
        for ln in lines:
            hs = np.median([q[3] for q in ln[1]])
            if abs(cy - ln[0]) <= LINE_TOL * hs:
                ln[1].append(b)
                ln[0] = float(np.median([q[1] + q[3] / 2 for q in ln[1]]))
                break
        else:
            lines.append([cy, [b]])
    lines.sort(key=lambda ln: ln[0])
    return [sorted(ln[1], key=lambda b: b[0]) for ln in lines]


def line_words(line, cap):
    """One line's boxes grouped into words: a gap wider than WORD_GAP x cap
    starts a word."""
    words, right = [], None
    for b in sorted(line, key=lambda b: b[0]):
        if right is None or b[0] - right > WORD_GAP * cap:
            words.append([b])
        else:
            words[-1].append(b)
        right = b[0] + b[2] if right is None else max(right, b[0] + b[2])
    return words


def letters_in(word, ratio):
    """A blob holds as many letters as its width allows (touching letters
    merge into one blob), measured against ITS OWN height x the page's
    single-letter width/height `ratio`: a tight balloon's small type on a
    page of bigger type was under-counted by a page-wide letter width (22
    characters came back as 5 letters)."""
    return sum(max(1, int(round(w / (ratio * h)))) if ratio > 0 else 1
               for _, _, w, h in word)


def word_lengths(line, ratio, cap):
    """Letters per word along one line."""
    return [letters_in(w, ratio) for w in line_words(line, cap)]


def placeholder_text(boxes, paragraphs, ratio, strokes=None):
    """Placeholder text of the original's shape: its lines, its word
    lengths, `paragraphs` paragraphs split at the widest line gaps, and —
    given each box's stroke (`box_stroke`) — its emphasis: a word drawn at
    least BOLD_STROKE x the balloon's median stroke is set *bold*."""
    if not boxes:
        return ""
    cap = float(np.median([b[3] for b in boxes]))
    sk = dict(zip(boxes, strokes)) if strokes is not None else {}
    med = float(np.median(strokes)) if strokes is not None else 0.0
    lines = blob_lines(boxes)
    k = 0
    texts = []
    for ln in lines:
        words = []
        for wb in line_words(ln, cap):
            n = letters_in(wb, ratio)
            word = "".join(PLACEHOLDER[(k + i) % len(PLACEHOLDER)]
                           for i in range(n))
            k += n
            if med > 0 and np.median([sk[b] for b in wb]) >= BOLD_STROKE * med:
                word = f"*{word}*"
            words.append(word)
        texts.append(" ".join(words))
    splits = set()
    if paragraphs > 1 and len(lines) > 1:
        gaps = [(min(b[1] for b in lines[i + 1])
                 - max(b[1] + b[3] for b in lines[i]), i)
                for i in range(len(lines) - 1)]
        splits = {i for _, i in sorted(gaps, reverse=True)[:paragraphs - 1]}
    out = texts[0]
    for i in range(1, len(texts)):
        out += ("\n\n" if i - 1 in splits else " ") + texts[i]
    return out


def box_stroke(ink, box):
    """Stroke thickness of one letter box, as a fraction of its height."""
    x, y, w, h = box
    c = ink[y:y + h, x:x + w].astype(np.uint8)
    cs, _ = cv2.findContours(c, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(q, True) for q in cs)
    return c.sum() / (per / 2.0) / h if per > 0 else 0.0


def entry_boxes(img, b):
    """The original's letter boxes of an entry: wholly inside its block
    grown by half a letter height (within the bbox) — detection's block can
    stop a few px short of the lettering, and a letter crossing its edge
    was dropped. White-on-black captions are read off the inverted page."""
    src = 255 - img if b["kind"] == "dark" else img
    first = F.block_letters(src, b["block"])
    pad = int(round(0.5 * (np.median([q[3] for q in first]) if first
                           else 15)))
    bx, by, bw, bh = b["block"]
    X, Y, W, H = b.get("bbox", b["block"])
    x0, y0 = max(X, bx - pad), max(Y, by - pad)
    x1, y1 = min(X + W, bx + bw + pad), min(Y + H, by + bh + pad)
    if x1 <= x0 or y1 <= y0:
        return first
    return F.block_letters(src, [x0, y0, x1 - x0, y1 - y0])


def page_texts(img, bubbles):
    """One placeholder string per detected entry ("" where no letters)."""
    boxes = [entry_boxes(img, b) for b in bubbles]
    single = [q[2] / q[3] for bb in boxes for q in bb if q[2] <= 1.3 * q[3]]
    ratio = float(np.median(single)) if single else 0.0
    ink = img.min(axis=2) <= F.DARK_MAX
    white = (255 - img).min(axis=2) <= F.DARK_MAX   # as entry_boxes reads it
    out = []
    for bb, b in zip(boxes, bubbles):
        src = white if b["kind"] == "dark" else ink
        strokes = [box_stroke(src, q) for q in bb]
        out.append(placeholder_text(bb, b.get("paragraphs", 1) or 1,
                                    ratio, strokes))
    return out


# ---- the size gate ---------------------------------------------------------

MM_PER_IN = 25.4


def print_res():
    """Pixels per inch the pages print at (placed 1:1 on the template):
    the compose step's cached template geometry."""
    geo = json.loads((F.REPO / "template_geometry.json").read_text())
    return float(geo.get("res", 300.0))


def issue_caps_mm(ps, res):
    """{issue: median ORIGINAL dialogue cap height, in printed mm}."""
    caps = defaultdict(list)
    for p in ps:
        caps[issue_of(p["stem"])].append(p["oldh"])
    return {issue: round(float(np.median(hs)) * MM_PER_IN / res, 3)
            for issue, hs in caps.items()}


# ---- our predicted size (reported, not decisive) ---------------------------

def search_cap(F, fit, p):
    """Where a balloon's size search starts. The real fit searches from
    MAX_SIZE because the book size IS the percentile of these maxima; for
    the decision only min(max, max(book, original)) matters, and the book
    size sits far below twice the original's size."""
    orig = int(round(p["oldh"] / (fit.cap_height(100) / 100.0)))
    return min(F.MAX_SIZE, max(2 * orig, 30))


def page_entries(F, path, img, bubbles):
    """One page's dialogue balloons prepared by the real fit for their
    placeholder text (collect_page's own entries, each tagged with its
    stem). The page image is not kept."""
    def collect(texts):
        pending, out = [], []
        F.collect_page(path, {path.stem: texts}, pending)
        for _, entries, _, _, _ in pending:
            for p in entries:
                if p["b"]["kind"] == "bubble" and p["oldh"]:
                    p["stem"] = path.stem
                    out.append(p)
        return out
    texts = page_texts(img, bubbles)
    try:
        return collect(texts)
    except Exception:
        # every entry gets text here, including ones a transcript leaves ""
        # (SFX, art) that the fit was never written to prepare: one entry
        # it cannot prepare costs only itself, never the page
        out = []
        for i, t in enumerate(texts):
            if not t:
                continue
            only = [""] * len(texts)
            only[i] = t
            try:
                out += collect(only)
            except Exception as e:
                print(f"  !! {path.stem} b{i + 1:02d}: not measurable "
                      f"({type(e).__name__}: {e})")
        return out


def size_entries(F, fit, ps, sample=5, progress=False):
    """Size every balloon as the fit would set it, in two passes. The book
    size is the percentile of the per-balloon maxima, estimated on every
    `sample`-th balloon (searched from search_cap); then each balloon only
    needs min(its max, max(book, original)), so its search starts THERE —
    the first size tried usually fits (the pinned-book speed-up).
    -> (rows {stem, bi, oldh, max_size}, book)."""
    import time
    ratio0 = fit.cap_height(100) / 100.0
    t0, total = time.time(), len(ps[::sample]) + len(ps)
    done = 0

    def tick():
        nonlocal done
        done += 1
        if progress and (done % 100 == 0 or done == total):
            el = time.time() - t0
            print(f"  sizing {done}/{total} ({100 * done // total}%) — "
                  f"{el / 60:.0f} min elapsed, "
                  f"~{el / done * (total - done) / 60:.0f} min left",
                  flush=True)
    first = {}
    for i in range(0, len(ps), sample):
        first[i], _, _ = F.best_fit(fit, ps[i], search_cap(F, fit, ps[i]))
        tick()
    got = [m for m in first.values() if m]
    book = int(np.percentile(got, BOOK_ESTIMATE_PCTL)) if got else None
    rows = []
    for i, p in enumerate(ps):
        orig = int(round(p["oldh"] / ratio0))
        if book is None:
            cap = search_cap(F, fit, p)
        else:
            cap = min(F.MAX_SIZE, max(book, orig))
        if i in first and (first[i] is None or first[i] <= cap):
            size = first[i]
        else:
            size, _, _ = F.best_fit(fit, p, cap)
        tick()
        rows.append({"stem": p["stem"], "bi": p["bi"], "oldh": p["oldh"],
                     "max_size": size})
    return rows, book


def load_page(F, path):
    """(image, bubbles) of a page with detected entries, else None."""
    bpath = F.BUB / (path.stem + ".json")
    if not bpath.is_file():
        return None
    bubbles = json.loads(bpath.read_text())
    if not bubbles:
        return None
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), bubbles


def predict_bubbles(F, fit, pages):
    """Every dialogue balloon's largest fitting size for its placeholder
    text, page by page (the fit holds every page image; this need not)."""
    ps = []
    for path in pages:
        page = load_page(F, path)
        if page is not None:
            ps += page_entries(F, path, *page)
    return size_entries(F, fit, ps)[0]


def apply_book(rows, fit, book=None):
    """The size each balloon would print at, as the fit sets it: the book
    size (BOOK_PCTL of the maxima), dialogue clamped at max(book, the
    original's own size). -> {issue: [new/old cap height ratios]}."""
    sized = [r for r in rows if r["max_size"]]
    if not sized:
        return {}
    if book is None:
        book = int(np.percentile([r["max_size"] for r in sized],
                                 F.BOOK_PCTL))
    ratio0 = fit.cap_height(100) / 100.0
    out = defaultdict(list)
    for r in sized:
        orig = int(round(r["oldh"] / ratio0))
        size = min(r["max_size"], max(book, orig))
        out[issue_of(r["stem"])].append(fit.cap_height(size) / r["oldh"])
    return dict(out)


# ---- the font gate ---------------------------------------------------------

FONT_MIN_LETTERS = 6
FONT_MIN_BALLOONS = 5   # an issue measured on fewer balloons is "unsure"
# Each statistic is taken as THEIRS - OURS, per balloon, and an issue is
# judged on the MEDIAN of those differences (a few special voices — a
# droid, a storybook serif — cannot move a median).
# FONT_TOL: how far an issue's median may sit from ours and still be "the
# same kind of lettering"; twice that on any statistic = clearly different.
# Measured on the calibration run (qa-tools/decide_calibrate.py): every
# issue the user judged similar sits inside these, at most width 0.098,
# stroke 0.018, slant 0.100, lower 0.096 away.
FONT_TOL = {"width": 0.12, "stroke": 0.025, "slant": 0.12, "lower": 0.12}
# ...and the hand the user judged different is WIDER and MORE UPRIGHT than
# ours AT ONCE (width +0.051..+0.114, slant -0.061..-0.118), while every
# similar issue that is as wide leans the other way and every one as upright
# is narrower: each statistic alone overlaps, the pair separates.
WIDER, UPRIGHT = 0.03, 0.04


def letter_crops(img, b):
    """Boolean ink crops of the original's letters inside an entry's block."""
    src = 255 - img if b["kind"] == "dark" else img
    ink = src.min(axis=2) <= F.DARK_MAX
    return [ink[y:y + h, x:x + w] for x, y, w, h in entry_boxes(img, b)]


def shape_stats(crops):
    """What a lettering hand looks like, scale-free: median width / height
    of single letters, stroke thickness / height, slant (a right lean is
    positive) and the share of short letters (lowercase). None under
    FONT_MIN_LETTERS letters."""
    crops = [c for c in crops if c.size and c.any()]
    if len(crops) < FONT_MIN_LETTERS:
        return None
    hs = np.array([c.shape[0] for c in crops], float)
    mh = float(np.median(hs))
    single = [c for c in crops if c.shape[1] <= 1.3 * mh] or crops
    width = float(np.median([c.shape[1] / c.shape[0] for c in single]))
    strokes, slants = [], []
    for c in single:
        u8 = c.astype(np.uint8)
        cs, _ = cv2.findContours(u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        per = sum(cv2.arcLength(q, True) for q in cs)
        if per > 0:
            strokes.append(u8.sum() / (per / 2.0) / c.shape[0])
        ys, xs = np.nonzero(c)
        yu = (c.shape[0] - 1 - ys).astype(float)       # y pointing up
        if yu.var() > 0:
            slants.append(float(np.cov(xs, yu)[0, 1] / yu.var()))
    return {"width": width,
            "stroke": float(np.median(strokes)) if strokes else 0.0,
            "slant": float(np.median(slants)) if slants else 0.0,
            "lower": float((hs < 0.8 * mh).mean())}


def render_glyphs(fit, size, chars=PLACEHOLDER):
    """Our regular font's glyphs at `size` px of em, as boolean ink crops."""
    from PIL import Image, ImageDraw
    font = fit.font("regular", size)
    out = []
    for ch in chars:
        im = Image.new("L", (size * 3, size * 3), 0)
        ImageDraw.Draw(im).text((size, size), ch, font=font, fill=255)
        a = np.array(im) > 127
        ys, xs = np.nonzero(a)
        if len(ys):
            out.append(a[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    return out


_REF = {}


def font_reference(fit, cap_px):
    """shape_stats of our font with its cap height at `cap_px`."""
    size = max(8, int(round(cap_px / fit.cap_ratio())))
    if size not in _REF:
        _REF[size] = shape_stats(render_glyphs(fit, size))
    return _REF[size]


def font_delta(theirs, ours):
    """THEIRS - OURS per statistic."""
    return {k: theirs[k] - ours[k] for k in FONT_TOL}


def median_delta(deltas):
    return {k: float(np.median([d[k] for d in deltas])) for k in FONT_TOL}


def issue_font(deltas):
    """An issue's font from its balloons' differences to ours: "different"
    when the median is a wider, more upright hand, or any statistic is
    twice its tolerance away; "similar" when every one is within it;
    "unsure" in between or on too few balloons."""
    if len(deltas) < FONT_MIN_BALLOONS:
        return "unsure"
    m = median_delta(deltas)
    if m["width"] >= WIDER and m["slant"] <= -UPRIGHT:
        return "different"
    off = {k: abs(m[k]) / FONT_TOL[k] for k in FONT_TOL}
    if any(v >= 2 for v in off.values()):
        return "different"
    if all(v <= 1 for v in off.values()):
        return "similar"
    return "unsure"


def font_verdicts(fit, img, bubbles):
    """Per measurable dialogue balloon: its differences to our font at its
    own cap height."""
    out = []
    for b in bubbles:
        if b["kind"] != "bubble":
            continue
        crops = letter_crops(img, b)
        theirs = shape_stats(crops)
        if theirs is None:
            continue
        ours = font_reference(fit, np.median([c.shape[0] for c in crops]))
        out.append(font_delta(theirs, ours))
    return out


# ---- the decision ----------------------------------------------------------

def assess(pages_dir, detail=None):
    """Measure every issue of the configured comic from its pages in
    `pages_dir` (the PRISTINE copies) -> {issue: decision record}. Writes
    nothing. `detail` (a dict), if given, receives the per-balloon sizing
    rows and the estimated book size (for the calibration tool)."""
    from issue_plan import decide_issue
    import time
    fit = F.Fitter()
    paths = sorted(p for p in Path(pages_dir).iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    ps, deltas = [], defaultdict(list)
    entries, pages = defaultdict(int), defaultdict(list)
    t0 = time.time()
    for n, path in enumerate(paths, 1):
        issue = issue_of(path.stem)
        pages[issue].append(path.stem)
        page = load_page(F, path)
        if page is not None:
            img, bubbles = page
            entries[issue] += len(bubbles)
            ps += page_entries(F, path, img, bubbles)
            deltas[issue] += font_verdicts(fit, img, bubbles)
        if n % 10 == 0 or n == len(paths):
            el = time.time() - t0
            print(f"  measure {n}/{len(paths)} ({100 * n // len(paths)}%) — "
                  f"{el / 60:.0f} min elapsed, "
                  f"~{el / n * (len(paths) - n) / 60:.0f} min left",
                  flush=True)
    caps = issue_caps_mm(ps, print_res())
    rows, book = size_entries(F, fit, ps, progress=True)
    ratios = apply_book(rows, fit, book)
    if detail is not None:
        detail.update(rows=rows, book=book)
    out = {}
    for issue in sorted(pages):
        rs = ratios.get(issue)
        ratio = round(float(np.median(rs)), 3) if rs else None
        font = issue_font(deltas[issue])
        rec = decide_issue(font, caps.get(issue), entries[issue])
        rec.update({
            "font": font, "orig_cap_mm": caps.get(issue),
            "size_ratio": ratio,
            "balloons": sum(1 for r in rows if issue_of(r["stem"]) == issue),
            "font_balloons": len(deltas[issue]),
            "font_delta": ({k: round(v, 3) for k, v in
                            median_delta(deltas[issue]).items()}
                           if deltas[issue] else None),
            "pages": pages[issue]})
        out[issue] = rec
    return out


def main():
    from issue_plan import DECISIONS, report_lines
    work = F.WORK
    if (work / DECISIONS).is_file():
        print(f"done: {work / DECISIONS} exists — decisions are never "
              "recomputed; change one in issue_overrides.json")
        return 0
    if ((work / "layout.json").is_file()
            or (work / "sheets" / "manifest.json").is_file()):
        print("legacy comic (sheets made before this step existed): every "
              "issue re-lettered, nothing written")
        return 0
    decisions = assess(work / "pristine")
    (work / DECISIONS).write_text(
        json.dumps(decisions, ensure_ascii=False, indent=1))
    for line in report_lines(work):
        print(line)
    return 0


if __name__ == "__main__":
    if F.COMIC is None:
        sys.exit('usage: reletter_decide.py "<comic folder name>"')
    sys.exit(main())
