"""Keep-or-re-letter measurements (relettering/reletter_decide.py): the
placeholder text rebuilt from the original's letter blobs, the size
prediction and the font gate. Need cv2 (.venv-reletter); skip otherwise.

    .venv-reletter/bin/python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

# a processed comic to run the real-page test against (read-only), e.g.
# RELETTER_TEST_COMIC="<comic folder name>"; skipped when unset
REAL_COMIC = os.environ.get("RELETTER_TEST_COMIC", "")


def load():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_fit
    if reletter_fit.COMIC is None:
        reletter_fit.configure("unit-test-comic")
    import reletter_decide
    return reletter_decide, reletter_fit


def row(y, xs, h=20, w=10):
    return [(x, y, w, h) for x in xs]


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class Placeholder(unittest.TestCase):
    def test_blob_lines_groups_rows_and_sorts_by_x(self):
        D, _ = load()
        boxes = row(40, [24, 0, 12]) + row(10, [50, 0])
        lines = D.blob_lines(boxes)
        self.assertEqual(len(lines), 2)
        self.assertEqual([b[0] for b in lines[0]], [0, 50])
        self.assertEqual([b[0] for b in lines[1]], [0, 12, 24])

    def test_word_lengths_split_on_wide_gaps(self):
        D, _ = load()
        # ratio = the page's single-letter width / height (10 / 20)
        self.assertEqual(
            D.word_lengths(row(0, [0, 12, 24, 50, 62]), 0.5, 20), [3, 2])

    def test_merged_blob_counts_its_letters(self):
        D, _ = load()
        self.assertEqual(D.word_lengths([(0, 0, 30, 20)], 0.5, 20), [3])

    def test_small_lettering_counts_by_its_own_height(self):
        """A tight balloon's small type on a page of bigger type: counting
        by the PAGE's letter width under-counted it (a 22-character balloon
        came back as 5 letters and fit 20 sizes too big)."""
        D, _ = load()
        self.assertEqual(D.word_lengths([(0, 0, 50, 10)], 0.5, 10), [10])

    def test_placeholder_text(self):
        D, _ = load()
        boxes = row(0, [0, 12, 24, 50, 62]) + row(30, [0, 12, 24, 50, 62])
        self.assertEqual(D.placeholder_text(boxes, 1, 0.5), "ABC DE FGH IJ")

    def test_paragraphs_split_at_the_widest_line_gap(self):
        D, _ = load()
        boxes = (row(0, [0, 12]) + row(30, [0, 12]) + row(150, [0, 12]))
        self.assertEqual(D.placeholder_text(boxes, 2, 0.5), "AB CD\n\nEF")

    def test_heavy_strokes_set_the_word_bold(self):
        """Emphasis is lettered bold, and our bold-italic sets ~12% wider:
        a placeholder without it fits a size the real text cannot."""
        D, _ = load()
        boxes = row(0, [0, 12, 40, 52, 64, 76])
        strokes = [0.20, 0.20, 0.14, 0.14, 0.14, 0.14]
        self.assertEqual(D.placeholder_text(boxes, 1, 0.5, strokes),
                         "*AB* CDEF")
        self.assertEqual(D.placeholder_text(boxes, 1, 0.5, [0.14] * 6),
                         "AB CDEF")

    def test_no_letters_no_text(self):
        D, _ = load()
        self.assertEqual(D.placeholder_text([], 1, 0.5), "")

    def test_letters_crossing_the_block_edge_are_kept(self):
        """Detection's block can be a few px short of the lettering; a
        letter crossing its edge is still this balloon's."""
        D, F = load()
        img = np.full((200, 300, 3), 255, np.uint8)
        for x in (60, 80, 100, 120):
            cv2.rectangle(img, (x, 90), (x + 10, 110), (0, 0, 0), 3)
        # the middle two lie inside the block, the outer two cross its
        # left and right edges by a px or two
        b = {"kind": "bubble", "block": [62, 88, 66, 24],
             "bbox": [30, 60, 140, 80]}
        self.assertEqual(len(F.block_letters(img, b["block"])), 2)
        self.assertEqual(len(D.entry_boxes(img, b)), 4)


class FakeFit:
    def cap_height(self, size):
        return float(size)          # cap ratio 1.0: orig px == oldh


class FakeF:
    MAX_SIZE, BOOK_PCTL = 80, 35

    def __init__(self):
        self.caps = []

    def best_fit(self, fit, p, cap):
        self.caps.append(cap)
        return min(cap, p["true"]), None, 0.0


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class UnmeasurableEntry(unittest.TestCase):
    """Every entry gets placeholder text, including ones a transcript would
    leave "" (SFX, art) — the fit was never written to prepare those, and
    one that cannot be prepared must cost only itself, not the page."""

    def test_a_failing_entry_is_skipped_not_the_page(self):
        D, _ = load()

        class F:
            @staticmethod
            def collect_page(path, transcripts, pending):
                texts = transcripts[path.stem]
                if texts[1]:
                    raise ValueError("min() iterable argument is empty")
                pending.append((path, [
                    {"bi": i + 1, "b": {"kind": "bubble"}, "oldh": 18.0}
                    for i, t in enumerate(texts) if t], [], None, None))
        saved = D.page_texts
        D.page_texts = lambda img, bubbles: ["AB", "CD", "EF"]
        try:
            got = D.page_entries(F, Path("S 6.01 002.jpg"), None, [{}] * 3)
        finally:
            D.page_texts = saved
        self.assertEqual([p["bi"] for p in got], [1, 3])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class TwoPassSizing(unittest.TestCase):
    """The book size is estimated on a sample, then every balloon searches
    down from max(book, original): the size apply_book needs is exactly
    min(true max, max(book, original)), so nothing above it is probed."""

    def test_sizes_are_the_clamped_maxima(self):
        D, _ = load()
        f = FakeF()
        ps = [{"stem": f"S 6.01 {i:03d}", "bi": 1, "oldh": 18.0,
               "true": t, "b": {"kind": "bubble"}}
              for i, t in enumerate([22, 24, 30, 26, 40, 21, 23, 25, 27, 50])]
        rows, book = D.size_entries(f, FakeFit(), ps, sample=2)
        # estimated at BOOK_ESTIMATE_PCTL, not the fit's 35: placeholder
        # text fits a little more easily than the real words
        self.assertLess(D.BOOK_ESTIMATE_PCTL, f.BOOK_PCTL)
        self.assertEqual(book, int(np.percentile(
            [min(p["true"], 36) for p in ps[::2]], D.BOOK_ESTIMATE_PCTL)))
        for p, r in zip(ps, rows):
            self.assertEqual(r["max_size"], min(p["true"], max(book, 18)))
        self.assertLessEqual(max(f.caps), 36)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class ApplyBook(unittest.TestCase):
    def test_dialogue_clamp_and_issue_grouping(self):
        D, F = load()
        fit = F.Fitter()
        c15, c20, c25 = (fit.cap_height(s) for s in (15, 20, 25))
        rows = [
            # small original, room to spare: raised to the book size
            {"stem": "S 6.01 002", "bi": 1, "oldh": c15, "max_size": 20},
            {"stem": "S 6.01 003", "bi": 1, "oldh": c15, "max_size": 20},
            # big original: held at its own size (max(book, original))
            {"stem": "S 6.02 002", "bi": 1, "oldh": c25, "max_size": 30},
            {"stem": "S 6.02 002", "bi": 2, "oldh": c25, "max_size": None},
        ]
        got = D.apply_book(rows, fit)
        self.assertEqual(sorted(got), ["6.01", "6.02"])
        for r in got["6.01"]:
            self.assertAlmostEqual(r, c20 / c15, places=6)
        self.assertEqual(len(got["6.02"]), 1)
        self.assertAlmostEqual(got["6.02"][0], 1.0, places=6)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class IssueCaps(unittest.TestCase):
    def test_median_cap_in_printed_mm(self):
        D, _ = load()
        ps = [{"stem": "S 6.01 002", "oldh": 15.0},
              {"stem": "S 6.01 003", "oldh": 15.0},
              {"stem": "S 6.01 003", "oldh": 30.0},
              {"stem": "S 6.02 002", "oldh": 30.0}]
        got = D.issue_caps_mm(ps, 300.0)
        self.assertEqual(sorted(got), ["6.01", "6.02"])
        self.assertAlmostEqual(got["6.01"], 1.27, places=3)   # 15px @300
        self.assertAlmostEqual(got["6.02"], 2.54, places=3)


def shear(crop, k):
    """Lean a glyph right by k (x shifts k px per px of height, top most)."""
    h, w = crop.shape
    pad = int(abs(k) * h) + 2
    m = np.float32([[1, -k, k * h], [0, 1, 0]])
    out = cv2.warpAffine(crop.astype(np.uint8) * 255, m, (w + pad, h))
    return out > 127


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FontGate(unittest.TestCase):
    def glyphs(self, size=40):
        D, F = load()
        return D, F.Fitter(), D.render_glyphs(F.Fitter(), size)

    def delta(self, D, fit, crops, gl):
        ours = D.font_reference(fit, np.median([g.shape[0] for g in gl]))
        return D.font_delta(D.shape_stats(crops), ours)

    def test_our_own_glyphs_are_similar(self):
        D, fit, gl = self.glyphs()
        self.assertEqual(D.issue_font([self.delta(D, fit, gl[:8], gl)] * 6),
                         "similar")

    def test_a_far_off_statistic_is_different(self):
        D, fit, gl = self.glyphs()
        up = D.shape_stats(gl[:8])
        lean = D.shape_stats([shear(g, 0.25) for g in gl[:8]])
        self.assertGreater(lean["slant"] - up["slant"], 0.15)
        d = self.delta(D, fit, [shear(g, 0.25) for g in gl[:8]], gl)
        self.assertEqual(D.issue_font([d] * 6), "different")

    def test_a_wider_more_upright_hand_is_different(self):
        """The combination the labelled issues share and the similar ones
        never do — each statistic alone sits inside the similar range."""
        D, _ = load()
        d = {"width": D.WIDER + 0.02, "stroke": 0.0,
             "slant": -(D.UPRIGHT + 0.02), "lower": 0.0}
        self.assertEqual(D.issue_font([d] * 6), "different")
        wide_only = dict(d, slant=0.03)
        upright_only = dict(d, width=-0.04)
        self.assertEqual(D.issue_font([wide_only] * 6), "similar")
        self.assertEqual(D.issue_font([upright_only] * 6), "similar")

    def test_a_thin_stroke_measures_thinner(self):
        D, fit, gl = self.glyphs()
        thin = [cv2.erode(g.astype(np.uint8), np.ones((3, 3), np.uint8),
                          iterations=1) > 0 for g in gl[:8]]
        self.assertLess(D.shape_stats(thin)["stroke"],
                        D.shape_stats(gl[:8])["stroke"])

    def test_too_few_letters_measure_nothing(self):
        D, _, gl = self.glyphs()
        self.assertIsNone(D.shape_stats(gl[:5]))

    def test_issue_font_is_the_median_so_a_minority_cannot_flip_it(self):
        D, _ = load()
        same = {"width": 0.0, "stroke": 0.0, "slant": 0.0, "lower": 0.0}
        odd = {"width": 0.5, "stroke": 0.2, "slant": 0.5, "lower": 0.5}
        self.assertEqual(D.issue_font([same] * 6 + [odd] * 4), "similar")
        self.assertEqual(D.issue_font([same] * (D.FONT_MIN_BALLOONS - 1)),
                         "unsure")
        self.assertEqual(D.issue_font([]), "unsure")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class Main(unittest.TestCase):
    """A comic fitted before this step existed is never decided: writing a
    decisions file into it would move pages the user approved."""

    def run_main_in(self, files):
        import tempfile
        D, F = load()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for name in files:
                (work / name).write_text("{}")
            saved = F.WORK
            F.WORK = work
            try:
                rc = D.main()
            finally:
                F.WORK = saved
            return rc, sorted(p.name for p in work.iterdir())

    def test_main_refuses_legacy_comic(self):
        rc, listing = self.run_main_in(["layout.json"])
        self.assertEqual(rc, 0)
        self.assertEqual(listing, ["layout.json"])

    def test_main_never_rewrites_decisions(self):
        rc, listing = self.run_main_in(["issue_decisions.json"])
        self.assertEqual(rc, 0)
        self.assertEqual(listing, ["issue_decisions.json"])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
@unittest.skipUnless(REAL_COMIC and
                     (REPO / "relettering" / REAL_COMIC / "pristine").is_dir(),
                     "needs a processed comic's pristine pages")
class RealPage(unittest.TestCase):
    def test_placeholder_fit_runs_on_a_real_page(self):
        D, F = load()
        saved = F.COMIC
        F.configure(REAL_COMIC)        # read-only: collect_page only reads
        try:
            pages = [q for q in sorted((F.WORK / "pristine").iterdir())
                     if D.load_page(F, q) is not None]
            rows = D.predict_bubbles(F, F.Fitter(), pages[len(pages) // 2:
                                                         len(pages) // 2 + 1])
        finally:
            F.configure(saved)
        self.assertTrue(rows)
        for r in rows:
            self.assertGreater(r["oldh"], 0)
            if r["max_size"] is not None:
                self.assertTrue(10 < r["max_size"] <= 80, r)


if __name__ == "__main__":
    unittest.main()
