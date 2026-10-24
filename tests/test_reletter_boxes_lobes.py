"""Caption boxes and lobes (relettering/reletter_fit.py): the frame finder's
fill regrow, flush-left alignment, side-by-side lobes and the page edge.
Synthetic images only — no comic data.

These need cv2, which lives ONLY in .venv-reletter:
    .venv-reletter/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def load_fit():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_fit
    reletter_fit.configure("unit-test-comic")
    return reletter_fit


def framed_box(dark_top):
    """A 400x140 page with a black-framed 300x80 box. With `dark_top` the
    fill over most of its width grades smoothly from just under FRAME_SOFT
    at the top edge to white, a level per row."""
    img = np.full((140, 400, 3), 200, np.uint8)
    cv2.rectangle(img, (48, 28), (351, 111), (20, 20, 20), 3)
    img[31:109, 51:349] = 255
    if dark_top:
        for y in range(31, 109):
            img[y, 51:300] = int(min(255, 122 + (y - 31)))
    for x in range(120, 280, 24):           # a line of "letters"
        cv2.rectangle(img, (x, 66), (x + 12, 82), (10, 10, 10), -1)
    return img


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FrameBoxGradedFill(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def test_plain_box_found(self):
        img = framed_box(False)
        got = self.F.frame_box(img, [120, 66, 172, 17], [110, 60, 190, 30])
        self.assertIsNotNone(got)

    def test_fill_dark_at_one_edge_still_a_box(self):
        img = framed_box(True)
        got = self.F.frame_box(img, [120, 66, 172, 17], [110, 60, 190, 30])
        self.assertIsNotNone(got)
        x0, y0, x1, y1 = got
        self.assertLessEqual(abs(y0 - 31), 4)       # the dark band is fill
        self.assertLessEqual(abs(x1 - 349), 4)

    def test_regrow_stays_inside_the_flood_box(self):
        reg = np.full((40, 60, 3), 125, np.uint8)   # all in the band
        comp = np.zeros((40, 60), bool)
        comp[10:30, 10:30] = True
        dark = ~comp
        got = self.F.regrow_fill(reg, comp, dark)
        self.assertFalse(got[:, 30:].any())
        self.assertFalse(got[30:, :].any())


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FlushLeft(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def page(self, starts, ends):
        img = np.full((200, 400, 3), 255, np.uint8)
        for i, (a, c) in enumerate(zip(starts, ends)):
            y = 20 + 30 * i
            for x in range(a, c - 10, 16):
                cv2.rectangle(img, (x, y), (x + 10, y + 14), (0, 0, 0), -1)
        return img

    def test_flush_left_found(self):
        img = self.page([40, 40, 41, 40], [300, 220, 340, 180])
        b = {"kind": "bubble", "block": [30, 10, 340, 130]}
        left = self.F.orig_flush_left(img, b, 15.0)
        self.assertIsNotNone(left)
        self.assertAlmostEqual(left, 40, delta=2)

    def test_centred_is_not_flush(self):
        img = self.page([40, 80, 20, 100], [300, 260, 320, 240])
        b = {"kind": "bubble", "block": [10, 10, 340, 130]}
        self.assertIsNone(self.F.orig_flush_left(img, b, 15.0))

    def test_justified_is_not_flush(self):
        img = self.page([40, 40, 40, 40], [300, 300, 302, 300])
        b = {"kind": "bubble", "block": [30, 10, 300, 130]}
        self.assertIsNone(self.F.orig_flush_left(img, b, 15.0))

    def test_lines_set_flush_inside_their_rows(self):
        rows = {y: (50, 350) for y in range(0, 100)}
        e = {"line_height": 20, "lines": [
            {"y_top": 10, "cx": 200, "width": 100},
            {"y_top": 30, "cx": 200, "width": 280}]}
        self.F.flush_left_lines(e, rows, 60)
        self.assertAlmostEqual(e["lines"][0]["cx"] - 50, 60, delta=0.5)
        # too wide to start at the edge: held inside its row span
        self.assertLessEqual(e["lines"][1]["cx"] + 140, 348)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LobesAndEdges(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def test_side_by_side(self):
        a = {y: (0, 100) for y in range(0, 50)}
        c = {y: (95, 200) for y in range(30, 90)}
        self.assertTrue(self.F.lobes_side_by_side([a, c]))

    def test_stacked_is_not_side_by_side(self):
        a = {y: (0, 100) for y in range(0, 50)}
        c = {y: (10, 110) for y in range(50, 90)}
        self.assertFalse(self.F.lobes_side_by_side([a, c]))

    def test_page_edge_cleared(self):
        rows = {y: (0, 300) for y in range(0, 60)}
        got = self.F.clear_page_edge(rows, (500, 300), 8)
        self.assertEqual(min(got), 8)
        self.assertTrue(all(a >= 8 and c <= 292 for a, c in got.values()))

    def test_interior_rows_untouched(self):
        rows = {y: (40, 200) for y in range(50, 90)}
        self.assertEqual(self.F.clear_page_edge(rows, (500, 300), 8), rows)


if __name__ == "__main__":
    unittest.main()
