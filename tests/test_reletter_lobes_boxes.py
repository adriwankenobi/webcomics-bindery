"""Balloons found from the original lettering (letter_lobes and its parts)
and caption boxes found from their drawn frame (frame_box,
auto_caption_boxes). Synthetic pages only — no comic data.

    .venv-reletter/bin/python -m unittest discover tests
"""
import json
import sys
import tempfile
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


def letters(img, x0, y0, cols, rows, step=22, lh=20, gap=28):
    """A paragraph of letter-sized black blobs; returns its block box."""
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * step, y0 + r * gap
            cv2.rectangle(img, (x, y), (x + 14, y + lh), (10, 10, 10), -1)
    return [x0, y0, (cols - 1) * step + 15, (rows - 1) * gap + lh + 1]


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class ReadingOrder(unittest.TestCase):
    """Two balloons side by side are read left to right even when the right
    one's top is a few px higher — sorting on the top row first swapped the
    texts of four entries in book 3."""

    def setUp(self):
        self.fit = load_fit()

    def test_side_by_side_right_higher_reads_left_first(self):
        left, right = (0, 100, 300, 260), (320, 92, 600, 240)
        self.assertEqual(self.fit.reading_order([right, left]), [1, 0])

    def test_stacked_reads_top_first(self):
        top, low = (100, 0, 400, 100), (0, 140, 300, 260)
        self.assertEqual(self.fit.reading_order([low, top]), [1, 0])

    def test_mask_lobes_side_by_side_left_first(self):
        m = np.zeros((220, 640), np.uint8)
        cv2.ellipse(m, (150, 115), (130, 80), 0, 0, 360, 1, -1)
        cv2.ellipse(m, (470, 105), (130, 80), 0, 0, 360, 1, -1)  # higher
        lobes = self.fit.mask_lobes(m, 0, 0, 2)
        self.assertIsNotNone(lobes)
        self.assertLess(min(v[0] for v in lobes[0].values()),
                        min(v[0] for v in lobes[1].values()),
                        "the right balloon came first")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LetterGroups(unittest.TestCase):
    def setUp(self):
        self.fit = load_fit()

    def test_two_paragraphs_split_and_a_stray_speck_is_absorbed(self):
        a = [(10 + 22 * c, 10 + 28 * r, 14, 20) for r in range(3)
             for c in range(8)]
        b = [(300 + 22 * c, 20 + 28 * r, 14, 20) for r in range(3)
             for c in range(6)]
        speck = [(900, 600, 9, 9)]          # far from everything
        g = self.fit.letter_groups(a + b + speck, 2)
        self.assertIsNotNone(g, "the speck was split off alone")
        ga = [x for x in g if a[0] in x][0]
        gb = [x for x in g if b[0] in x][0]
        self.assertIsNot(ga, gb)
        self.assertTrue(set(a) <= set(ga) and set(b) <= set(gb))
        self.assertIn(speck[0], ga + gb)

    def test_too_few_letters_is_refused(self):
        self.assertIsNone(self.fit.letter_groups(
            [(0, 0, 10, 12), (40, 0, 10, 12)], 2))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class GeodesicOwner(unittest.TestCase):
    """Two balloons joined by a neck meet AT the neck: straight-line
    distance gave the big balloon's corner to the small one's text."""

    def setUp(self):
        self.fit = load_fit()

    def test_corner_of_big_balloon_goes_to_its_own_text(self):
        reg = np.zeros((320, 640), np.uint8)
        cv2.circle(reg, (100, 150), 70, 1, -1)              # small balloon
        cv2.circle(reg, (450, 150), 150, 1, -1)             # big balloon
        reg[140:161, 160:310] = 1                           # the neck
        reg = reg > 0
        seeds = np.zeros(reg.shape, np.int32)
        seeds[140:160, 80:120] = 1           # the small balloon's text
        seeds[200:220, 540:580] = 2          # the big balloon's text
        owner = self.fit.geodesic_owner(reg, seeds, 2)
        # the big balloon's upper-left: nearer the small balloon's text in
        # a straight line, but only reachable from it through the neck
        p = (70, 340)
        self.assertTrue(reg[p])
        d1 = np.hypot(p[1] - 100, p[0] - 150)
        d2 = np.hypot(p[1] - 560, p[0] - 210)
        self.assertLess(d1, d2)
        self.assertEqual(owner[p], 2)
        self.assertEqual(owner[150, 60], 1)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LetterLobes(unittest.TestCase):
    """Two balloons side by side on a coloured panel, returned as one entry
    whose mask leaked across the whole panel (2-030-2): each balloon comes
    back on its own, in reading order, and the panel is not part of it."""

    def setUp(self):
        self.fit = load_fit()

    def _page(self):
        img = np.full((360, 760, 3), (240, 170, 110), np.uint8)   # orange
        for (cx, cy, ax, ay) in ((190, 190, 160, 100), (560, 175, 170, 110)):
            cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360,
                        (255, 255, 255), -1)
            cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (0, 0, 0), 3)
        a = letters(img, 90, 150, 9, 3)
        b = letters(img, 460, 120, 9, 4)
        block = [a[0], b[1], b[0] + b[2] - a[0], a[1] + a[3] - b[1]]
        return img, block

    def test_leaked_mask_two_balloons(self):
        img, block = self._page()
        b = {"kind": "bubble", "bbox": [20, 20, 720, 320], "block": block}
        mask = np.ones((320, 720), np.uint8)          # the leak
        got = self.fit.letter_lobes(img, b, mask, 2)
        self.assertIsNotNone(got, "the two balloons were not found")
        _, lobes, jobs = got
        self.assertEqual(len(lobes), 2)
        lx = [min(v[0] for v in L.values()) for L in lobes]
        self.assertLess(lx[0], lx[1], "not in reading order")
        # each lobe stays inside its balloon: nothing reaches the panel's
        # left edge or the gap between the balloons' widest points
        self.assertGreater(lx[0], 25)
        self.assertLess(max(v[1] for v in lobes[0].values()), 400)
        self.assertEqual(len(jobs), 2)

    def test_one_balloon_with_a_good_mask_is_left_alone(self):
        img = np.full((300, 500, 3), (240, 170, 110), np.uint8)
        cv2.ellipse(img, (250, 150), (200, 110), 0, 0, 360,
                    (255, 255, 255), -1)
        cv2.ellipse(img, (250, 150), (200, 110), 0, 0, 360, (0, 0, 0), 3)
        block = letters(img, 150, 110, 9, 3)
        m = np.zeros((300, 500), np.uint8)
        cv2.ellipse(m, (250, 150), (195, 105), 0, 0, 360, 1, -1)
        b = {"kind": "bubble", "bbox": [0, 0, 500, 300], "block": block}
        self.assertIsNone(self.fit.letter_lobes(img, b, m, 1),
                          "a balloon detection got right was replaced")


def framed_box(img, x0, y0, x1, y1):
    """A yellow-to-white gradient caption box with a black frame."""
    for yy in range(y0, y1):
        t = (yy - y0) / max(1, y1 - y0 - 1)
        img[yy, x0:x1] = (255, 235, int(60 + 190 * t))
    cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 2, y1 + 2), (0, 0, 0), 3)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FrameBox(unittest.TestCase):
    def setUp(self):
        self.fit = load_fit()

    def test_gradient_box_found_from_its_frame(self):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 700, 300)
        blk = letters(img, 150, 230, 12, 2)            # only the low lines
        r = self.fit.frame_box(img, blk, blk)
        self.assertIsNotNone(r)
        self.assertTrue(all(abs(a - b) <= 2 for a, b in
                            zip(r, (100, 100, 700, 300))), r)

    def test_oval_balloon_is_not_a_box(self):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        cv2.ellipse(img, (400, 200), (250, 120), 0, 0, 360,
                    (255, 255, 255), -1)
        cv2.ellipse(img, (400, 200), (250, 120), 0, 0, 360, (0, 0, 0), 3)
        blk = letters(img, 290, 170, 10, 2)
        self.assertIsNone(self.fit.frame_box(img, blk, blk))

    def test_art_inside_the_frame_is_not_a_caption(self):
        img = np.full((500, 900, 3), (230, 230, 230), np.uint8)
        framed_box(img, 100, 100, 700, 400)
        cv2.rectangle(img, (500, 150), (560, 350), (0, 0, 0), -1)  # art
        blk = letters(img, 150, 200, 8, 2)
        self.assertIsNone(self.fit.frame_box(img, blk, blk))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class AutoCaptionBoxes(unittest.TestCase):
    """A box detection found as two strips, its top line in neither (the
    p2-066 shape): one member carries the whole caption and the box."""

    def setUp(self):
        self.fit = load_fit()
        self.tmp = tempfile.TemporaryDirectory()
        self.fit.BUB = Path(self.tmp.name)
        self.fit.LAYOUT_OVERRIDES = {}

    def tearDown(self):
        self.tmp.cleanup()

    def test_part_found_box_becomes_one_entry(self):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 700, 330)
        letters(img, 150, 120, 20, 2)                  # never detected
        b1 = letters(img, 150, 180, 10, 3)
        b2 = letters(img, 450, 180, 10, 3)
        stem = "Series 7 - Title 012"
        bubbles = []
        for i, blk in enumerate((b1, b2), 1):
            x, y, w, h = blk[0] - 4, blk[1] - 4, blk[2] + 8, blk[3] + 8
            bubbles.append({"kind": "bubble", "bbox": [x, y, w, h],
                            "block": blk})
            cv2.imwrite(str(self.fit.BUB / f"{stem}-b{i:02d}-mask.png"),
                        np.full((h, w), 255, np.uint8))
        texts = self.fit.auto_caption_boxes(
            stem, img, bubbles, ["LEFT HALF", "RIGHT HALF"])
        self.assertEqual(texts, ["LEFT HALF RIGHT HALF", ""])
        ov = self.fit.override_for(stem, 1)
        self.assertIsNotNone(ov)
        self.assertTrue(all(abs(a - b) <= 2 for a, b in
                            zip(ov["box"], (100, 100, 700, 330))))
        self.assertIn(1, self.fit.AUTO_MEMBERS[stem])

    def test_whole_sentence_members_stay_separate_lines(self):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 700, 330)
        letters(img, 150, 120, 20, 2)
        b1 = letters(img, 150, 180, 10, 3)
        b2 = letters(img, 450, 180, 10, 3)
        stem = "Series 7 - Title 014"
        bubbles = []
        for i, blk in enumerate((b1, b2), 1):
            x, y, w, h = blk[0] - 4, blk[1] - 4, blk[2] + 8, blk[3] + 8
            bubbles.append({"kind": "bubble", "bbox": [x, y, w, h],
                            "block": blk})
            cv2.imwrite(str(self.fit.BUB / f"{stem}-b{i:02d}-mask.png"),
                        np.full((h, w), 255, np.uint8))
        texts = self.fit.auto_caption_boxes(
            stem, img, bubbles, ["LA CAPITAL DE KIROS.", "19 MÁS TARDE."])
        self.assertEqual(texts[0], "LA CAPITAL DE KIROS.\n19 MÁS TARDE.")

    def test_hand_override_wins(self):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 700, 330)
        letters(img, 150, 120, 20, 2)
        b1 = letters(img, 150, 180, 10, 3)
        stem = "Series 7 - Title 013"
        x, y, w, h = b1[0] - 4, b1[1] - 4, b1[2] + 8, b1[3] + 8
        cv2.imwrite(str(self.fit.BUB / f"{stem}-b01-mask.png"),
                    np.full((h, w), 255, np.uint8))
        self.fit.LAYOUT_OVERRIDES = {"7-013 b01": {"box": [1, 2, 3, 4]}}
        texts = self.fit.auto_caption_boxes(
            stem, img, [{"kind": "tint", "bbox": [x, y, w, h],
                         "block": b1}], ["TEXT"])
        self.assertEqual(texts, ["TEXT"])
        self.assertEqual(self.fit.override_for(stem, 1)["box"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
