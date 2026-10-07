"""Caption boxes detection only part-holds, graded balloon fills, bites
sealed through a tail, and graded caption boxes with too few letters for
the letter passes. Synthetic pages only — no comic data.

    .venv-reletter/bin/python -m unittest discover tests
"""
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


def load_detect():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_detect
    return reletter_detect


def letters(img, x0, y0, cols, rows, step=22, lh=20, gap=28, col=(10, 10, 10)):
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * step, y0 + r * gap
            cv2.rectangle(img, (x, y), (x + 14, y + lh), col, 2)
    return [x0, y0, (cols - 1) * step + 15, (rows - 1) * gap + lh + 1]


def framed_box(img, x0, y0, x1, y1):
    """A yellow-to-white gradient caption box with a black frame."""
    for yy in range(y0, y1):
        t = (yy - y0) / max(1, y1 - y0 - 1)
        img[yy, x0:x1] = (255, 235, int(60 + 190 * t))
    cv2.rectangle(img, (x0 - 3, y0 - 3), (x1 + 2, y1 + 2), (0, 0, 0), 3)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class PartlyHeldBox(unittest.TestCase):
    """Every letter found, but the strip holds a fraction of the frame."""

    def setUp(self):
        self.fit = load_fit()
        self.tmp = tempfile.TemporaryDirectory()
        self.fit.BUB = Path(self.tmp.name)
        self.fit.LAYOUT_OVERRIDES = {}
        self.fit.POLICY = {}

    def tearDown(self):
        self.tmp.cleanup()

    def _page(self, strip_h):
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 600, 220)
        blk = letters(img, 150, 120, 12, 2)
        x, y, w, h = 100, 100, 500, strip_h
        cv2.imwrite(str(self.fit.BUB / "Series 7 - Title 020-b01-mask.png"),
                    np.full((h, w), 255, np.uint8))
        return img, [{"kind": "tint", "bbox": [x, y, w, h], "block": blk}]

    def test_strip_holding_part_of_the_frame_becomes_the_box(self):
        img, bubbles = self._page(80)               # 80 of 120 rows
        self.fit.auto_caption_boxes("Series 7 - Title 020", img, bubbles,
                                    ["TEXT"])
        ov = self.fit.override_for("Series 7 - Title 020", 1)
        self.assertIsNotNone(ov)
        self.assertTrue(all(abs(a - b) <= 2 for a, b in
                            zip(ov["box"], (100, 100, 600, 220))))

    def test_box_held_whole_is_left_alone(self):
        img, bubbles = self._page(118)
        self.fit.auto_caption_boxes("Series 7 - Title 020", img, bubbles,
                                    ["TEXT"])
        self.assertIsNone(self.fit.override_for("Series 7 - Title 020", 1))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FramePairsPolicy(unittest.TestCase):
    """A recognised strip pair becomes one box only under the policy."""

    def setUp(self):
        self.fit = load_fit()
        self.tmp = tempfile.TemporaryDirectory()
        self.fit.BUB = Path(self.tmp.name)
        self.fit.LAYOUT_OVERRIDES = {}

    def tearDown(self):
        self.tmp.cleanup()
        self.fit.POLICY = {}

    def _run(self, policy):
        self.fit.POLICY = policy
        img = np.full((500, 900, 3), (90, 120, 200), np.uint8)
        framed_box(img, 100, 100, 600, 220)
        stem = "Series 7 - Title 030"
        b1 = letters(img, 150, 115, 12, 2)
        b2 = letters(img, 150, 172, 12, 1)
        bubbles = [{"kind": "tint", "bbox": [100, 100, 500, 62],
                    "block": b1},
                   {"kind": "bubble", "bbox": [100, 160, 500, 60],
                    "block": b2}]
        for i, b in enumerate(bubbles, 1):
            cv2.imwrite(str(self.fit.BUB / f"{stem}-b{i:02d}-mask.png"),
                        np.full(b["bbox"][3:1:-1], 255, np.uint8))
        texts = self.fit.auto_caption_boxes(stem, img, bubbles,
                                            ["UPPER HALF", "LOWER."])
        return texts

    def test_pair_stays_two_strips_by_default(self):
        self.assertEqual(self._run({}), ["UPPER HALF", "LOWER."])

    def test_pair_is_one_box_under_the_policy(self):
        self.assertEqual(self._run({"frame_pairs": True}),
                         ["UPPER HALF LOWER.", ""])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class GradedFill(unittest.TestCase):
    def setUp(self):
        self.fit = load_fit()

    def test_flat_fill_keeps_the_row_median(self):
        region = np.full((60, 200, 3), 250, np.uint8)
        good = np.ones((60, 200), bool)
        self.assertIsNone(self.fit.fill_trend(region, good))

    def test_horizontal_stripes_keep_the_row_median(self):
        region = np.full((60, 200, 3), 250, np.uint8)
        region[::4] = (90, 200, 230)               # hologram stripes
        good = np.ones((60, 200), bool)
        self.assertIsNone(self.fit.fill_trend(region, good))

    def test_glow_to_blue_rim_is_followed_along_the_row(self):
        xs = np.abs(np.arange(200) - 100) / 100.0
        row = np.stack([255 - 57 * xs, 255 - 20 * xs, 255 - 1 * xs], axis=1)
        region = np.repeat(row[None], 60, axis=0).astype(np.uint8)
        good = np.ones((60, 200), bool)
        good[20:40, :] = False                      # a line of type
        est = self.fit.fill_trend(region, good)
        self.assertIsNotNone(est)
        err = np.abs(est[20:40].astype(int) - region[20:40].astype(int))
        self.assertLess(err.max(), 8)

    def test_wall_widens_only_for_the_ground_the_letters_stand_on(self):
        h, w = 120, 300
        dark = np.zeros((h, w), np.uint8)
        dark[40:60, 40:52] = 1                      # a letter
        first = np.ones((h, w), bool)
        inblock = np.zeros((h, w), bool)
        inblock[30:70, 30:200] = True
        shift = np.zeros((h, w), np.int16)
        self.assertEqual(self.fit.fill_wall(dark, first, inblock, shift),
                         self.fit.COLOUR_WALL)
        shift[30:70, 30:65] = 70                    # letter on blue ground
        self.assertGreaterEqual(
            self.fit.fill_wall(dark, first, inblock, shift), 80)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class BiteThroughATail(unittest.TestCase):
    """A pocket the welded line end seals off is given back even when the
    rim channel runs down a tail and out of the repair window."""

    def setUp(self):
        self.fit = load_fit()

    def _case(self):
        page = np.full((400, 400, 3), 120, np.uint8)
        page[20:181, 20:281] = 255
        cv2.rectangle(page, (20, 20), (280, 180), (0, 0, 0), 3)
        page[178:400, 146:155] = 255                # the tail, off the page
        page[178:400, 143:146] = 0
        page[178:400, 155:158] = 0
        page[45:76, 192:200] = 0                    # the welded line end
        mask = np.zeros((400, 400), np.uint8)
        mask[25:176, 25:276] = 1                    # a rim channel inside
        mask[176:200, 149:152] = 1                  # ...and down the tail
        mask[40:80, 192:276] = 0                    # the bite
        x, y, w, h = 25, 25, 251, 175
        return page, mask[y:y + h, x:x + w], (x, y), (20, 10, 240, 140)

    def test_pocket_sealed_by_the_mask_is_returned(self):
        page, m, org, blk = self._case()
        out = self.fit.repair_letter_bites(m, page, org, blk)
        self.assertGreater(int(out.sum()), int(m.sum()) + 1500)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class GradedBoxDetection(unittest.TestCase):
    """A short caption on a graded box: too few letters for either letter
    pass, found inside its frame by the graded-box pass."""

    def setUp(self):
        self.det = load_detect()

    def test_short_caption_in_a_graded_frame_is_found(self):
        img = np.full((600, 900, 3), (90, 120, 200), np.uint8)
        for yy in range(100, 170):
            t = (yy - 100) / 69.0
            img[yy, 100:420] = (int(240 + 15 * t), int(200 + 50 * t),
                                int(40 + 210 * t))
        cv2.rectangle(img, (97, 97), (422, 172), (0, 0, 0), 3)
        letters(img, 160, 125, 6, 1, col=(20, 20, 40))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 7 - Title 040.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _, bubbles = self.det.detect_page(p)
        self.assertEqual([b["kind"] for b in bubbles], ["tint"])
        x, y, w, h = bubbles[0]["bbox"]
        self.assertTrue(x <= 160 and y <= 125
                        and x + w >= 270 and y + h >= 145)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class SiblingLetters(unittest.TestCase):
    """A joined balloon's mask can run over its sibling's text; that text
    is cut out of the writing area."""

    def setUp(self):
        self.fit = load_fit()

    def test_sibling_text_inside_the_mask_is_cut_from_the_rows(self):
        img = np.full((400, 600, 3), 255, np.uint8)
        upper = letters(img, 100, 40, 10, 3)        # the sibling's lines
        lower = letters(img, 150, 250, 8, 3)        # this balloon's
        mask = np.ones((300, 500), np.uint8)        # covers both
        me = {"kind": "bubble", "bbox": [50, 50, 500, 300], "block": lower}
        sib = {"kind": "bubble", "bbox": [50, 20, 500, 140], "block": upper}
        box = self.fit.sibling_letter_box(img, me, 1, mask, [me, sib])
        self.assertIsNotNone(box)
        rows = {y: (50, 550) for y in range(50, 350)}
        cut = self.fit.cut_box(rows, box)
        # no row the sibling's text occupies keeps any of its columns...
        self.assertTrue(all(c <= box[0] or a >= box[2]
                            for yy, (a, c) in cut.items()
                            if box[1] <= yy < box[3]))
        # ...and this balloon's own rows are untouched
        self.assertEqual(cut[300], (50, 550))

    def test_no_sibling_text_no_cut(self):
        img = np.full((400, 600, 3), 255, np.uint8)
        lower = letters(img, 150, 250, 8, 3)
        me = {"kind": "bubble", "bbox": [50, 50, 500, 300], "block": lower}
        self.assertIsNone(self.fit.sibling_letter_box(
            img, me, 1, np.ones((300, 500), np.uint8), [me]))
