"""A balloon drawn with a DASHED outline (a whisper) on near-white art.
Synthetic pages only — no comic data.

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


def load_detect():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_detect
    return reletter_detect


def letters(img, x0, y0, cols, rows, step=22, lh=20, gap=28, accents=False):
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * step, y0 + r * gap
            cv2.rectangle(img, (x, y), (x + 14, y + lh), (10, 10, 10), 2)
            if accents and c % 3 == 0:
                # an acute accent: a short slanted stroke over the letter
                cv2.line(img, (x + 5, y - 3), (x + 11, y - 8), (10, 10, 10), 3)
    return [x0, y0, (cols - 1) * step + 15, (rows - 1) * gap + lh + 1]


def dashed_balloon(ground=(244, 238, 224), centre=(450, 330),
                   axes=(220, 130), period=14, dash=10):
    """A white oval with a dashed black outline on pale ground (skin, sky):
    the ground is too close to the fill for any brightness or colour test,
    so the dashes are the only wall."""
    img = np.full((700, 900, 3), ground, np.uint8)
    cv2.ellipse(img, centre, axes, 0, 0, 360, (252, 252, 252), -1)
    for a in range(0, 360, period):
        cv2.ellipse(img, centre, axes, 0, a, a + dash, (15, 15, 15), 4)
    return img


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DashedOutline(unittest.TestCase):
    def setUp(self):
        self.det = load_detect()

    def _ring(self, img):
        return self.det.dashed_rings(img, self.det.letter_mask(img))

    def test_dashes_are_found_and_sealed(self):
        img = dashed_balloon()
        letters(img, 320, 285, 12, 3)
        ring, seal = self._ring(img)
        self.assertGreater(int(ring.sum()), 0, "no dashed ring found")
        # every dash sits on the oval, never on the text
        ys, xs = np.nonzero(ring)
        self.assertFalse(((ys > 270) & (ys < 370) & (xs > 310)
                          & (xs < 590)).any(), "a letter taken for a dash")
        self.assertGreater(int((seal > 0).sum()), int((ring > 0).sum()),
                           "the gaps between the dashes are not sealed")

    def test_accents_stay_letters(self):
        # an accent is a stroke too: it must not chain into the ring, or a
        # seal line is drawn through the lettering
        img = dashed_balloon()
        letters(img, 320, 285, 12, 3, accents=True)
        ring, seal = self._ring(img)
        self.assertGreater(int(ring.sum()), 0)
        self.assertFalse(seal[270:375, 310:590].any(),
                         "a seal line crosses the lettering")

    def test_balloon_mask_stays_inside_the_dashes(self):
        img = dashed_balloon()
        blk = letters(img, 320, 285, 12, 3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 7 - Title 041.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            page, bubbles = self.det.detect_page(p)
        self.assertEqual(len(bubbles), 1, [b["kind"] for b in bubbles])
        b = bubbles[0]
        x, y, w, h = b["bbox"]
        # inside the oval (230..670 x 200..460), not the whole page
        self.assertTrue(x >= 225 and y >= 195 and x + w <= 675
                        and y + h <= 465, b["bbox"])
        # the block is the TEXT, not the dashed outline round it
        self.assertLessEqual(b["block"][2], blk[2] + 30, b["block"])
        self.assertLessEqual(b["block"][3], blk[3] + 30, b["block"])
        # ...and it is marked, so the fit leaves its mask alone
        self.assertTrue(b.get("dashed"), "dashed entry not marked")
        # the caller gets the page untouched (the crops come from it)
        self.assertTrue((page == img).all(), "seal lines leaked into page")

    def test_solid_outline_is_left_alone(self):
        img = np.full((700, 900, 3), 255, np.uint8)
        cv2.ellipse(img, (450, 330), (220, 130), 0, 0, 360, (0, 0, 0), 3)
        letters(img, 320, 285, 12, 3, accents=True)
        ring, seal = self._ring(img)
        self.assertEqual(int(ring.sum()), 0)
        self.assertEqual(int(seal.sum()), 0)

    def test_parallel_speed_lines_round_the_text_are_not_a_ring(self):
        # speed lines / scan lines chain round lettering too, but they are
        # all parallel — an outline's dashes point every way
        img = np.full((700, 900, 3), (240, 235, 225), np.uint8)
        cv2.rectangle(img, (300, 250), (600, 400), (252, 252, 252), -1)
        letters(img, 340, 290, 11, 3)
        for yy in range(200, 460, 16):
            for xx in range(220, 680, 40):
                if 300 <= xx <= 600 and 250 <= yy <= 400:
                    continue
                cv2.line(img, (xx, yy), (xx + 26, yy), (10, 10, 10), 3)
        ring, seal = self._ring(img)
        self.assertEqual(int(ring.sum()), 0)

    def test_a_column_of_strokes_in_the_text_is_not_a_ring(self):
        # lettering full of I's and dashes is still lettering
        img = np.full((700, 900, 3), 255, np.uint8)
        for r in range(4):
            for c in range(14):
                x, y = 300 + c * 20, 260 + r * 30
                if c % 2:
                    cv2.line(img, (x + 7, y), (x + 7, y + 20), (10, 10, 10), 3)
                else:
                    cv2.rectangle(img, (x, y), (x + 14, y + 20),
                                  (10, 10, 10), 2)
        ring, seal = self._ring(img)
        self.assertEqual(int(ring.sum()), 0)


if __name__ == "__main__":
    unittest.main()
