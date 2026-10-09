"""A short balloon whose words are split by an ELLIPSIS ("PU... PUES,
YO..."): the dots are too small to be letters, so paragraph_blocks sees two
clusters under MIN_LETTERS, and the short-utterance pass's wider join sees
one at MIN_LETTERS or more. It fell between the two passes and was never
detected. Synthetic pages only — no comic data.

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


def word(img, x0, y0, n, step=22, lh=20):
    for c in range(n):
        x = x0 + c * step
        cv2.rectangle(img, (x, y0), (x + 14, y0 + lh), (10, 10, 10), 2)
    return x0 + (n - 1) * step + 15


def dots(img, x0, y, n=3, step=8):
    for c in range(n):
        cv2.rectangle(img, (x0 + c * step, y), (x0 + c * step + 3, y + 3),
                      (10, 10, 10), -1)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class EllipsisSplitShortBalloon(unittest.TestCase):
    def setUp(self):
        self.det = load_detect()

    def page(self, first, second):
        img = np.full((500, 800, 3), 252, np.uint8)
        cv2.ellipse(img, (400, 250), (190, 55), 0, 0, 360, (15, 15, 15), 4)
        y = 240
        end = word(img, 245, y, first)
        dots(img, end + 6, y + 17)
        word(img, end + 36, y, second)
        return img

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "The Clone Wars 1 - Test 001.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def test_pieces_under_min_letters_joined_over_it(self):
        img = self.page(2, 7)
        # the premise: neither piece is a paragraph block on its own
        self.assertEqual(self.det.paragraph_blocks(
            self.det.letter_mask(img)), [])
        found = self.detect(img)
        self.assertEqual(len(found), 1)
        bx, by, bw, bh = found[0]["block"]
        self.assertLessEqual(bx, 245)
        self.assertGreaterEqual(bx + bw, 245 + 2 * 22 + 36 + 6 * 22)


if __name__ == "__main__":
    unittest.main()
