"""A short two-line balloon in BIG, thin-stroked lettering with wide word
gaps: paragraph_blocks splits it into clusters under MIN_LETTERS (the gaps
exceed its join), and the short-utterance pass, which does join it, took
two lines of big letters for "tall" because thin strokes fill less of their
boxes than its solid-lettering bar — the balloon was never detected.
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


# blob boxes (x, y, w, h) of two short lines whose word gaps (~35 px)
# exceed paragraph_blocks' join and whose words do not overlap the other
# line's: it sees two clusters of 4 and 5 letters
BOXES = [(461, 444, 30, 22), (530, 440, 26, 26), (562, 440, 28, 28),
         (596, 443, 21, 23),
         (410, 478, 26, 26), (441, 477, 27, 27), (476, 481, 24, 23),
         (546, 478, 26, 26), (579, 481, 29, 23)]


def letters(img):
    """Hollow letters: thin strokes, ~1/3 of each box inked."""
    for x, y, w, h in BOXES:
        cv2.rectangle(img, (x, y), (x + w, y + h), (25, 25, 25), 2)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class WideGappedShortBalloon(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def page(self):
        img = np.full((1000, 1100, 3), 5, np.uint8)         # printed on black
        cv2.ellipse(img, (520, 470), (150, 95), 0, 0, 360, (250, 250, 250), -1)
        letters(img)
        return img

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 1 - Title 001.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def test_two_lines_of_big_thin_letters_are_one_balloon(self):
        img = self.page()
        # the premise: no paragraph block of its own
        self.assertEqual(self.det.paragraph_blocks(
            self.det.letter_mask(img)), [])
        found = self.detect(img)
        self.assertEqual(len(found), 1)
        bx, by, bw, bh = found[0]["block"]
        self.assertLessEqual(by, 440)
        self.assertGreaterEqual(by + bh, 504)


if __name__ == "__main__":
    unittest.main()
