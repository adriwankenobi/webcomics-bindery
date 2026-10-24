"""A short balloon of THREE stacked short lines ("NO, / AMIGO / MÍO.")
JOINED to a bigger balloon, whose letters fused in the upscale: too few
blobs for paragraph_blocks (MIN_LETTERS), and the short-utterance pass's
fixed height cap (one or two lines) refused the three-line cluster. It fell
between the two passes and was never detected. The taller cap holds only
for a lobe of an already-detected balloon: free-standing, it admits art.
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


def word(img, x0, y0, n, step=22, lh=17):
    for c in range(n):
        x = x0 + c * step
        cv2.rectangle(img, (x, y0), (x + 14, y0 + lh), (10, 10, 10), 2)


def stack(img, cx, cy, lines, pitch=22):
    y = cy - (len(lines) * pitch) // 2
    for i, n in enumerate(lines):
        word(img, cx - (n * 22) // 2, y + i * pitch, n)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class ThreeLineShortBalloon(unittest.TestCase):
    def setUp(self):
        self.det = load_detect()

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012 002.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def joined_page(self, short, small_y=230):
        """A big balloon with a paragraph, and a small one joined to it
        (outlines drawn, then both interiors painted: one light region)."""
        img = np.full((600, 900, 3), 120, np.uint8)
        small, big = ((300, small_y), (95, 70)), ((520, 330), (190, 110))
        for c, ax in (small, big):
            cv2.ellipse(img, c, ax, 0, 0, 360, (15, 15, 15), 4)
        for c, ax in (small, big):
            cv2.ellipse(img, c, (ax[0] - 3, ax[1] - 3), 0, 0, 360,
                        (252, 252, 252), -1)
        stack(img, 530, 345, [6, 7, 6])
        stack(img, 300, small_y - 5, short)
        return img

    def test_joined_three_short_lines_found(self):
        img = self.joined_page([2, 2, 2])
        found = self.detect(img)
        self.assertEqual(len(found), 2)
        tops = sorted(found, key=lambda e: e["block"][1])
        bx, by, bw, bh = tops[0]["block"]
        # the small lobe's block holds all three lines
        self.assertLessEqual(by, 225 - 33)
        self.assertGreaterEqual(by + bh, 225 - 33 + 2 * 22 + 17)

    def test_tall_cluster_level_with_host_text_refused(self):
        # beside the host's text, in the same rows: a piece of those lines
        # (a caption's column of small-caps initials), not a lobe of its own
        img = self.joined_page([2, 2, 2], small_y=345)
        self.assertEqual(len(self.detect(img)), 1)

    def test_free_standing_tall_cluster_refused(self):
        img = np.full((500, 800, 3), 252, np.uint8)
        cv2.ellipse(img, (400, 250), (110, 75), 0, 0, 360, (15, 15, 15), 4)
        stack(img, 400, 250, [2, 2, 2])
        self.assertEqual(self.detect(img), [])

    def test_one_short_line_still_found(self):
        img = np.full((500, 800, 3), 252, np.uint8)
        cv2.ellipse(img, (400, 250), (110, 75), 0, 0, 360, (15, 15, 15), 4)
        stack(img, 400, 250, [3])
        self.assertEqual(len(self.detect(img)), 1)


if __name__ == "__main__":
    unittest.main()
