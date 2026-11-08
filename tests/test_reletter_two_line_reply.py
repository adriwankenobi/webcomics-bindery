"""A free-standing two-line reply ("NO / NEITHER..."-sized: 7 blobs, an
ellipsis fused into the last word) in big, ROUND, thin-stroked lettering.
Too few blobs for paragraph_blocks; the short-utterance pass joins it, but
its strokes fill ~0.31 of their boxes, under the 0.38 bar a cluster of
fewer than MIN_LETTERS blobs had to clear to count as lettering — so its two
lines read as "tall", and a tall cluster with no host balloon is refused.
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


# blob boxes (x, y, w, h) measured on a real page: a 2-letter word over a
# 5-blob line whose ellipsis fused with its last letters
BOXES = [(593, 434, 25, 22), (627, 436, 27, 20),
         (511, 466, 24, 24), (543, 466, 56, 27), (607, 466, 20, 28),
         (635, 466, 43, 24), (686, 470, 26, 20)]
STROKE = 2


def letters(img):
    """Round hollow letters: thin elliptical strokes."""
    for x, y, w, h in BOXES:
        cv2.ellipse(img, (x + w // 2, y + h // 2),
                    (w // 2 - 1, h // 2 - 1), 0, 0, 360, (25, 25, 25),
                    STROKE)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FreeStandingTwoLineReply(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def page(self):
        img = np.full((1000, 1100, 3), 40, np.uint8)        # dark art
        cv2.ellipse(img, (605, 462), (140, 75), 0, 0, 360,
                    (250, 250, 250), -1)
        cv2.ellipse(img, (605, 462), (140, 75), 0, 0, 360, (10, 10, 10), 4)
        letters(img)
        return img

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def test_premise_thin_round_strokes(self):
        lm = self.det.letter_mask(self.page())
        n, _, st, _ = cv2.connectedComponentsWithStats(lm, 8)
        self.assertEqual(n - 1, len(BOXES))
        inked = st[1:, 4].sum() / (st[1:, 2] * st[1:, 3]).sum()
        self.assertTrue(0.30 <= inked < 0.38, inked)
        self.assertEqual(self.det.paragraph_blocks(lm), [])

    def test_the_reply_is_a_balloon(self):
        found = self.detect(self.page())
        self.assertEqual(len(found), 1)
        bx, by, bw, bh = found[0]["block"]
        self.assertLessEqual(by, 436)
        self.assertGreaterEqual(by + bh, 490)


if __name__ == "__main__":
    unittest.main()
