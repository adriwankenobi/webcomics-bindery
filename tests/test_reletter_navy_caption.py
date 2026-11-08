"""A short caption (one word and an ellipsis: too few letters for a
paragraph) in a pale-blue box with a NAVY frame. Only the framed-caption
pass can find it, and it requires a frame "dark in every channel" — navy
(0, 7, 111) is dark by luminance but its blue channel is far over that bar,
so the box was refused and kept its original lettering. It was also hidden
by a neighbouring balloon whose bbox touched its letter block by a pixel
row: the pass skipped any overlap at all.
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

FILL = (125, 173, 213)
NAVY = (0, 7, 111)
INK = (10, 15, 60)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class NavyFramedCaption(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def page(self):
        img = np.full((900, 1000, 3), (90, 170, 230), np.uint8)   # sky
        cv2.rectangle(img, (200, 300), (480, 370), FILL, -1)
        cv2.rectangle(img, (200, 300), (480, 370), NAVY, 6)
        for k in range(6):                                 # 6 letters
            x = 250 + k * 28
            cv2.rectangle(img, (x, 322), (x + 16, 346), INK, 3)
        for x in (420, 432, 444):                          # the ellipsis
            cv2.rectangle(img, (x, 341), (x + 5, 346), INK, -1)
        return img

    def test_premise_navy_is_not_dark_in_every_channel(self):
        self.assertGreater(max(NAVY), 90)

    def test_a_navy_framed_short_caption_is_found(self):
        found = [e for e in self.detect(self.page())
                 if e["block"][1] < 380 and e["block"][1] + e["block"][3] > 300]
        self.assertEqual(len(found), 1)
        x, y, w, h = found[0]["bbox"]
        self.assertTrue(195 <= x <= 215 and 295 <= y <= 315, found[0]["bbox"])


if __name__ == "__main__":
    unittest.main()
