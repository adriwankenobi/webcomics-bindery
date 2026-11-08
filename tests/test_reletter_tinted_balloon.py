"""A BALLOON on a tinted fill (pale blue, jagged outline) is found by the
tinted-caption-box pass, which grows a RECTANGLE round its letters and keeps
no mask: the fit then wipes it as a caption box, painting fill over the
outline's spikes and the art in the rectangle's corners. A tinted region
that is a closed shape leaving its rectangle's corners empty is a balloon,
and comes back as one — its interior as the mask. A real tinted box stays
a box, a drop cap over one corner included; a balloon cut by the page edge
is judged by its other corners.
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

FILL = (140, 200, 245)        # pale blue: below LIGHT_MIN in its weakest channel
INK = (20, 25, 60)            # navy lettering and outline


def letters(img, x0, y0, rows, cols, step=24):
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * step, y0 + r * 36
            cv2.rectangle(img, (x, y), (x + 14, y + 24), INK, 3)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class TintedBalloon(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def balloon_page(self):
        img = np.full((900, 1000, 3), (20, 60, 70), np.uint8)   # dark sea
        pts = []
        for k in range(28):                                  # jagged ellipse
            t = 2 * np.pi * k / 28
            r = 1.0 if k % 2 else 1.12
            pts.append((500 + 190 * r * np.cos(t), 450 + 110 * r * np.sin(t)))
        poly = np.array(pts, np.int32)
        cv2.fillPoly(img, [poly], FILL)
        cv2.polylines(img, [poly], True, INK, 5)
        letters(img, 390, 395, 3, 9)
        return img, poly

    def box_page(self):
        img = np.full((900, 1000, 3), (20, 60, 70), np.uint8)
        cv2.rectangle(img, (300, 360), (700, 540), FILL, -1)
        cv2.rectangle(img, (300, 360), (700, 540), INK, 4)
        letters(img, 390, 395, 3, 9)
        return img

    def test_a_tinted_balloon_is_a_balloon_with_its_shape(self):
        img, poly = self.balloon_page()
        found = [e for e in self.detect(img) if e["block"][1] < 520]
        self.assertEqual(len(found), 1)
        e = found[0]
        self.assertEqual(e["kind"], "bubble")
        self.assertIsNotNone(e.get("_mask"))
        x, y, w, h = e["bbox"]
        m = e["_mask"].astype(bool)
        inside = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(inside, [poly], 1)
        outside = cv2.dilate(inside, np.ones((9, 9), np.uint8)) == 0
        self.assertEqual(outside[y:y + h, x:x + w][m].sum(), 0,
                         "the mask reaches past the outline")

    def test_a_balloon_cut_by_the_page_edge_is_a_balloon(self):
        img, _ = self.balloon_page()
        img = np.ascontiguousarray(img[:, 330:])             # cut on the left
        found = [e for e in self.detect(img) if e["block"][1] < 520]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], "bubble")

    def test_a_tinted_box_with_a_drop_cap_stays_a_box(self):
        img = self.box_page()
        cv2.rectangle(img, (300, 360), (370, 450), (200, 30, 20), -1)  # cap
        cv2.rectangle(img, (300, 360), (370, 450), INK, 3)
        found = [e for e in self.detect(img) if e["block"][1] < 520]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], "tint")

    def test_a_tinted_box_stays_a_box(self):
        found = [e for e in self.detect(self.box_page())
                 if e["block"][1] < 520]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["kind"], "tint")


if __name__ == "__main__":
    unittest.main()
