"""A balloon with a DARK coloured fill (a mid green whose weakest channel
is ~55) and black lettering. clean_bubble's old-lettering cut-off was
max(60, fill - 60) — a floor meant for light fills — so here it sat ABOVE
the fill: the whole interior counted as lettering and was wiped to white.
The cut-off must stay under the fill — and so must the WALL the balloon's
interior is flooded inside (drawn ink under DARK_MAX - 20): above a dark
fill the whole fill is wall, the interior comes back empty and the letters
are repainted with the fallback white.
Synthetic images only — no comic data.

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

GREEN = (98, 147, 56)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DarkFillBalloon(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_fit
        reletter_fit.configure("unit-test-comic")
        self.fit = reletter_fit

    def balloon(self, fill, halos=None):
        img = np.full((300, 500, 3), (200, 210, 240), np.uint8)   # sky
        cv2.ellipse(img, (250, 150), (200, 100), 0, 0, 360, fill, -1)
        cv2.ellipse(img, (250, 150), (200, 100), 0, 0, 360, (10, 10, 10), 4)
        if halos:   # rims the scan leaves round dark strokes: light, or a
            # greyish green between the ink and the fill
            for t, org in (("Para dar a todos", (110, 140)),
                           ("a veces, los pocos", (100, 180))):
                cv2.putText(img, t, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            halos, 7, cv2.LINE_AA)
        cv2.putText(img, "Para dar a todos", (110, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 15, 15), 2,
                    cv2.LINE_AA)
        cv2.putText(img, "a veces, los pocos", (100, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (15, 15, 15), 2,
                    cv2.LINE_AA)
        if fill != (250, 250, 250):
            # the bright halo specks an upscale leaves beside dark strokes
            for px, py in ((160, 120), (240, 165), (330, 128), (200, 190)):
                img[py:py + 3, px:px + 3] = (235, 240, 225)
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.ellipse(mask, (250, 150), (196, 96), 0, 0, 360, 1, -1)
        b = {"kind": "bubble", "bbox": [0, 0, 500, 300],
             "block": [100, 110, 290, 80]}
        return img, b, mask

    def clean(self, fill, halos=None):
        img, b, mask = self.balloon(fill, halos)
        self.fit.clean_bubble(img, b, mask)
        inner = np.zeros(mask.shape, np.uint8)
        cv2.ellipse(inner, (250, 150), (180, 85), 0, 0, 360, 1, -1)
        px = img[inner > 0].astype(int)
        return px, img

    def test_green_fill_stays_green(self):
        px, _ = self.clean(GREEN)
        med = np.median(px, axis=0)
        self.assertTrue(np.all(np.abs(med - GREEN) <= 12), med)

    def test_the_lettering_is_wiped(self):
        px, _ = self.clean(GREEN)
        self.assertLess((px.max(axis=1) <= 40).mean(), 0.005)

    def test_no_white_ghost_letters(self):
        # the wiped letters are repainted with the FILL, not with white
        px, _ = self.clean(GREEN)
        self.assertLess((px.min(axis=1) >= 180).mean(), 0.002)

    def ghost_share(self, halos):
        _, img = self.clean(GREEN, halos=halos)
        blk = img[110:190, 100:390].reshape(-1, 3).astype(int)
        return (np.abs(blk - GREEN).max(axis=1) > 30).mean()

    def test_light_halos_round_the_old_letters_go_too(self):
        self.assertLess(self.ghost_share((200, 215, 190)), 0.003)

    def test_grey_rims_round_the_old_letters_go_too(self):
        self.assertLess(self.ghost_share((62, 84, 46)), 0.003)

    def test_white_balloon_unchanged_rule(self):
        px, _ = self.clean((250, 250, 250))
        self.assertLess((px.min(axis=1) <= 100).mean(), 0.005)


if __name__ == "__main__":
    unittest.main()
