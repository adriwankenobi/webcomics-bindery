"""Balloons drawn TIGHT round big lettering, balloons JOINED with no outline
between them, compound entries whose lobes the waist cut cannot separate,
and tails. Synthetic masks and pages only — no comic data.

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


def load(name):
    sys.path.insert(0, str(REPO / "relettering"))
    return __import__(name)


def word(img, x0, y0, n, step=22, lh=17, w=14):
    for c in range(n):
        x = x0 + c * step
        cv2.rectangle(img, (x, y0), (x + w, y0 + lh), (10, 10, 10), 2)


def detect(det, img):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "Series 13 - Title 012 002.jpg"
        cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return det.detect_page(p)[1]


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class TightShortBalloon(unittest.TestCase):
    """Two short lines of BIG letters in a balloon drawn tight round them:
    the cluster box of the short-utterance pass (a wide join) is taller than
    the fixed 60 px 'two lines' cap and the balloon is under twice its area,
    so it was never detected."""

    def setUp(self):
        self.det = load("reletter_detect")

    def test_found(self):
        img = np.full((500, 800, 3), 252, np.uint8)
        cv2.ellipse(img, (400, 250), (78, 50), 0, 0, 360, (15, 15, 15), 4)
        for i in range(2):
            word(img, 355, 222 + i * 31, 3, step=31, lh=26, w=20)
        found = self.detect(img)
        self.assertEqual(len(found), 1)
        bx, by, bw, bh = found[0]["block"]
        self.assertLessEqual(by, 222)
        self.assertGreaterEqual(by + bh, 222 + 31 + 26)

    def detect(self, img):
        return detect(self.det, img)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CarveRivals(unittest.TestCase):
    """A flood that holds two joined balloons keeps only its own side of
    the seam between the two letterings."""

    def setUp(self):
        self.det = load("reletter_detect")

    def test_keeps_own_side(self):
        comp = np.zeros((200, 400), np.uint8)
        cv2.circle(comp, (110, 100), 90, 1, -1)
        cv2.circle(comp, (290, 100), 90, 1, -1)
        lb = np.zeros_like(comp)
        lb[80:120, 70:150] = 1                  # own letters
        lb[80:120, 250:330] = 1                 # the joined balloon's
        out = self.det.carve_rivals(comp, lb, (60, 70, 100, 60),
                                    [(240, 70, 100, 60)], (0, 0))
        self.assertTrue(out[100, 110])
        self.assertFalse(out[100, 290])
        self.assertFalse(out[:, 210:].any())

    def test_rival_outside_the_flood_changes_nothing(self):
        comp = np.zeros((200, 400), np.uint8)
        cv2.circle(comp, (110, 100), 90, 1, -1)
        lb = np.zeros_like(comp)
        lb[80:120, 70:150] = 1
        lb[80:120, 300:380] = 1                 # letters, but not in comp
        out = self.det.carve_rivals(comp, lb, (60, 70, 100, 60),
                                    [(290, 70, 100, 60)], (0, 0))
        self.assertTrue((out == comp).all())


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FitShapes(unittest.TestCase):
    def setUp(self):
        self.fit = load("reletter_fit")

    def oval_rows(self, cx=300, cy=200, ax=150, ay=80):
        rows = {}
        for y in range(cy - ay + 1, cy + ay):
            half = ax * (1 - ((y - cy) / ay) ** 2) ** 0.5
            rows[y] = (int(cx - half), int(cx + half))
        return rows

    def test_tail_mouth_is_clipped(self):
        rows = self.oval_rows()
        for y in range(240, 280):               # a tail off the right
            a, b = rows[y]
            rows[y] = (a, max(b, 420 + (y - 240) * 2))
        out = self.fit.clip_tails(rows)
        self.assertLessEqual(out[270][1], self.oval_rows()[240][1] + 4)
        self.assertEqual(out[200], rows[200])   # the body is untouched

    def test_waist_is_not_a_tail(self):
        top = self.oval_rows(cy=150, ay=70)
        low = self.oval_rows(cy=280, ay=70)
        rows = {**top}
        for y, sp in low.items():
            rows[y] = sp if y not in rows else (min(rows[y][0], sp[0]),
                                                max(rows[y][1], sp[1]))
        out = self.fit.clip_tails(rows)
        self.assertEqual(out[280], rows[280])

    def test_paragraph_must_stay_in_its_band(self):
        bands = [{y: (0, 300) for y in range(100, 200)},
                 {y: (0, 300) for y in range(200, 300)}]
        lh = 30
        good = [(110, 0, 300, []), (140, 0, 300, []),
                (215, 0, 300, []), (245, 0, 300, [])]
        spill = [(110, 0, 300, []), (140, 0, 300, []), (170, 0, 300, []),
                 (200, 0, 300, []), (247, 0, 300, [])]
        self.assertTrue(self.fit.paras_in_bands(good, lh, bands))
        self.assertFalse(self.fit.paras_in_bands(spill, lh, bands))

    def test_keep_puts_back_coloured_art_not_lettering(self):
        before = np.full((100, 200, 3), 250, np.uint8)
        cv2.circle(before, (50, 50), 20, (160, 40, 160), -1)     # purple SFX
        cv2.circle(before, (50, 50), 20, (20, 20, 20), 2)        # its outline
        cv2.rectangle(before, (120, 40), (140, 60), (15, 15, 15), -1)  # letter
        img = np.full_like(before, 250)                          # wiped
        self.fit.keep_coloured(img, before, [(0, 0, 200, 100)])
        self.assertTrue((img[50, 50] == before[50, 50]).all())
        self.assertTrue((img[50, 30] == before[50, 30]).all())  # outline
        self.assertTrue((img[50, 130] == 250).all())            # letter stays wiped

    def test_solidity_tells_a_joined_pair_from_one_oval(self):
        oval = np.zeros((300, 400), np.uint8)
        cv2.ellipse(oval, (200, 150), (180, 110), 0, 0, 360, 1, -1)
        pair = np.zeros((300, 400), np.uint8)
        cv2.circle(pair, (120, 110), 90, 1, -1)
        cv2.circle(pair, (280, 190), 90, 1, -1)
        self.assertGreater(self.fit.mask_solidity(oval),
                           self.fit.JOINED_SOLIDITY)
        self.assertLess(self.fit.mask_solidity(pair),
                        self.fit.JOINED_SOLIDITY)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DataKeys(unittest.TestCase):
    """A page whose file name is NFD on disk is found under the NFC key its
    transcripts were written with, and the other way round."""

    def test_either_unicode_form_finds_the_data(self):
        import unicodedata
        fit = load("reletter_fit")
        nfc = unicodedata.normalize("NFC", "Series 13 - Título 012")
        nfd = unicodedata.normalize("NFD", nfc)
        self.assertNotEqual(nfc, nfd)
        self.assertEqual(fit.data_key(nfd, {nfc: []}), nfc)
        self.assertEqual(fit.data_key(nfc, {nfd: []}), nfd)
        self.assertEqual(fit.data_key("Series 13 - Other 001", {nfc: []}),
                         "Series 13 - Other 001")
