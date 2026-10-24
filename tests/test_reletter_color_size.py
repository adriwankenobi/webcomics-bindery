"""`color` and `size: "orig"` layout overrides: display type set in our
font at its printed colour and size. Synthetic pages only — no comic data.

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


def load_fit():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_fit
    reletter_fit.configure("unit-test-comic")
    return reletter_fit


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class OrigInk(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def page(self, rgb):
        img = np.full((200, 300, 3), 250, np.uint8)
        # three letter-sized strokes, with a blended rim around each
        for x in (40, 110, 180):
            cv2.rectangle(img, (x - 2, 58), (x + 32, 132),
                          tuple(int((c + 250) / 2) for c in rgb), -1)
            cv2.rectangle(img, (x, 60), (x + 30, 130), rgb, -1)
        return img

    def test_coloured_letters_keep_their_colour(self):
        for rgb in ((231, 6, 7), (245, 130, 70)):
            img = self.page(rgb)
            got = self.F.orig_ink_rgb(img, {"kind": "bubble",
                                            "block": [30, 50, 200, 90]})
            self.assertEqual(got, list(rgb))

    def test_no_ink_is_none(self):
        img = np.full((100, 100, 3), 250, np.uint8)
        self.assertIsNone(self.F.orig_ink_rgb(
            img, {"kind": "bubble", "block": [10, 10, 50, 50]}))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class EntryColour(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def test_default_by_kind(self):
        self.assertEqual(self.F.entry_color({"kind": "dark"}, None, None),
                         "white")
        self.assertEqual(self.F.entry_color({"kind": "bubble"}, {}, [1, 2, 3]),
                         "black")

    def test_orig_and_explicit(self):
        b = {"kind": "bubble"}
        self.assertEqual(self.F.entry_color(b, {"color": "orig"}, (9, 8, 7)),
                         [9, 8, 7])
        self.assertEqual(self.F.entry_color(b, {"color": [1, 2, 3]}, None),
                         [1, 2, 3])
        # nothing measured: "orig" falls back to the kind's colour
        self.assertEqual(self.F.entry_color(b, {"color": "orig"}, None),
                         "black")

    def test_orig_size(self):
        self.assertEqual(self.F.orig_size(39.0, 0.7), 56)
        self.assertEqual(self.F.orig_size(0.0, 0.7), 0)


if __name__ == "__main__":
    unittest.main()
