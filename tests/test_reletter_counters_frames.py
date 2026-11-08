"""Cleaner leftovers on a poorly scanned book.

1. A line welded to the balloon's edge (a bite): its letters were wiped,
   but their COUNTERS were neither fill nor holes in it, so the pale
   antialiased rim inside each O stayed behind as a faint ring.
2. A caption box with a COLOURED frame (red on yellow): the frame finder
   only knew black frames, so the box wipe painted the frame's sides with
   the fill and the box printed bigger than the original.
3. A TILTED caption box: its upright bbox takes in wedges of the artwork,
   which read as ink and were repainted with the fill.
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


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CleanerLeftovers(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_fit
        reletter_fit.configure("unit-test-comic")
        self.fit = reletter_fit

    def test_counter_rims_of_a_welded_line_are_wiped(self):
        H, W = 260, 460
        img = np.full((H, W, 3), 12, np.uint8)               # dark panel
        cv2.ellipse(img, (230, 120), (210, 110), 0, 0, 360,
                    (250, 250, 250), -1)                     # no outline
        mask = np.zeros((H, W), np.uint8)
        cv2.ellipse(mask, (230, 120), (210, 110), 0, 0, 360, 255, -1)

        def letter_o(cx, cy):
            cv2.ellipse(img, (cx, cy), (11, 13), 0, 0, 360, (50, 50, 50), 5)
            # the upscaler's pale rim just inside the counter
            cv2.ellipse(img, (cx, cy), (7, 9), 0, 0, 360, (200, 200, 200), 1)
        for cx in range(110, 360, 30):
            letter_o(cx, 80)                                 # deep inside
        for cx in range(150, 320, 30):
            letter_o(cx, 200)                                # last line...
        # ...welded to the edge: the dark panel bites up into it
        cv2.rectangle(img, (140, 214), (330, 240), (12, 12, 12), -1)
        mask[214:241, 140:331] = 0
        rim = img.min(axis=2) == 200
        b = {"kind": "bubble", "bbox": [0, 0, W, H],
             "block": [95, 64, 280, 152]}
        self.fit.clean_bubble(img, b, mask)
        left = img[rim].min(axis=1)
        self.assertGreater(left.min(), 235,
                           "a counter's pale rim survived the wipe")

    def test_coloured_box_frame_survives_the_box_wipe(self):
        H, W = 200, 560
        img = np.full((H, W, 3), 20, np.uint8)
        x0, y0, x1, y1 = 60, 40, 500, 175
        cv2.rectangle(img, (x0 - 8, y0 - 8), (x1 + 7, y1 + 7),
                      (240, 60, 0), -1)                      # red frame
        cv2.rectangle(img, (x0 + 6, y0), (x1 - 7, y1 - 1),
                      (250, 240, 60), -1)                    # yellow fill
        for cy in (70, 115):
            for cx in range(100, 460, 24):
                cv2.rectangle(img, (cx, cy), (cx + 14, cy + 22),
                              (15, 15, 15), -1)              # letters
        frame = np.zeros((H, W), bool)
        frame[y0:y1, x0:x0 + 6] = True                       # sides INSIDE
        frame[y0:y1, x1 - 6:x1] = True                       # the box rect
        self.fit.clean_caption_box(img, (x0, y0, x1, y1))
        red = (img[..., 0] > 180) & (img[..., 1] < 90)
        self.assertGreater(red[frame].mean(), 0.9,
                           "the box frame was wiped with the fill")
        letters = np.zeros((H, W), bool)
        letters[70:138, 100:474] = True
        self.assertLess((img.max(axis=2) < 100)[letters].mean(), 0.02,
                        "the lettering survived")

    def test_tilted_box_spares_the_art_outside_its_edges(self):
        H, W = 220, 560
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (235, 120, 40)                              # orange art
        for x in range(0, W, 40):                            # with strokes
            cv2.line(img, (x, 0), (x + 60, H), (250, 190, 90), 6)
        quad = np.array([[60, 70], [500, 30], [505, 175], [64, 200]],
                        np.int32)                            # tilted box
        cv2.fillPoly(img, [quad], (190, 170, 230))           # lavender
        cv2.polylines(img, [quad], True, (70, 20, 70), 5)    # dark frame
        for cy in (90, 135):
            for cx in range(110, 450, 24):
                cv2.rectangle(img, (cx, cy), (cx + 14, cy + 22),
                              (40, 15, 30), -1)              # letters
        before = img.copy()
        inside = np.zeros((H, W), np.uint8)
        cv2.fillPoly(inside, [quad], 1)
        outside = cv2.dilate(inside, np.ones((9, 9), np.uint8)) == 0
        box = (56, 26, 510, 205)                             # its bbox
        self.fit.clean_caption_box(img, box)
        x0, y0, x1, y1 = box
        win = np.zeros((H, W), bool)
        win[y0:y1, x0:x1] = True
        moved = np.abs(img.astype(int) - before).max(axis=2) > 30
        self.assertLess(moved[win & outside].mean(), 0.01,
                        "the art outside the tilted box was repainted")
        letters = np.zeros((H, W), bool)
        letters[90:158, 110:448] = True
        self.assertLess((img.max(axis=2) < 100)[letters].mean(), 0.02,
                        "the lettering survived")


if __name__ == "__main__":
    unittest.main()
