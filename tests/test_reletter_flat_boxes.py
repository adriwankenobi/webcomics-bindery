"""Caption boxes on a FLAT COLOURED fill — white letters on purple or blue,
red letters on yellow. The white-on-black pass wants a black ring round each
letter, the light-paper and tint passes paper or letters dark in every
channel, so all three were blind to them: the boxes came back missing, or as
strips seeded on a few letters. `flat_caption_boxes` finds them (white ->
'dark', coloured -> 'tint'), detection replaces the strips with the box, and
the fit's `tint_ink` takes coloured letters for ink. Synthetic pages only.

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

PURPLE, BLUE, YELLOW = (77, 47, 149), (3, 135, 210), (251, 239, 20)
WHITE, RED, BLACK = (252, 252, 252), (215, 30, 25), (10, 10, 10)


def load(name):
    sys.path.insert(0, str(REPO / "relettering"))
    return __import__(name)


def caption(img, x0, y0, w, h, fill, ink, frame=WHITE, lines=(8, 7, 5)):
    """A framed caption box with rows of letter-sized strokes."""
    cv2.rectangle(img, (x0 - 4, y0 - 4), (x0 + w + 3, y0 + h + 3), frame, 4)
    img[y0:y0 + h, x0:x0 + w] = fill
    y = y0 + 16
    for n in lines:
        for c in range(n):
            x = x0 + 18 + c * 24
            cv2.rectangle(img, (x, y), (x + 14, y + 18), ink, 2)
        y += 28
    return img


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FlatCaptionBoxes(unittest.TestCase):
    def setUp(self):
        self.det = load("reletter_detect")

    def page(self, ground=(40, 90, 50)):
        img = np.full((700, 900, 3), ground, np.uint8)
        # textured art round the box: a flat patch of art fades into more of
        # itself, a box ends at its frame
        rng = np.random.default_rng(7)
        noise = rng.integers(-35, 35, img.shape)
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012 002.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def test_white_on_purple_is_a_dark_box(self):
        img = caption(self.page(), 200, 200, 240, 110, PURPLE, WHITE)
        found = self.det.flat_caption_boxes(img)
        self.assertEqual(len(found), 1)
        kind, _, (x, y, w, h) = found[0]
        self.assertEqual(kind, "dark")
        # inside the frame, holding the whole fill
        self.assertGreaterEqual(x, 196); self.assertLessEqual(x, 206)
        self.assertGreaterEqual(x + w, 434); self.assertLessEqual(x + w, 444)
        self.assertGreaterEqual(y + h, 304)

    def test_white_on_blue_is_a_dark_box(self):
        img = caption(self.page((20, 20, 20)), 150, 300, 240, 110, BLUE, WHITE)
        kinds = [k for k, _, _ in self.det.flat_caption_boxes(img)]
        self.assertEqual(kinds, ["dark"])

    def test_red_on_yellow_is_a_tint_box(self):
        img = caption(self.page(), 300, 150, 240, 110, YELLOW, RED, frame=RED)
        kinds = [k for k, _, _ in self.det.flat_caption_boxes(img)]
        self.assertEqual(kinds, ["tint"])

    def test_dark_letters_on_a_muted_fill_are_not_a_box(self):
        # tan art with dark marks: not vivid enough for the coloured pass
        img = caption(self.page(), 300, 150, 240, 110, (200, 170, 140), BLACK,
                      frame=BLACK)
        self.assertEqual(self.det.flat_caption_boxes(img), [])

    def test_box_replaces_entries_seeded_inside_it(self):
        img = caption(self.page(), 200, 200, 240, 110, PURPLE, WHITE)
        found = self.detect(img)
        inside = [e for e in found
                  if 190 <= e["bbox"][0] and e["bbox"][0] + e["bbox"][2] <= 450
                  and 190 <= e["bbox"][1] and e["bbox"][1] + e["bbox"][3] <= 320]
        self.assertEqual([e["kind"] for e in inside], ["dark"])


    def test_small_flat_patch_is_art(self):
        # a face or a sign fragment: fill and white marks, but no caption size
        img = caption(self.page(), 200, 200, 80, 34, PURPLE, WHITE, lines=(3,))
        self.assertEqual(self.det.flat_caption_boxes(img), [])

    def test_panel_holding_a_balloon_keeps_the_balloon(self):
        # a flat-coloured sign panel with white marks AND a paper balloon over
        # it: a caption box never holds a balloon, so nothing is replaced
        img = caption(self.page(), 150, 150, 520, 360, BLUE, WHITE,
                      lines=(9, 9, 9))
        cv2.ellipse(img, (400, 420), (150, 70), 0, 0, 360, BLACK, 4)
        cv2.ellipse(img, (400, 420), (146, 66), 0, 0, 360, WHITE, -1)
        for row, n in ((400, 7), (428, 6)):
            for c in range(n):
                x = 400 - n * 12 + c * 24
                cv2.rectangle(img, (x, row), (x + 14, row + 18), BLACK, 2)
        found = self.detect(img)
        self.assertTrue(any(e["kind"] == "bubble"
                            and e["bbox"][1] < 420 < e["bbox"][1] + e["bbox"][3]
                            for e in found))
        self.assertFalse(any(e["kind"] == "dark" and e["bbox"][2] > 400
                             for e in found))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class TintInk(unittest.TestCase):
    def setUp(self):
        self.fit = load("reletter_fit")

    def test_red_letters_on_yellow_are_ink(self):
        reg = caption(np.zeros((150, 280, 3), np.uint8), 10, 10, 260, 130,
                      YELLOW, RED, frame=YELLOW)
        ink = self.fit.tint_ink(reg)
        self.assertTrue(ink[26:44, 28].all())            # a letter's stem
        self.assertFalse(ink[20:140, 240:265].any())     # bare fill

    def test_navy_on_blue_unchanged(self):
        reg = caption(np.zeros((150, 280, 3), np.uint8), 10, 10, 260, 130,
                      (60, 130, 200), (15, 25, 60), frame=(60, 130, 200))
        old = reg.max(axis=2) <= 85
        self.assertTrue((self.fit.tint_ink(reg) == old).all())


if __name__ == "__main__":
    unittest.main()
