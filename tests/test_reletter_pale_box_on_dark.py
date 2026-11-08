"""A PALE caption box (dark letters on light blue, navy frame) on a dark
page. The white-on-black caption pass reads the pale gaps between the dark
letters as white "letters", and the box passes its solid-dark test — dark
pixels and white letters together cover it — so the box became a `dark`
entry and the clean painted it black. A white-on-black caption is mostly
dark INSIDE its text block; dark lettering on a pale fill is mostly light.
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

FONT = cv2.FONT_HERSHEY_SIMPLEX if HAVE_CV2 else None


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class PaleBoxOnDarkPage(unittest.TestCase):
    def test_premise_the_halos_read_as_white_letters(self):
        import reletter_detect as D
        img = self.page((190, 212, 253), (15, 15, 20), (25, 25, 110),
                        halos=True)
        fl = D.flat_letters(img)
        self.assertGreater((fl == 1).sum(), 0)

    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def page(self, fill, ink, frame, halos=False):
        img = np.full((700, 900, 3), (40, 60, 150), np.uint8)   # dark blue
        cv2.rectangle(img, (80, 60), (520, 170), frame, -1)
        cv2.rectangle(img, (92, 72), (508, 158), fill, -1)
        cv2.putText(img, "Mas tarde, en", (110, 130), FONT, 1.4, ink, 3,
                    cv2.LINE_AA)
        if halos:   # the bright slivers an upscale leaves beside strokes
            for x in range(118, 460, 22):
                cv2.ellipse(img, (x, 92), (3, 7), 0, 0, 360,
                            (252, 252, 252), -1)
        return img

    def test_dark_letters_on_a_pale_box_are_not_a_white_letter_box(self):
        img = self.page((190, 212, 253), (15, 15, 20), (25, 25, 110),
                        halos=True)
        kinds = [k for k, _, _ in self.det.flat_caption_boxes(img)]
        self.assertNotIn("dark", kinds)

    def test_white_letters_on_a_purple_box_still_are(self):
        img = self.page((120, 40, 150), (250, 250, 250), (60, 20, 80))
        kinds = [k for k, _, _ in self.det.flat_caption_boxes(img)]
        self.assertIn("dark", kinds)


if __name__ == "__main__":
    unittest.main()
