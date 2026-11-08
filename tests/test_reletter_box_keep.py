"""A caption box wiped WHOLE (clean_caption_box) must spare the coloured art
an override's `keep` names — a red drop cap printed over the box's corner
was wiped with the lettering, leaving half a red slab. The per-entry
cleaner already honoured `keep`; the box wipe did not.
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
class BoxWipeKeepsDropCap(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_fit
        reletter_fit.configure("unit-test-comic")
        self.fit = reletter_fit
        self.saved = dict(reletter_fit.LAYOUT_OVERRIDES)

    def tearDown(self):
        self.fit.LAYOUT_OVERRIDES.clear()
        self.fit.LAYOUT_OVERRIDES.update(self.saved)

    def page(self):
        img = np.full((300, 700, 3), (10, 8, 12), np.uint8)
        box = (50, 60, 650, 240)
        cv2.rectangle(img, box[:2], (box[2] - 1, box[3] - 1), (240, 180, 90), -1)
        # a red drop cap with a black outline, in the box's corner
        cv2.rectangle(img, (56, 66), (120, 170), (20, 20, 20), -1)
        cv2.rectangle(img, (61, 71), (115, 165), (200, 30, 20), -1)
        for cx in range(160, 620, 26):
            cv2.rectangle(img, (cx, 100), (cx + 18, 126), (20, 20, 20), -1)
        return img, box

    def test_keep_spares_the_drop_cap_and_the_lettering_goes(self):
        img, box = self.page()
        stem = "Series 13 - Title 012"
        self.fit.LAYOUT_OVERRIDES[self.fit.override_key(stem, 1)] = {
            "keep": [[52, 62, 124, 174]]}
        before = img.copy()
        self.fit.wipe_caption_boxes(img, stem, [(box, {1})])
        red = (before[..., 0] > 150) & (before[..., 1] < 60)
        self.assertGreater((img[red][:, 0] > 150).mean(), 0.98,
                           "the drop cap was wiped")
        self.assertLess((img[100:126, 160:620].min(axis=2) < 60).mean(), 0.02,
                        "the box's lettering survived")

    def test_without_keep_the_whole_box_is_wiped(self):
        img, box = self.page()
        self.fit.wipe_caption_boxes(img, "Series 13 - Title 013", [(box, {1})])
        # the red D (green ~30) is repainted with the orange fill (green ~180)
        self.assertLess((img[71:165, 61:115, 1] < 90).mean(), 0.5)

    def test_box_override_keeps_the_detected_wipe(self):
        """A `box` override is the TEXT area: the detected box is still
        wiped (a box narrowed to clear a drop cap leaves old letters beside
        the cap that only the detected box covers)."""
        stem = "Series 13 - Title 014"
        bubbles = [{"kind": "tint", "bbox": [96, 86, 444, 177],
                    "block": [96, 86, 444, 170]}]
        self.fit.LAYOUT_OVERRIDES[self.fit.override_key(stem, 1)] = {
            "box": [214, 88, 540, 280]}
        boxes = self.fit.boxes_to_wipe(bubbles, stem, {1})
        self.assertIn(((96, 86, 540, 263), {1}), boxes)
        self.assertIn(((214, 88, 540, 280), {1}), boxes)

    def test_a_wipe_override_replaces_the_detected_box(self):
        """A detected caption box that reaches up over the art above it
        painted a band of fill-coloured rows there; a `wipe` override names
        the region to wipe instead of it."""
        stem = "Series 13 - Title 015"
        bubbles = [{"kind": "tint", "bbox": [130, 54, 285, 183],
                    "block": [130, 54, 285, 175]}]
        self.assertTrue(self.fit.boxes_to_wipe(bubbles, stem, {1}),
                        "premise: the tint entry is a caption box")
        self.fit.LAYOUT_OVERRIDES[self.fit.override_key(stem, 1)] = {
            "box": [146, 102, 404, 236], "wipe": [136, 104, 404, 236]}
        boxes = self.fit.boxes_to_wipe(bubbles, stem, {1})
        self.assertEqual(boxes, [((146, 102, 404, 236), {1}),
                                 ((136, 104, 404, 236), {1})])


if __name__ == "__main__":
    unittest.main()
