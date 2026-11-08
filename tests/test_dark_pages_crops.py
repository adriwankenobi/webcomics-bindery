"""Unit tests for dark-paper pages (page kind) and the per-page crop override.
Run: .venv/bin/python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

W, H = 900, 1400


def dark_story_page():
    """A story page printed on black: mostly black paper, two framed panels
    of dark art, a white balloon. Every row of black paper reads as 'ink' to
    a light-paper test."""
    im = Image.new("RGB", (W, H), (4, 4, 6))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 80, W - 60, 600), fill=(16, 10, 22), outline=(120, 120, 140))
    d.rectangle((60, 640, W - 60, 1300), fill=(12, 9, 16), outline=(120, 120, 140))
    d.ellipse((500, 120, 760, 260), fill=(250, 250, 250))
    return im


def sketch_page():
    """An extras page on white paper: tall runs of dark strokes."""
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for x in range(150, 750, 40):
        d.line((x, 200, x + 30, 1200), fill=(20, 20, 20), width=3)
    return im


class DarkPaperIsNeverAnArtPage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, im, name):
        p = self.root / name
        im.save(p)
        return p

    def test_a_story_page_printed_on_black_is_a_regular_page(self):
        # it used to come back 'art': no page number, and detection skipped
        # it, so its balloons were never re-lettered
        p = self.save(dark_story_page(), "Comic 002.png")
        self.assertIsNone(process.paper_page_kind(p))

    def test_a_sketch_page_on_white_paper_is_still_art(self):
        p = self.save(sketch_page(), "Comic 003.png")
        self.assertEqual(process.paper_page_kind(p), "art")

    @unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
    def test_detection_agrees_with_the_pipeline(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        for im, name in ((dark_story_page(), "a.png"), (sketch_page(), "b.png")):
            p = self.save(im, name)
            self.assertEqual(reletter_detect.paper_page_kind(p),
                             process.paper_page_kind(p))


class UpscaleOverrides(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.comic = self.root / "Comic"
        self.comic.mkdir()
        # black page margin + the next page's edge at the right: the content
        # crop keeps them all
        im = dark_story_page()
        ImageDraw.Draw(im).rectangle((W - 25, 0, W - 1, H - 1),
                                     fill=(200, 160, 120))
        self.src = self.comic / "Comic 004.jpg"
        im.save(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, entries):
        import json
        up = self.root / "upscaled" / "Comic"
        up.mkdir(parents=True, exist_ok=True)
        (up / process.UPSCALE_OVERRIDES).write_text(json.dumps(entries))

    def spec(self, entries):
        self.write(entries)
        with mock.patch.object(process, "folio_plan", return_value={}), \
                mock.patch.object(process, "REPO", self.root):
            return process.upscale_spec(self.src)

    def test_the_crop_rectangle_replaces_the_content_crop(self):
        s = self.spec({"Comic 004.jpg": {"crop": [60, 80, W - 59, 1301]}})
        self.assertEqual(s["img"].size, (W - 59 - 60, 1301 - 80))
        self.assertIn("crop override", s["note"])

    def test_pages_without_an_override_keep_the_content_crop(self):
        auto = self.spec({})
        self.assertEqual(auto["img"].size[0], W - 60)   # held open by the edge
        other = self.spec({"Comic 005.jpg": {"crop": [0, 0, 10, 10]}})
        self.assertEqual(other["img"].size, auto["img"].size)
        self.assertEqual(other["model"], process.REALESRGAN_MODEL)

    def test_a_pattern_covers_a_story_and_the_exact_name_wins(self):
        s = self.spec({"Comic 0*.jpg": {"model": "anime", "crop": [0, 0, 50, 60]},
                       "Comic 004.jpg": {"crop": [0, 0, 100, 120]}})
        self.assertEqual(s["model"], "anime")
        self.assertEqual(s["img"].size, (100, 120))

    def test_rotate_deskews_before_the_crop(self):
        # a frame line tilted by the scan: after the rotation it is level
        im = Image.new("RGB", (W, H), (0, 0, 0))
        ImageDraw.Draw(im).line((100, 300, 800, 300 - 12), fill=(200, 200, 200),
                                width=3)
        im.save(self.src)
        import math
        ang = math.degrees(math.atan2(12, 700))
        s = self.spec({"Comic 004.jpg": {"rotate": -ang, "crop": [0, 0, W, H]}})
        import numpy as np
        a = np.asarray(s["img"].convert("L"))
        ys = [int(np.argmax(a[:, x])) for x in (200, 450, 700)]
        self.assertLessEqual(max(ys) - min(ys), 1)

    def test_unknown_fields_are_refused(self):
        with self.assertRaises(ValueError):
            self.spec({"Comic 004.jpg": {"crops": [0, 0, 1, 1]}})

    def test_levels_stretch_a_greyed_black_and_keep_white(self):
        im = Image.new("RGB", (200, 200), (22, 18, 26))
        ImageDraw.Draw(im).rectangle((50, 50, 150, 150), fill=(250, 250, 250))
        out = process.apply_levels(im)
        self.assertLess(max(out.getpixel((5, 5))), 6)
        self.assertGreater(min(out.getpixel((100, 100))), 250)

    def test_overrides_file_is_read_by_nfc_name(self):
        import json
        import unicodedata
        self.write({unicodedata.normalize("NFD", "Comic ó 004.jpg"):
                    {"crop": [1, 2, 3, 4]}})
        with mock.patch.object(process, "REPO", self.root):
            got = process.upscale_overrides(self.comic)
        self.assertEqual(got, {"Comic ó 004.jpg": {"crop": [1, 2, 3, 4]}})


if __name__ == "__main__":
    unittest.main()
