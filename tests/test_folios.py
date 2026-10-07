"""Unit tests for erasing the sources' own printed page numbers (folios).
Run: .venv/bin/python -m unittest discover tests"""
import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402

W, H = 1280, 1850
FONT = ImageFont.load_default(size=30)


def page(number=None, paper=(255, 255, 255), ink=(60, 60, 60), band=None,
         width=W, numbers_at=(0.5,), footer=None):
    """A story page: a framed panel of 'art', and optionally the original
    edition's page number centred under it (or under each spread half)."""
    im = Image.new("RGB", (width, H), paper)
    d = ImageDraw.Draw(im)
    d.rectangle((60, 60, width - 60, int(H * 0.94)), fill=(120, 150, 200),
                outline=(0, 0, 0), width=6)
    if band:
        d.rectangle((0, int(H * 0.95), width, H), fill=band)
    for k, cx in enumerate(numbers_at):
        if number is not None:
            d.text((int(width * cx), int(H * 0.972)), str(number + k),
                   font=FONT, fill=ink, anchor="mm")
    if footer:
        d.text((width // 2, int(H * 0.972)), footer, font=FONT, fill=ink,
               anchor="mm")
    return im


class Folios(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, name, im):
        p = self.dir / name
        im.save(p, quality=95)
        return p

    def test_labels_8_connected_components(self):
        m = np.zeros((10, 10), bool)
        m[1, 1:4] = True
        m[2, 4] = True          # diagonal neighbour: same component
        m[5:7, 5:7] = True
        sizes = sorted(len(ys) for ys, xs in process._label(m))
        self.assertEqual(sizes, [4, 4])

    def test_issue_key_is_the_name_minus_the_page_number(self):
        self.assertEqual(process._folio_issue("Comic 2 - Title 006"),
                         "Comic 2 - Title")
        self.assertEqual(process._folio_issue("Comic 2 - Title 001a cover"),
                         "Comic 2 - Title")

    def test_numbered_issue_is_found_and_erased_to_paper(self):
        pages = [self.save(f"Comic 1 - A {i:03d}.jpg", page(i + 2))
                 for i in range(4, 12)]
        plan, report = process.find_folios(pages)
        self.assertEqual(len(plan), len(pages))
        self.assertIn("8 of 8 pages", report[0])
        with Image.open(pages[0]) as im:
            out = np.asarray(process.erase_folios(im, plan[pages[0].name]),
                             dtype=int)
        strip = out[int(H * 0.95):, W // 2 - 80:W // 2 + 80]
        self.assertLessEqual(np.abs(strip - 255).max(), 12)
        # the panel border above is kept
        self.assertLess(out[int(H * 0.94) - 2, W // 2].max(), 80)

    def test_unnumbered_issue_with_specks_finds_nothing(self):
        # an issue's worth of pages with scan dust under the panels: some of
        # it digit-sized by chance, but never on enough pages at one height
        rng = random.Random(1)
        pages = []
        for i in range(40):
            im = page()
            d = ImageDraw.Draw(im)
            for _ in range(rng.randint(0, 2)):
                x, y = rng.randint(520, 760), rng.randint(int(H * 0.95), H - 40)
                d.ellipse((x, y, x + rng.randint(2, 14),
                           y + rng.randint(2, 34)), fill=(70, 70, 70))
            pages.append(self.save(f"Comic 2 - B {i + 3:03d}.jpg", im))
        plan, report = process.find_folios(pages)
        self.assertEqual(plan, {})
        self.assertEqual(report, [])

    def test_a_line_of_text_at_the_bottom_is_not_a_page_number(self):
        pages = [self.save(f"Comic 3 - C {i:03d}.jpg",
                           page(footer="CONTINUED 1N NEXT 155UE"))
                 for i in range(3, 11)]
        plan, _ = process.find_folios(pages)
        self.assertEqual(plan, {})

    def test_spreads_carry_one_number_under_each_half(self):
        pages = [self.save(f"Comic 4 - D {i:03d}.jpg", page(i + 2))
                 for i in range(4, 10)]
        pages.append(self.save("Comic 4 - D 010.jpg",
                               page(12, width=2 * W, numbers_at=(0.25, 0.75))))
        plan, _ = process.find_folios(pages)
        spread = plan["Comic 4 - D 010.jpg"]
        self.assertEqual(len(spread), 2)
        self.assertEqual(sorted(round(f["box"][0] / (2 * W), 1)
                                for f in spread), [0.2, 0.7])

    def test_white_numbers_on_a_black_band(self):
        pages = [self.save(f"Comic 5 - E {i:03d}.jpg",
                           page(i + 2, ink=(255, 255, 255), band=(0, 0, 0)))
                 for i in range(4, 12)]
        plan, _ = process.find_folios(pages)
        self.assertEqual(len(plan), len(pages))
        self.assertEqual(plan[pages[0].name][0]["pol"], "light")
        with Image.open(pages[0]) as im:
            out = np.asarray(process.erase_folios(im, plan[pages[0].name]),
                             dtype=int)
        self.assertLessEqual(out[int(H * 0.96):, W // 2 - 80:W // 2 + 80].max(),
                             12)


class FolioBandTrim(unittest.TestCase):
    """After the folio is erased, scan dust left in its band must not hold
    the crop open below the art."""

    def _erased(self, dust):
        im = page()                       # art ends at 0.94 H, no number
        d = ImageDraw.Draw(im)
        for x, y in dust:
            d.rectangle((x, y, x + 3, y), fill=(0, 0, 0))
        return im

    def test_dust_in_the_band_is_trimmed(self):
        # off the last row: ON it, it is an edge stripe and crop_to_content
        # drops it itself (EdgeStripeCrop)
        im = self._erased([(700, H - 6)])
        cropped = process.crop_to_content(im)
        self.assertEqual(cropped.height, H - 60 - 5)      # held open
        out = process.trim_folio_band(cropped, im)
        self.assertLessEqual(out.height, int(H * 0.94) - 60 + 2)

    def test_dust_outside_the_cropped_columns_is_trimmed(self):
        im = self._erased([(0, H - 4)])
        out = process.trim_folio_band(process.crop_to_content(im), im)
        self.assertLessEqual(out.height, int(H * 0.94) - 60 + 2)

    def test_side_speck_is_trimmed_when_the_bottom_is_already_tight(self):
        # a 1px speck down the side, in from the scan edge (so no edge
        # stripe), on a page whose bottom needs no trim
        im = page()
        ImageDraw.Draw(im).line((W - 12, 300, W - 12, 900), fill=(0, 0, 0))
        out = process.trim_folio_band(process.crop_to_content(im), im)
        self.assertEqual(out.width, W - 120 + 1)

    def test_real_content_below_a_gap_is_kept(self):
        im = page()
        ImageDraw.Draw(im).rectangle((200, H - 40, W - 200, H - 10),
                                     fill=(0, 0, 0))   # a caption strip
        cropped = process.crop_to_content(im)
        self.assertEqual(process.trim_folio_band(cropped, im).size,
                         cropped.size)


class EdgeStripeCrop(unittest.TestCase):
    """A scan-edge stripe (the gutter shadow, the scanner lid's edge) is a
    few px wide, runs along the scan's edge and sits a gap of paper away
    from the art. It must not hold the crop open — on its own axis OR on
    the other one, where it holds every row it runs down."""

    ART = (60, 60, W - 60, int(H * 0.94))

    def test_left_stripe_down_the_top_is_dropped_on_both_axes(self):
        im = page()                      # art box = ART
        ImageDraw.Draw(im).rectangle((0, 0, 7, 555), fill=(215, 215, 215))
        self.assertEqual(process.crop_to_content(im).size,
                         (self.ART[2] - self.ART[0] + 1,
                          self.ART[3] - self.ART[1] + 1))

    def test_one_px_stripes_on_every_edge_are_dropped(self):
        im = page()
        d = ImageDraw.Draw(im)
        for box in ((W - 1, 0, W - 1, H - 1), (0, 0, W - 1, 0),
                    (0, H - 2, W - 1, H - 1)):
            d.rectangle(box, fill=(150, 150, 150))
        self.assertEqual(process.crop_to_content(im).size,
                         (self.ART[2] - self.ART[0] + 1,
                          self.ART[3] - self.ART[1] + 1))

    def test_full_bleed_art_and_wide_edge_content_are_kept(self):
        im = page()
        d = ImageDraw.Draw(im)
        d.rectangle((0, 300, 40, 900), fill=(0, 0, 0))   # 41px: art, not a stripe
        self.assertEqual(process.crop_to_content(im).size[0],
                         self.ART[2] + 1)
        bleed = Image.new("RGB", (W, H), (120, 150, 200))   # art to every edge
        self.assertEqual(process.crop_to_content(bleed).size, (W, H))


class ReletteredPagesKeepTheirUpscale(unittest.TestCase):
    """Once a book is fitted, its upscaled/ pages ARE the cleaned working
    pages: a crop or sizing rule that moves one must not re-upscale it in
    place (layout.json, the masks and the positional transcripts all stay on
    the old geometry). It is reported for a deliberate rebuild instead."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.comic = root / "Book"
        self.out = root / "upscaled" / "Book"
        self.comic.mkdir()
        self.out.mkdir(parents=True)
        page().save(self.comic / "Book 001.jpg", quality=95)
        page().save(self.comic / "Book 002.jpg", quality=95)
        Image.new("RGB", (10, 10)).save(self.out / "Book 001.jpg")  # wrong size
        self.reletter = root / "relettering"
        self.calls = []
        self.patches = [(process, "REPO", root),     # folios.json cache
                        (process, "RELETTER_DIR", self.reletter),
                        (process, "REALESRGAN", self.comic / "Book 001.jpg"),
                        (process, "upscale_realesrgan",
                         lambda src, out: self.calls.append(src.name))]
        self.saved = [(o, n, getattr(o, n)) for o, n, _ in self.patches]
        for o, n, v in self.patches:
            setattr(o, n, v)

    def tearDown(self):
        for o, n, v in self.saved:
            setattr(o, n, v)
        self.tmp.cleanup()

    def test_unfitted_book_is_re_upscaled(self):
        self.assertEqual(process.cmd_upscale(self.comic, self.out), 0)
        self.assertEqual(sorted(self.calls), ["Book 001.jpg", "Book 002.jpg"])

    def test_fitted_book_keeps_its_working_page(self):
        (self.reletter / "Book").mkdir(parents=True)
        (self.reletter / "Book" / "layout.json").write_text("{}")
        self.assertEqual(process.cmd_upscale(self.comic, self.out), 0)
        self.assertEqual(self.calls, ["Book 002.jpg"])   # missing: produced
        with Image.open(self.out / "Book 001.jpg") as im:
            self.assertEqual(im.size, (10, 10))         # untouched


if __name__ == "__main__":
    unittest.main()
