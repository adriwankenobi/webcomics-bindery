"""Unit tests for the side-by-side comparison PDF (process.cmd_compare).
Run: .venv/bin/python -m unittest discover tests"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

try:
    import pikepdf  # only in .venv
    HAVE_PIKEPDF = True
except ImportError:
    HAVE_PIKEPDF = False

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402

PAGE_PT = (450.0, 666.0)  # a 1875x2775 page at 300dpi


def page_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1875, 2775), (200, 30, 30)).save(path, resolution=300)


@unittest.skipUnless(HAVE_PIKEPDF, "pikepdf only in .venv")
class ComparePdf(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.comic = root / "Series 13 - Title 012"
        self.pdf_dir = root / "pdf"
        self.comic.mkdir()
        # a JPEG page, a PNG page with transparency, and a horizontal spread
        Image.new("RGB", (800, 1200), (10, 120, 200)).save(
            self.comic / "Title 012 001.jpg", quality=90)
        Image.new("RGBA", (600, 900), (0, 0, 0, 0)).save(
            self.comic / "Title 012 002.png")
        Image.new("RGB", (2400, 1200), (40, 40, 40)).save(
            self.comic / "Title 012 003.JPG", quality=90)
        self.stems = ["Title 012 001", "Title 012 002",
                      "Title 012 003-1", "Title 012 003-2"]
        for s in self.stems:
            page_pdf(self.pdf_dir / (s + ".pdf"))
        self.plan = [Path(s + ".png") for s in self.stems[:2]] + [None] + \
            [Path(s + ".png") for s in self.stems[2:]]
        self.out = root / "comparison.pdf"

    def tearDown(self):
        self.tmp.cleanup()

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = process.cmd_compare(self.pdf_dir, self.out, None,
                                     self.comic, plan=self.plan)
        self.assertEqual(rc, 0)
        return pikepdf.open(self.out)

    @staticmethod
    def size(page):
        x0, y0, x1, y1 = [float(v) for v in page.mediabox]
        return x1 - x0, y1 - y0

    @staticmethod
    def images(page):
        return [pikepdf.PdfImage(x) for x in page.get_images().values()]

    def test_one_comparison_page_per_book_page(self):
        with self.build() as pdf:
            self.assertEqual(len(pdf.pages), len(self.plan))

    def test_original_left_at_page_height_then_final_page(self):
        w, h = PAGE_PT
        with self.build() as pdf:
            pw, ph = self.size(pdf.pages[0])
            self.assertAlmostEqual(ph, h, places=2)
            self.assertAlmostEqual(pw, h * 800 / 1200 + process.COMPARE_GAP
                                   + w, places=2)

    def test_jpeg_original_goes_in_byte_for_byte(self):
        with self.build() as pdf:
            raw = [i.obj.read_raw_bytes() for i in self.images(pdf.pages[0])]
        self.assertIn((self.comic / "Title 012 001.jpg").read_bytes(), raw)

    def test_non_jpeg_original_is_lossless_on_white(self):
        with self.build() as pdf:
            ims = [i for i in self.images(pdf.pages[1]) if i.width == 600]
            self.assertEqual(len(ims), 1)
            px = ims[0].as_pil_image().convert("RGB")
        self.assertEqual(px.getextrema(), ((255, 255),) * 3)

    def test_white_page_stays_white(self):
        with self.build() as pdf:
            blank = pdf.pages[2]
            self.assertEqual(self.size(blank), PAGE_PT)
            self.assertEqual(self.images(blank), [])

    def test_spread_halves_show_the_whole_spread(self):
        h = PAGE_PT[1]
        with self.build() as pdf:
            for page in pdf.pages[3:]:
                pw, _ = self.size(page)
                self.assertAlmostEqual(pw, 2 * h + process.COMPARE_GAP
                                       + PAGE_PT[0], places=2)

    def test_missing_page_pdf_is_refused(self):
        (self.pdf_dir / "Title 012 002.pdf").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = process.cmd_compare(self.pdf_dir, self.out, None,
                                     self.comic, plan=self.plan)
        self.assertEqual(rc, 1)
        self.assertFalse(self.out.exists())


class CompareName(unittest.TestCase):
    def test_sits_next_to_the_book(self):
        self.assertEqual(
            process.compare_path(Path("pdf/Series 13.pdf")),
            Path("pdf/Series 13 - comparison.pdf"))


if __name__ == "__main__":
    unittest.main()
