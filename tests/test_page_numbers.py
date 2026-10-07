"""Unit tests for the page-number glue in process.py (the numbers file).
Run: .venv/bin/python -m unittest discover tests"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402

GEOMETRY = {"canvas": [1875, 2775],
            "boxes": [[187, 115, 1687, 2673], [27, 37, 1847, 2737]],
            "res": 300.0}


class NumbersFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.comic = self.root / "Comic"
        self.comic.mkdir()
        (self.comic / "Comic 002.jpg").write_bytes(b"")
        self.xcf = self.root / "xcf"
        self.xcf.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_header_carries_the_safety_line_the_number_sits_above(self):
        files = [Path("Comic 002.jpg"), Path("Comic 003-1.jpg")]
        with mock.patch.object(process, "template_geometry",
                               return_value=GEOMETRY), \
                mock.patch.object(process, "paper_color",
                                  return_value=(250, 248, 240)):
            out = process.write_numbers_file(files, {"Comic 002.jpg": 5},
                                             self.comic, self.xcf)
        lines = out.read_text().splitlines()
        header = dict(l.split() for l in lines[:3])
        # the number zone ends at the safety line, above the band the
        # printer's cut can reach — not at the trim
        self.assertEqual(header["SAFETY_BOTTOM"], "2673")
        self.assertEqual(header["STRIP_BOTTOM"], "2737")
        self.assertLess(int(header["ART_BOTTOM"]), 2673)
        self.assertEqual(lines[3], "Comic 002.jpg\t5\t250,248,240")
        # no source for the spread half: no number, no measured colour
        self.assertEqual(lines[4], "Comic 003-1.jpg\t-\t-")


if __name__ == "__main__":
    unittest.main()
