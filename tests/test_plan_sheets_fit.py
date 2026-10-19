"""Sheets and fit follow the keep-or-re-letter plan: a kept issue gets no
sheets (so it is never transcribed) and no fit; a later flip is synced
without renumbering existing sheets or refitting laid-out pages.

    .venv/bin/python -m unittest discover tests        (pure parts)
    .venv-reletter/bin/python -m unittest discover tests  (all)
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "relettering"))

import make_sheets  # noqa: E402

try:
    import cv2
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

KEPT = "Series 6.01 002"
RELET = "Series 6.28 004"
NEW = "Series 6.28 005"
EMPTY = "Series 6.28 001"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


def decisions(work, keep=("6.01",), relet=("6.28",)):
    write_json(work / "issue_decisions.json", {
        **{i: {"decision": "keep", "unsure": False} for i in keep},
        **{i: {"decision": "reletter", "unsure": False} for i in relet}})


class PlanPages(unittest.TestCase):
    def test_kept_and_empty_pages_have_no_sheet(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            for stem, n in ((KEPT, 2), (RELET, 3), (EMPTY, 0)):
                write_json(work / "bubbles" / f"{stem}.json", [{}] * n)
            decisions(work)
            self.assertEqual(make_sheets.plan_pages(work / "bubbles", work),
                             {RELET: 3})


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class SheetSync(unittest.TestCase):
    def test_sync_drops_kept_and_appends_new_pages(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            bub = work / "bubbles"
            for stem in (KEPT, RELET, NEW):
                write_json(bub / f"{stem}.json", [{}])
                cv2.imwrite(str(bub / f"{stem}-b01.png"),
                            np.full((40, 60, 3), 255, np.uint8))
            sheets = work / "sheets"
            sheets.mkdir()
            for n in (1, 2):
                cv2.imwrite(str(sheets / f"sheet-{n:03d}.jpg"),
                            np.full((10, 10, 3), 255, np.uint8))
            write_json(sheets / "manifest.json", {
                make_sheets.short_key(KEPT): {"stem": KEPT, "count": 1},
                make_sheets.short_key(RELET): {"stem": RELET, "count": 1}})
            decisions(work)
            make_sheets.build(work)
            man = json.loads((sheets / "manifest.json").read_text())
            self.assertEqual(sorted(v["stem"] for v in man.values()),
                             [RELET, NEW])
            self.assertEqual(sorted(p.name for p in sheets.glob("*.jpg")),
                             ["sheet-001.jpg", "sheet-002.jpg",
                              "sheet-003.jpg"])

    def test_render_sheets_numbers_from_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            tiles = [np.full((30, 40, 3), 200, np.uint8)] * 3
            n = make_sheets.render_sheets(tiles, out, 7)
            self.assertEqual(n, 1)
            self.assertTrue((out / "sheet-007.jpg").is_file())


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FitHelpers(unittest.TestCase):
    def load(self):
        import reletter_fit
        if reletter_fit.COMIC is None:
            reletter_fit.configure("unit-test-comic")
        return reletter_fit

    def test_stems_to_add(self):
        # "done" = the pages the layout holds
        F = self.load()
        self.assertEqual(F.stems_to_add(["A"], ["C", "A", "B"], {"C"}),
                         ["B"])

    def test_fit_meta_only_where_the_plan_lives(self):
        """An approved comic (no decisions) never gets a new file from a
        re-fit; a planned one records its book size for --add."""
        F = self.load()
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            F.write_fit_meta(work, 20)
            self.assertFalse((work / "fit_meta.json").exists())
            decisions(work)
            F.write_fit_meta(work, 20)
            self.assertEqual(
                json.loads((work / "fit_meta.json").read_text()),
                {"book": 20})

    def test_merge_layout_never_overwrites(self):
        F = self.load()
        with self.assertRaises(ValueError):
            F.merge_layout({"A": []}, {"A": []})
        self.assertEqual(F.merge_layout({"A": [1]}, {"B": [2]}),
                         {"A": [1], "B": [2]})


if __name__ == "__main__":
    unittest.main()
