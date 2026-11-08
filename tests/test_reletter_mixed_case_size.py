"""The original's cap height on MIXED-CASE lettering. `old_cap_height` was
the median letter-blob height — on capitals that is the cap height, but in
mixed case most blobs are x-height letters, so the "original size" read the
x-height: the size ceiling max(book, original) then never rose above the
book size and the new capitals printed smaller than the original's.
Synthetic pages only — no comic data.

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

FONT = cv2.FONT_HERSHEY_SIMPLEX if HAVE_CV2 else None


def load_fit():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_fit
    reletter_fit.configure("unit-test-comic")
    return reletter_fit


def block(lines, scale=1.25, thick=3, pitch=50):
    img = np.full((60 + pitch * len(lines), 700, 3), 250, np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (20, 50 + k * pitch), FONT, scale, (20, 20, 20),
                    thick, cv2.LINE_AA)
    return img, {"kind": "bubble",
                 "block": [10, 10, 680, 40 + pitch * len(lines)]}


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class MixedCaseCapHeight(unittest.TestCase):
    def setUp(self):
        self.F = load_fit()

    def caps(self):
        img, b = block(["HALLARE EL PUENTE", "Y DETENDRE LA NAVE."])
        return self.F.old_cap_height(img, b)

    def test_premise_x_height_is_well_under_the_caps(self):
        img = np.full((80, 300, 3), 250, np.uint8)
        cv2.putText(img, "xxxx", (20, 50), FONT, 1.25, (20, 20, 20), 3)
        lm = img.min(axis=2) < 128
        ys = np.nonzero(lm.any(axis=1))[0]
        self.assertLess(ys.max() - ys.min() + 1, 0.85 * self.caps())

    def test_mixed_case_reads_the_cap_height_not_the_x_height(self):
        img, b = block(["Hallare el puente", "y detendre la nave."])
        self.assertAlmostEqual(self.F.old_cap_height(img, b), self.caps(),
                               delta=3)

    def test_a_short_mixed_line_keeps_the_median(self):
        # too few letters to tell a cluster of capitals from chance
        img, b = block(["Oh, no."])
        h = self.F.old_cap_height(img, b)
        self.assertGreater(h, 0)
        self.assertLess(h, self.caps())


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class IssueMixedCaps(unittest.TestCase):
    """Short balloons of a mixed-case issue take the issue's cap height."""
    def setUp(self):
        self.F = load_fit()

    @staticmethod
    def page(stem, entries):
        pend = [{"b": {"kind": k}, "oldh": h, "mixed": m}
                for k, h, m in entries]
        return (Path(stem + ".jpg"), pend, [], None, None)

    def test_short_balloons_of_a_mixed_issue_rise(self):
        allp = [self.page("Series 2.01 - Title 002",
                          [("bubble", 28.0, True), ("bubble", 21.0, False)]),
                self.page("Series 2.01 - Title 003",
                          [("bubble", 30.0, True), ("bubble", 29.0, True),
                           ("margin", 15.0, False)]),
                self.page("Series 2.02 - Title 002",
                          [("bubble", 21.0, False)])]
        self.F.issue_mixed_caps(allp)
        self.assertEqual(allp[0][1][1]["oldh"], 29.0)   # raised
        self.assertEqual(allp[0][1][0]["oldh"], 29.0)
        self.assertEqual(allp[1][1][2]["oldh"], 15.0)   # margin note: no
        self.assertEqual(allp[2][1][0]["oldh"], 21.0)   # other issue: no

    def test_every_mixed_balloon_takes_the_issue_size(self):
        # one letterer, one size: a balloon whose accents or descenders
        # read a few px tall must not be set bigger than its neighbours;
        # a shout in capitals (not mixed) keeps its own, larger size
        allp = [self.page("Series 2.01 - Title 002",
                          [("bubble", 28.0, True), ("bubble", 28.0, True),
                           ("bubble", 34.0, True), ("bubble", 36.0, False)])]
        self.F.issue_mixed_caps(allp)
        self.assertEqual([p["oldh"] for p in allp[0][1]],
                         [28.0, 28.0, 28.0, 36.0])

    def test_a_single_page_rebuild_sees_the_whole_issue(self):
        """The issue's other pages, outside the run, are measured from
        their pristine copies — or a one-page rebuild sizes its page
        smaller than the rest of its story."""
        import json
        import tempfile
        F = self.F
        saved = (F.REPO, F.COMIC, F.BUB)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            work = root / "relettering" / "Series"
            (work / "pristine").mkdir(parents=True)
            (work / "bubbles").mkdir()
            img, b = block(["Hallare el puente", "y detendre la nave."])
            caps = F.cap_measure(img, b)
            self.assertTrue(caps[1])
            trans = {}
            for n in ("002", "003", "004"):
                stem = f"Series 2.01 - Title {n}"
                cv2.imwrite(str(work / "pristine" / f"{stem}.jpg"), img)
                (work / "bubbles" / f"{stem}.json").write_text(
                    json.dumps([b]))
                trans[stem] = ["Hallare el puente y detendre la nave."]
            try:
                F.REPO, F.COMIC, F.BUB = root, "Series", work / "bubbles"
                allp = [self.page("Series 2.01 - Title 005",
                                  [("bubble", 19.0, False)])]
                F.issue_mixed_caps(allp, trans)
            finally:
                F.REPO, F.COMIC, F.BUB = saved
        self.assertAlmostEqual(allp[0][1][0]["oldh"], caps[0], delta=1)

    def test_too_few_mixed_balloons_change_nothing(self):
        allp = [self.page("Series 2.01 - Title 002",
                          [("bubble", 28.0, True), ("bubble", 29.0, True),
                           ("bubble", 21.0, False)])]
        self.F.issue_mixed_caps(allp)
        self.assertEqual(allp[0][1][2]["oldh"], 21.0)


if __name__ == "__main__":
    unittest.main()
