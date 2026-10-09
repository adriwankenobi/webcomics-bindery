"""The straight cuts that split two joined balloons (detection's merge and
notch stages). Synthetic masks only — no comic data.

    .venv-reletter/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def load_detect():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_detect
    return reletter_detect


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CutMask(unittest.TestCase):
    """`cut_mask` zeroes one side of a mask at a PAGE coordinate, so it has
    to subtract the mask's own origin on THAT axis. It used to subtract the
    other one (bbox x for a row cut): a cut meant for y=541 on a mask at
    (295, 354) landed at y=600, inside the lower balloon's first line, which
    then belonged to no mask and was never cleaned."""

    def setUp(self):
        self.det = load_detect()

    def test_row_cut_lands_on_the_page_row(self):
        mask = np.ones((400, 350), np.uint8)
        bbox = [295, 354, 350, 400]             # x and y origins differ
        upper = self.det.cut_mask(mask, bbox, 541, 0, keep_before=True)
        lower = self.det.cut_mask(mask, bbox, 541, 0, keep_before=False)
        rows_u = np.flatnonzero(upper.any(axis=1)) + bbox[1]
        rows_l = np.flatnonzero(lower.any(axis=1)) + bbox[1]
        self.assertEqual(rows_u.max(), 540)
        self.assertEqual(rows_l.min(), 541)

    def test_column_cut_lands_on_the_page_column(self):
        mask = np.ones((300, 400), np.uint8)
        bbox = [120, 700, 400, 300]
        left = self.det.cut_mask(mask, bbox, 300, 1, keep_before=True)
        cols = np.flatnonzero(left.any(axis=0)) + bbox[0]
        self.assertEqual(cols.max(), 299)

    def test_cut_that_leaves_almost_nothing_is_skipped(self):
        mask = np.ones((100, 100), np.uint8)
        bbox = [0, 0, 100, 100]
        self.assertIsNone(self.det.cut_mask(mask, bbox, 95, 0,
                                            keep_before=False))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class NotchCut(unittest.TestCase):
    """`notch_cut` picks the straight cut between two joined lobes: the
    narrowest row (column) of their union between the two text blocks."""

    def setUp(self):
        self.det = load_detect()

    def test_waist_between_the_blocks_is_the_cut(self):
        # widths: wide lobe, a neck at 50, wide lobe
        prof = np.array([300] * 40 + [40] * 3 + [20] + [40] * 3 + [300] * 40)
        self.assertEqual(self.det.notch_cut(prof, 35, 52, 38, 50), 43)

    def test_no_waist_means_no_straight_cut(self):
        # a DIAGONAL pair: the union widens where the second lobe comes in
        # and narrows where the first ends — the narrowest row of the window
        # is its far end, 10px inside the lower lobe's own lettering, and
        # cutting there left its first line outside every mask
        prof = np.array([401] * 6 + list(range(402, 578, 4))
                        + list(range(577, 515, -4)))
        n = len(prof)
        self.assertIsNone(self.det.notch_cut(prof, 0, n, 10, n - 11))

    def test_cut_never_lands_inside_a_block(self):
        # the narrowest row sits 4px inside the lower block: the cut moves
        # to the block's edge, never into its lettering
        prof = np.array([300] * 10 + [200] * 10 + [150] * 6 + [90]
                        + [200] * 10)
        k = self.det.notch_cut(prof, 0, len(prof), 12, 22)
        self.assertEqual(k, 22)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FollowLobeSpike(unittest.TestCase):
    """A burst balloon: at the block's centre row a long spike runs far out
    to one side, so that row's run is centred well off the text. Measured
    from it alone, the rows above and below the spike "drift" and the walk
    stopped inside the lettering — the burst's first line was never
    cleaned and stood behind the new one."""

    def setUp(self):
        self.det = load_detect()

    def test_walk_is_not_stopped_by_a_spike_at_the_seed_row(self):
        comp = np.zeros((160, 260), np.uint8)
        comp[20:141, 90:251] = 1            # the body
        comp[60:81, 0:251] = 1              # the spike, at the seed row
        walk = self.det.follow_lobe(comp, 70, (110, 230), 120, text_drift=True)
        rows = np.flatnonzero(walk.any(axis=1))
        self.assertEqual((rows.min(), rows.max()), (20, 140))

    def test_a_real_leak_still_stops_the_walk(self):
        comp = np.zeros((160, 400), np.uint8)
        comp[20:100, 90:251] = 1            # the balloon
        # a leak that walks steadily off to the right below it
        for k, yy in enumerate(range(100, 160)):
            comp[yy, 90 + 3 * k:251 + 3 * k] = 1
        walk = self.det.follow_lobe(comp, 60, (110, 230), 140, text_drift=True)
        self.assertLess(np.flatnonzero(walk.any(axis=1)).max(), 140)


def hollow_letters(img, x0, y0, cols, rows, step=22, lh=20, gap=28):
    for r in range(rows):
        for c in range(cols):
            x, y = x0 + c * step, y0 + r * gap
            cv2.rectangle(img, (x, y), (x + 14, y + lh), (10, 10, 10), 3)
    return [x0, y0, (cols - 1) * step + 15, (rows - 1) * gap + lh + 1]


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class JoinedStack(unittest.TestCase):
    """Three balloons joined in a stack (one white shape): every lobe's
    first mask is the union of all three, so the bottom lobe "overlaps" the
    top one too. It must pair with its neighbour, never across the middle
    lobe."""

    def setUp(self):
        self.det = load_detect()

    def test_each_lobe_pairs_with_its_neighbour(self):
        import tempfile
        img = np.full((700, 700, 3), (100, 140, 180), np.uint8)
        centres = [(330, 130), (350, 320), (330, 510)]
        for c in centres:
            cv2.ellipse(img, c, (234, 109), 0, 0, 360, (0, 0, 0), -1)
        for c in centres:
            cv2.ellipse(img, c, (230, 105), 0, 0, 360, (255, 255, 255), -1)
        for c in centres:
            hollow_letters(img, c[0] - 110, c[1] - 45, 11, 4)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 7 - Title 050.png"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            _, bubbles = self.det.detect_page(p)
        self.assertEqual(len(bubbles), 3)
        g = [b.get("group") for b in bubbles]
        self.assertNotEqual(g[0], g[2], "top and bottom lobes paired "
                                        "across the middle one")
        self.assertEqual(g[1], g[2])


if __name__ == "__main__":
    unittest.main()
