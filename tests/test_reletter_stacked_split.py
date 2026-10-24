"""Two balloons (or a caption box over another) stacked so close that their
lettering is nearer than paragraph_blocks' join: the lines of BOTH fuse into
one block, the block picks one balloon, and the other one's lines are never
an entry of their own (no key, no mask — the original lettering stays and
the transcript of the picked balloon has nowhere to put the rest). A block
whose letters sit in DIFFERENT light components, in groups that do not
interleave (stacked, diagonal, side by side), is two blocks. Synthetic pages only — no comic data.

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


def load_detect():
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_detect
    return reletter_detect


def word(img, x0, y0, n, step=22, lh=17):
    for c in range(n):
        x = x0 + c * step
        cv2.rectangle(img, (x, y0), (x + 14, y0 + lh), (10, 10, 10), 2)


def lines(img, cx, y0, counts, pitch=31):
    """Lines of n letters, 17 px tall, `pitch` apart (gap 14 px)."""
    for i, n in enumerate(counts):
        word(img, cx - (n * 22) // 2, y0 + i * pitch, n)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class StackedBalloonsSplit(unittest.TestCase):
    def setUp(self):
        self.det = load_detect()

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012 002.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def boxes_page(self, gap):
        """One frame cut in two by a rule: two caption boxes, one under the
        other. The upper's last line and the lower's first are `gap` px
        apart — no more than the upper's own lines (14) plus the rule."""
        img = np.full((700, 900, 3), 120, np.uint8)
        cv2.rectangle(img, (250, 200), (650, 390), (15, 15, 15), 4)
        cv2.rectangle(img, (253, 203), (647, 387), (252, 252, 252), -1)
        cv2.line(img, (250, 299), (650, 299), (15, 15, 15), 3)
        lines(img, 450, 210, [8, 9, 8])                 # ends y 289
        lines(img, 450, 289 + gap, [8, 7])
        return img

    def test_close_stacked_boxes_are_two_entries(self):
        found = self.detect(self.boxes_page(gap=20))
        self.assertEqual(len(found), 2)
        a, b = sorted(found, key=lambda e: e["block"][1])
        # each block stays inside its own box
        self.assertLessEqual(a["block"][1] + a["block"][3], 299)
        self.assertGreaterEqual(b["block"][1], 300)

    def test_one_box_five_lines_stays_one_block(self):
        img = np.full((700, 900, 3), 120, np.uint8)
        cv2.rectangle(img, (250, 200), (650, 381), (15, 15, 15), 4)
        cv2.rectangle(img, (253, 203), (647, 378), (252, 252, 252), -1)
        lines(img, 450, 210, [8, 9, 8, 8, 7])
        found = self.detect(img)
        self.assertEqual(len(found), 1)
        self.assertGreaterEqual(found[0]["block"][3], 4 * 31 + 17 - 4)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class SplitByBalloonUnit(unittest.TestCase):
    """The helper on its own: labels = two light components."""
    def setUp(self):
        self.det = load_detect()

    def test_stray_letters_join_the_nearest_group(self):
        letters = np.zeros((200, 400), np.uint8)
        labels = np.zeros((200, 400), np.int32)
        labels[0:100] = 1
        labels[100:200] = 2
        for i in range(10):                         # 10 letters in comp 1
            letters[60:75, 20 + 30 * i:32 + 30 * i] = 1
        for i in range(10):                         # 10 letters in comp 2
            letters[110:125, 20 + 30 * i:32 + 30 * i] = 1
        labels[82:100, 305:330] = 3                 # a stray third region
        letters[84:93, 310:324] = 1                 # ...one letter in it
        letters = np.where(letters > 0, 1, 0).astype(np.uint8)
        labels[letters > 0] = 0                     # letters are dark ink
        out = self.det.split_by_balloon([(15, 55, 380, 75)], letters, labels)
        self.assertEqual(len(out), 2)
        a, b = sorted(out, key=lambda r: r[1])
        self.assertLess(a[1] + a[3], 105)          # the stray kept with comp 1
        self.assertGreaterEqual(a[0] + a[2], 324)
        self.assertGreaterEqual(b[1], 105)

    def grid(self, comp_of):
        """Two rows of 12 letters; comp_of(col) = the paper under each."""
        letters = np.zeros((200, 800), np.uint8)
        labels = np.zeros((200, 800), np.int32)
        for c in range(12):
            x = 20 + 40 * c
            labels[:, x - 8:x + 32] = comp_of(c)
            for y in (60, 90):
                letters[y:y + 15, x:x + 12] = 1
        labels[letters > 0] = 0
        return letters, labels

    def test_side_by_side_neighbours_split(self):
        """A short balloon beside a big one, its line running on from the
        big one's: two balloons, two blocks."""
        letters, labels = self.grid(lambda c: 1 if c < 6 else 2)
        out = self.det.split_by_balloon([(10, 55, 480, 55)], letters, labels)
        self.assertEqual(len(out), 2)
        a, b = sorted(out)
        self.assertLess(a[0] + a[2], b[0] + 25)     # apart, but for padding

    def test_interleaved_groups_not_split(self):
        """Letters of two components mixed along the same lines are ONE
        text on a fill cut by art, not two balloons."""
        letters, labels = self.grid(lambda c: 1 + c % 2)
        blk = [(10, 55, 480, 55)]
        self.assertEqual(self.det.split_by_balloon(blk, letters, labels), blk)


if __name__ == "__main__":
    unittest.main()
