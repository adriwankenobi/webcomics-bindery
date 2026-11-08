"""Mixed-case lettering set large: a line of x-height-only words ("nave.",
"que se") starts an ascender's height lower than a capital would, so the
gap from the line above runs past paragraph_blocks' vertical join and the
balloon comes out as TWO entries (or its last short line is left to the
short-utterance pass as an entry of its own). Lines stacked in the same
balloon, overlapping, with an ink-free gap of at most about a letter height
are one block.
And a stray mark just outside the balloon (a ship in the sky beside the
text) joined to its lines must not pull the block — and with it the bubble
mask — out over the art.
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

SKY = (205, 225, 245)          # light art: the same light class as paper
INK = (20, 40, 50)
FONT = cv2.FONT_HERSHEY_SIMPLEX if HAVE_CV2 else None
SCALE, THICK = 1.25, 3
PITCH = 50                     # baseline to baseline
AX = 180                       # balloon half-width: text near the outline


def text_line(img, txt, cx, base):
    (w, _), _ = cv2.getTextSize(txt, FONT, SCALE, THICK)
    cv2.putText(img, txt, (cx - w // 2, base), FONT, SCALE, INK, THICK,
                cv2.LINE_AA)


def balloon(img, cx, cy, ax, ay, lines, top_base):
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (252, 252, 252), -1)
    cv2.ellipse(img, (cx, cy), (ax, ay), 0, 0, 360, (10, 10, 10), 5)
    for k, t in enumerate(lines):
        text_line(img, t, cx, top_base + k * PITCH)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class MixedCaseLines(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_detect
        self.det = reletter_detect

    def detect(self, img):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "Series 13 - Title 012.jpg"
            cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            return self.det.detect_page(p)[1]

    def three_lines(self, ships=False):
        img = np.full((900, 1000, 3), SKY, np.uint8)
        balloon(img, 330, 300, AX, 120,
                ["Hallare el puente", "y detendre la", "nave."], 262)
        if ships:   # two dark marks in the sky just right of the outline
            for x, y in ((330 + AX + 14, 250), (330 + AX + 42, 290)):
                cv2.ellipse(img, (x, y), (14, 10), 0, 0, 360, INK, -1)
        return img

    def test_premise_last_line_is_not_joined(self):
        lm = self.det.letter_mask(self.three_lines())
        blocks = self.det.paragraph_blocks(lm)
        self.assertEqual(len(blocks), 1)
        bx, by, bw, bh = blocks[0]
        self.assertLess(by + bh, 262 + 2 * PITCH - 20)   # "nave." left out

    def test_three_lines_are_one_entry(self):
        found = self.detect(self.three_lines())
        self.assertEqual(len(found), 1, [f["block"] for f in found])
        bx, by, bw, bh = found[0]["block"]
        self.assertGreaterEqual(by + bh, 262 + 2 * PITCH)

    def test_a_stray_outside_the_balloon_stays_out(self):
        found = self.detect(self.three_lines(ships=True))
        inside = [f for f in found if f["block"][0] < 330 < sum(
            f["block"][::2])]
        self.assertEqual(len(inside), 1, [f["block"] for f in found])
        bx, by, bw, bh = inside[0]["block"]
        self.assertLessEqual(bx + bw, 330 + AX)       # not past the outline
        x, y, w, h = inside[0]["bbox"]
        self.assertLessEqual(x + w, 330 + AX + 8)     # nor is its mask

    def test_two_stacked_balloons_stay_two(self):
        img = np.full((900, 1000, 3), SKY, np.uint8)
        balloon(img, 330, 220, 200, 70, ["Hallare el puente",
                                         "y detendre la"], 212)
        balloon(img, 330, 350, 200, 50, ["Nave lista ya"], 362)
        found = self.detect(img)
        self.assertEqual(len(found), 2, [f["block"] for f in found])


if __name__ == "__main__":
    unittest.main()
