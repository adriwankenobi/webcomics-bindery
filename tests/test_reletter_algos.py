"""Unit tests for the re-lettering algorithms (relettering/reletter_fit.py).

These need cv2, which lives ONLY in .venv-reletter:
    .venv-reletter/bin/python -m unittest discover tests
Under the base .venv (what `just test` uses) they skip themselves.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

try:
    import cv2  # noqa: F401
    import numpy as np
    HAVE_CV2 = True
except ImportError:  # base .venv — the pipeline glue tests still run
    HAVE_CV2 = False


def load_fit():
    """Import reletter_fit as a LIBRARY: configure() points it at a comic, so a
    test never has to fake sys.argv to get at the algorithms."""
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_fit
    reletter_fit.configure("unit-test-comic")
    return reletter_fit


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CleanBubbleKeepsOutline(unittest.TestCase):
    """A leaked mask must never let the repaint erase the balloon outline.

    Detection sometimes hands clean_bubble a mask that spills past the
    balloon (a tail that exits the crop window, or a balloon on a light
    background). Everything dark inside the mask used to be treated as old
    lettering, so the outline itself was painted out and the balloon
    vanished from the page.
    """

    def setUp(self):
        self.fit = load_fit()

    def _page(self):
        # white page, a black balloon outline ellipse, dark letters inside
        img = np.full((300, 400, 3), 255, np.uint8)
        cv2.ellipse(img, (200, 150), (150, 100), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(110, 290, 22):      # a row of letter-sized blobs
            cv2.rectangle(img, (cx, 140), (cx + 14, 162), (10, 10, 10), -1)
        return img

    def test_leaked_mask_keeps_outline_and_still_wipes_letters(self):
        img = self._page()
        before = img.copy()
        b = {"kind": "bubble", "bbox": [0, 0, 400, 300],
             "block": [110, 140, 180, 22]}
        mask = np.full((300, 400), 255, np.uint8)   # the leak: whole bbox
        self.fit.clean_bubble(img, b, mask)

        outline = (before.min(axis=2) < 100) & (np.arange(300)[:, None] < 60)
        kept = (img.min(axis=2) < 100) & outline
        self.assertGreater(
            kept.sum(), 0.7 * outline.sum(),
            "balloon outline was erased by the repaint (leaked mask)")

        letters = np.zeros((300, 400), bool)
        letters[140:162, 110:290] = True
        left = (img.min(axis=2) < 100) & letters
        self.assertLess(left.sum(), 0.1 * letters.sum(),
                        "old lettering survived the clean")


    def test_leaked_mask_does_not_tint_the_repaint(self):
        """A mask that spills onto a light background must not colour the
        patch: the row median has to come from the balloon fill alone, or
        the wiped letters come back as coloured ghost blocks."""
        # the balloon is SMALL inside a wide bbox, so on every row the
        # leaked sky outnumbers the fill — that is what tints the median
        img = np.full((300, 700, 3), 255, np.uint8)
        img[:, :] = (198, 218, 245)                  # pale sky everywhere
        cv2.ellipse(img, (350, 150), (110, 70), 0, 0, 360,
                    (255, 255, 255), -1)             # white balloon fill
        cv2.ellipse(img, (350, 150), (110, 70), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(270, 430, 22):
            cv2.rectangle(img, (cx, 140), (cx + 14, 162), (10, 10, 10), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 700, 300],
             "block": [270, 140, 160, 22]}
        mask = np.full((300, 700), 255, np.uint8)    # leaks over the sky
        self.fit.clean_bubble(img, b, mask)

        patch = img[140:162, 270:430].reshape(-1, 3).astype(int)
        self.assertLess(
            abs(int(patch[:, 2].mean()) - int(patch[:, 0].mean())), 12,
            "letters were repainted with the background tint, not the fill")
        self.assertGreater(patch.min(axis=1).mean(), 200,
                           "repaint is far darker than the balloon fill")

    def test_truncated_mask_still_wipes_the_lettering(self):
        """A joined-balloon split can truncate the mask part-way down the
        text. Letters straddling that edge are not ENCLOSED by the fill, so
        an enclosure-only rule left them on the page — the new text was
        then drawn on top of them, and a single stranded letter read as a
        "wild" letter. 341 letters shipped this way."""
        img = np.full((240, 400, 3), 255, np.uint8)
        cv2.ellipse(img, (200, 120), (185, 110), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (200, 120), (185, 110), 0, 0, 360, (0, 0, 0), 3)
        for cy in (60, 100, 140):
            for cx in range(60, 340, 24):
                cv2.rectangle(img, (cx, cy), (cx + 18, cy + 24), (10, 10, 10), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 400, 240],
             "block": [60, 60, 298, 104]}
        # the mask stops at y=150, part-way THROUGH the third line
        mask = np.zeros((240, 400), np.uint8)
        cv2.ellipse(mask, (200, 120), (185, 110), 0, 0, 360, 255, -1)
        mask[150:, :] = 0
        self.fit.clean_bubble(img, b, mask)

        straddling = np.zeros((240, 400), bool)
        straddling[140:164, 60:340] = True
        inside = np.zeros((240, 400), bool)
        inside[60:84, 60:340] = True
        self.assertLess((img.min(axis=2) < 100)[inside].mean(), 0.1,
                        "lettering inside the mask survived")
        # the straddling line is partly outside the mask; the part the mask
        # does cover must still be wiped
        covered = straddling.copy(); covered[150:, :] = False
        self.assertLess((img.min(axis=2) < 100)[covered].mean(), 0.15,
                        "lettering straddling the mask edge survived")

    def test_wide_line_does_not_split_the_fill(self):
        """A line of lettering that spans the balloon cuts the light fill in
        two. Keeping only the largest fragment left the other half's
        lettering on the page — the new text then landed on top of it, and a
        single stranded letter read as a "wild" letter beside the balloon."""
        img = np.full((260, 420, 3), 255, np.uint8)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (0, 0, 0), 3)
        # a full-width line: its letters + gaps sever the fill above/below
        for cx in range(25, 400, 20):
            cv2.rectangle(img, (cx, 118), (cx + 17, 142), (10, 10, 10), -1)
        # and one short line well below it
        for cx in range(150, 260, 20):
            cv2.rectangle(img, (cx, 180), (cx + 17, 202), (10, 10, 10), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 420, 260],
             "block": [25, 118, 392, 84]}
        mask = np.zeros((260, 420), np.uint8)
        cv2.ellipse(mask, (210, 130), (195, 120), 0, 0, 360, 255, -1)
        self.fit.clean_bubble(img, b, mask)

        lower = np.zeros((260, 420), bool)
        lower[180:202, 150:260] = True
        self.assertLess(
            (img.min(axis=2) < 100)[lower].mean(), 0.1,
            "lettering below the full-width line survived the clean")

    def test_touching_letters_are_still_wiped(self):
        """Neighbouring letters merge into one wide blob at the ink
        threshold. A size cap on what counts as lettering left whole words
        standing, so the wipe keys on shape (an island inside the fill),
        not on size."""
        img = self._page()
        cv2.rectangle(img, (110, 140), (290, 162), (10, 10, 10), -1)  # a word
        b = {"kind": "bubble", "bbox": [0, 0, 400, 300],
             "block": [110, 140, 180, 22]}
        mask = np.full((300, 400), 255, np.uint8)
        self.fit.clean_bubble(img, b, mask)

        word = np.zeros((300, 400), bool)
        word[140:162, 110:290] = True
        self.assertLess((img.min(axis=2) < 100)[word].mean(), 0.1,
                        "a merged word survived the clean")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LongWordHyphenation(unittest.TestCase):
    """One over-long word must not shrink the whole bubble.

    Words were unbreakable, so the size search kept stepping down until the
    longest word fitted on one line — a single shout like the Spanish
    "ENGRANAJE" dropped its balloon from the book size to 15px while every
    balloon around it stayed at 23px."""

    def setUp(self):
        self.fit = load_fit()
        self.f = self.fit.Fitter()

    def rows(self, x0, x1, y0, y1):
        return {y: (x0, x1) for y in range(y0, y1)}

    def test_long_word_is_hyphenated_instead_of_shrinking(self):
        rows = self.rows(0, 120, 0, 200)          # a narrow balloon
        words = self.fit.parse_runs("¡ENGRANAJE!")
        lines = self.fit.wrap_at(self.f, words, rows, 23, 100, set())
        self.assertIsNotNone(lines, "no layout at book size — bubble shrank")
        text = " ".join(w for _, _, _, runs in lines for _, w in runs)
        self.assertIn("-", text, f"word was not hyphenated: {text!r}")
        self.assertEqual(text.replace("- ", "").replace("-", ""),
                         "¡ENGRANAJE!", f"hyphenation altered the text: {text!r}")
        self.assertGreater(len(lines), 1, "hyphenation did not add a line")

    def test_a_short_word_is_not_hyphenated_to_gain_a_pixel(self):
        """Hyphenation exists so one LONG word cannot collapse a bubble.
        Breaking a short one to buy a size or two reads as a typo: p88's
        two-word balloon came back as "SI / PUE- / DO..." at 18px, which is
        worse than the same words whole at 16px."""
        self.assertIsNone(
            self.fit.split_word(self.f, "regular", "COSAS...", 18, 55),
            "a five-letter word was hyphenated")

    def test_a_long_word_is_still_hyphenated(self):
        self.assertIsNotNone(
            self.fit.split_word(self.f, "regular", "ENGRANAJE", 23, 60),
            "a long word must still break rather than shrink the bubble")

    def test_short_words_are_left_alone(self):
        rows = self.rows(0, 400, 0, 200)
        words = self.fit.parse_runs("HOLA QUE TAL")
        lines = self.fit.wrap_at(self.f, words, rows, 23, 100, set())
        self.assertIsNotNone(lines)
        text = " ".join(w for _, _, _, runs in lines for _, w in runs)
        self.assertNotIn("-", text, "hyphenated a word that already fitted")

    def test_break_keeps_at_least_two_letters_each_side(self):
        for word in ("PERPENDICULARES", "CATEGORIZÁNDOSE", "¡PARALELISMO"):
            head, tail = self.fit.split_word(self.f, "regular", word, 23, 90)
            self.assertGreaterEqual(len(head.rstrip("-")), 2, word)
            self.assertGreaterEqual(len(tail), 2, word)
            self.assertTrue(head.endswith("-"), word)
            self.assertEqual(head.rstrip("-") + tail, word, word)

    def test_hopeless_size_fails_fast(self):
        """At the top of the size search every word is far too wide. Rather
        than shatter each into fragments (and run the line-break DP over
        them), hyphenate must hand back the original words so the wrap
        fails immediately and the search steps down."""
        words = self.fit.parse_runs("PERPENDICULARES CATEGORIZÁNDOSE")
        out, hard, paras = self.fit.hyphenate(
            self.f, words, 80, 40.0, set(), set())
        self.assertEqual(out, words, "words were shattered at a hopeless size")
        self.assertEqual(hard, set())


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class BlockClippedToBalloon(unittest.TestCase):
    """The letterer's text block is the primary centering anchor, so a
    block that swallowed artwork beside the balloon (a stroke of fur, a
    panel line) drags the new text clean out of the lobe."""

    def setUp(self):
        self.fit = load_fit()

    def test_block_is_clipped_to_the_balloon_rows(self):
        # balloon occupies x 135..390 over its text rows
        rows = {y: (135, 390) for y in range(2100, 2250)}
        # the detected block starts far left of the balloon, at x=11
        cx, cy = self.fit.block_anchor((11, 2127, 337, 97), rows)
        self.assertGreaterEqual(cx, 135, "anchor still left of the balloon")
        self.assertLessEqual(cx, 390, "anchor right of the balloon")
        self.assertAlmostEqual(cy, 2127 + 97 / 2.0, places=3,
                               msg="vertical anchor must be left alone")

    def test_clean_block_is_untouched(self):
        rows = {y: (100, 400) for y in range(1000, 1100)}
        cx, cy = self.fit.block_anchor((150, 1010, 200, 80), rows)
        self.assertAlmostEqual(cx, 250.0, places=3)
        self.assertAlmostEqual(cy, 1050.0, places=3)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DisjointLobes(unittest.TestCase):
    """Two separate balloons sometimes land in ONE detection entry. Their
    union has no waist in the row profile (a row span is a single x-range),
    so the text used to be typeset straight across the gap, over both
    outlines."""

    def setUp(self):
        self.fit = load_fit()

    def test_side_by_side_balloons_split_into_two_lobes(self):
        m = np.zeros((200, 600), np.uint8)
        cv2.ellipse(m, (150, 100), (130, 80), 0, 0, 360, 1, -1)
        cv2.ellipse(m, (450, 100), (130, 80), 0, 0, 360, 1, -1)
        lobes = self.fit.mask_lobes(m, 1000, 2000, 2)
        self.assertIsNotNone(lobes, "side-by-side lobes were not separated")
        self.assertEqual(len(lobes), 2)
        left, right = lobes
        lx1 = max(v[1] for v in left.values())
        rx0 = min(v[0] for v in right.values())
        self.assertLessEqual(lx1, rx0 + 2, "lobes overlap — not a clean split")
        # page coordinates, not mask-local
        self.assertGreaterEqual(min(left), 2000)
        self.assertGreaterEqual(min(v[0] for v in left.values()), 1000)

    def test_single_balloon_is_not_split(self):
        m = np.zeros((200, 400), np.uint8)
        cv2.ellipse(m, (200, 100), (170, 80), 0, 0, 360, 1, -1)
        self.assertIsNone(self.fit.mask_lobes(m, 0, 0, 2),
                          "a single oval was split into two lobes")


class QaArtifactGate(unittest.TestCase):
    """The artifact gate must catch an erased balloon outline, which is the
    failure that shipped: low area (a thin stroke) but spanning the whole
    balloon. It must still ignore ordinary wiped lettering."""

    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qa_scan_mod", REPO / "relettering" / "qa_scan.py")
        # qa_scan runs a whole scan at import, so lift just the predicate
        src = (REPO / "relettering" / "qa_scan.py").read_text()
        start = src.index("def is_artifact")
        end = src.index("bad = 0")
        ns = {}
        exec(compile(src[start:end], "qa_scan", "exec"), ns)
        self.is_artifact = ns["is_artifact"]

    def test_erased_balloon_outline_is_caught(self):
        # a 417x286 oval outline, ~3px stroke: the page-5 failure
        self.assertTrue(self.is_artifact(385, 185, 2366))
        self.assertTrue(self.is_artifact(405, 157, 2489))   # page 104
        self.assertTrue(self.is_artifact(95, 134, 642))     # page 135

    def test_wiped_lettering_is_ignored(self):
        self.assertFalse(self.is_artifact(300, 26, 2400))   # a line of text
        self.assertFalse(self.is_artifact(28, 30, 500))     # one letter
        self.assertFalse(self.is_artifact(180, 22, 1500))   # a merged word

    def test_solid_repainted_art_still_caught(self):
        self.assertTrue(self.is_artifact(200, 120, 12000))


class RowProfileClipping(unittest.TestCase):
    """Detection can record a row span reaching outside the bubble's box.
    The mask that gets cleaned is bbox-sized, so text placed out there sits
    on artwork that was never repainted — it reads as text outside the
    balloon. main() clips the profile; this pins the arithmetic."""

    def test_spans_are_clipped_to_the_box(self):
        x, y, w = 135, 2135, 253
        raw = [[0, -90, 133], [1, -112, 122], [2, 10, 400], [3, 300, 400]]
        rows = {}
        for r in raw:
            rx0, rx1 = max(x + r[1], x), min(x + r[2], x + w)
            if rx1 > rx0:
                rows[y + r[0]] = (rx0, rx1)
        self.assertEqual(rows[2135], (135, 268))      # left overhang cut
        self.assertEqual(rows[2136], (135, 257))
        self.assertEqual(rows[2137], (145, 388))      # right overhang cut
        self.assertNotIn(2138, rows)                  # entirely outside
        for x0, x1 in rows.values():
            self.assertGreaterEqual(x0, x)
            self.assertLessEqual(x1, x + w)


if __name__ == "__main__":
    unittest.main()

@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class EmphasisGluedToPunctuation(unittest.TestCase):
    """*emphasis* must be honoured when the marker is glued to punctuation.

    parse_runs scanned "\\*([^*]+)\\*|(\\S+)" — the alternation is tried left
    to right at each position, so for "¡--*VENTANA*" the emphasis branch fails
    (no leading *) and the greedy \\S+ branch swallows the whole token,
    asterisks and all. The book printed a literal "¡--*VENTANA*".

    Runs are word level and the renderer puts a space between them, so a
    word cannot carry two styles: the emphasised token keeps its attached
    punctuation and takes the emphasis style."""

    def setUp(self):
        self.fit = load_fit()

    def test_marker_after_punctuation_is_parsed(self):
        runs = self.fit.parse_runs("¡--*VENTANA* A LA MONTAÑA!")
        self.assertEqual(runs[0], ("bolditalic", "¡--VENTANA"))

    def test_no_asterisk_ever_reaches_the_page(self):
        for text in ('¡--*VENTANA* A LA MONTAÑA!',
                     '"*UNO* CUENTA PARA LOS DEMAS--"',
                     '--*ANDAR* ES LO QUE RESULTA MATEMÁTICO.',
                     '¡...*OTROS* LLEGARÁN ESTE AÑO'):
            for _, word in self.fit.parse_runs(text):
                self.assertNotIn("*", word, f"literal asterisk from {text!r}")

    def test_plain_emphasis_still_splits_into_words(self):
        runs = self.fit.parse_runs("A *B C* D")
        self.assertEqual(runs, [("regular", "A"), ("bolditalic", "B"),
                                ("bolditalic", "C"), ("regular", "D")])

    def test_unemphasised_text_is_untouched(self):
        self.assertEqual(self.fit.parse_runs("HOLA QUE TAL"),
                         [("regular", "HOLA"), ("regular", "QUE"),
                          ("regular", "TAL")])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class OverflowInpaintStaysOnItsBalloon(unittest.TestCase):
    """The overflow inpaint must never reach a NEIGHBOURING balloon.

    clean_bubble's overflow step inpaints letter-sized blobs inside the
    letterer's text block that the mask did not cover — that is how a line
    the letterer spilled past the outline gets cleaned. But it used the RAW
    b["block"], and a blob cluster happily swallows the balloon next door:
    on one page the block spanned both balloons, so all 38 letter blobs of
    the neighbour were inpainted away. Nothing was typeset back (the
    neighbour is a separate bubble, or no bubble at all), so the page
    shipped with an empty balloon."""

    def setUp(self):
        self.fit = load_fit()

    def _two_balloons(self):
        img = np.full((300, 700, 3), 255, np.uint8)
        img[:, :] = (150, 120, 90)                       # dark art behind
        for cx in (150, 520):
            cv2.ellipse(img, (cx, 150), (110, 70), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, (cx, 150), (110, 70), 0, 0, 360, (0, 0, 0), 3)
            for lx in range(cx - 70, cx + 60, 22):
                cv2.rectangle(img, (lx, 138), (lx + 14, 162), (10, 10, 10), -1)
        return img

    def _right_bubble(self):
        mask = np.zeros((140, 220), np.uint8)
        cv2.ellipse(mask, (110, 70), (110, 70), 0, 0, 360, 255, -1)
        # the block was polluted by the cluster and spans BOTH balloons
        return {"kind": "bubble", "bbox": [410, 80, 220, 140],
                "block": [70, 130, 540, 40]}, mask

    def _dark(self, img, x0, x1):
        return (img[138:162, x0:x1].min(axis=2) < 100).mean()

    def test_neighbouring_balloon_keeps_its_lettering(self):
        img = self._two_balloons()
        b, mask = self._right_bubble()
        self.fit.clean_bubble(img, b, mask)
        self.assertGreater(self._dark(img, 80, 220), 0.25,
                           "the neighbouring balloon's lettering was erased")

    def test_its_own_lettering_is_still_wiped(self):
        img = self._two_balloons()
        b, mask = self._right_bubble()
        self.fit.clean_bubble(img, b, mask)
        self.assertLess(self._dark(img, 450, 590), 0.05,
                        "the bubble's own lettering survived the clean")

    def test_a_line_the_mask_missed_inside_the_bbox_is_still_cleaned(self):
        """What the step actually earns its keep on: lettering inside the
        bubble's own bbox that the mask never reached (a split fill leaves a
        line uncovered). Over this book that is all 1704 of the blobs it
        legitimately cleans."""
        img = self._two_balloons()
        for lx in range(460, 580, 22):                   # a second line
            cv2.rectangle(img, (lx, 176), (lx + 14, 198), (10, 10, 10), -1)
        b, mask = self._right_bubble()
        mask[96:, :] = 0                                 # mask stops early
        b["block"] = [450, 130, 150, 70]
        self.fit.clean_bubble(img, b, mask)
        missed = (img[176:198, 460:580].min(axis=2) < 100).mean()
        self.assertLess(missed, 0.05,
                        "lettering the mask missed was not cleaned")

    def _adjacent_neighbour(self):
        """The neighbour balloon touches this one, so no margin around the
        bubble can separate them (p234: the bubble's bbox ends at x=879 and
        the neighbour's lettering starts at x=899). What tells them apart is
        that the neighbour's letters are WALLED IN by its own outline, while
        a line the letterer spilled sits on open page."""
        img = np.full((300, 700, 3), 255, np.uint8)
        img[:, :] = (150, 120, 90)
        cv2.ellipse(img, (240, 150), (150, 90), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (240, 150), (150, 90), 0, 0, 360, (0, 0, 0), 3)
        cv2.ellipse(img, (450, 190), (100, 60), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (450, 190), (100, 60), 0, 0, 360, (0, 0, 0), 3)
        for lx in range(130, 340, 22):                 # this bubble's text
            cv2.rectangle(img, (lx, 138), (lx + 14, 162), (10, 10, 10), -1)
        for lx in range(400, 440, 22):                 # the NEIGHBOUR's text
            cv2.rectangle(img, (lx, 178), (lx + 14, 202), (10, 10, 10), -1)
        mask = np.zeros((180, 300), np.uint8)
        cv2.ellipse(mask, (150, 90), (150, 90), 0, 0, 360, 255, -1)
        b = {"kind": "bubble", "bbox": [90, 60, 300, 180],
             "block": [130, 138, 430, 70]}             # block reaches over
        return img, b, mask

    def test_adjacent_neighbour_balloon_keeps_its_lettering(self):
        img, b, mask = self._adjacent_neighbour()
        self.fit.clean_bubble(img, b, mask)
        left = (img[178:202, 400:454].min(axis=2) < 100).mean()
        self.assertGreater(left, 0.25,
                           "adjacent balloon's lettering was erased")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class FaintStrokesAreWipedToo(unittest.TestCase):
    """The old lettering's PALE strokes must be wiped with the rest of it.

    The ink cut-off adapts to the balloon fill (ref - 60, so ~150 on a white
    balloon) and the repaint reaches 4px around whatever crossed it. The
    upscaler leaves a letter's thin strokes and antialiased rims at 150-220,
    which is neither ink nor within the halo of any — measured up to 51px
    from the nearest wiped pixel. They survived as faint letter-shaped
    specks in 1428 bubbles on 263 pages, visible at 1:1 300dpi; detection's
    letter_mask cannot see them, so the leftover scan called it healthy.

    Inside the letterer's text block a softer cut-off is safe (it is a text
    area), which is what bounds the change."""

    def setUp(self):
        self.fit = load_fit()

    def _balloon(self):
        img = np.full((260, 420, 3), 250, np.uint8)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (250, 250, 250), -1)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (0, 0, 0), 3)
        return img

    def test_pale_strokes_leave_no_ghost(self):
        img = self._balloon()
        for cx in range(60, 350, 24):                 # solid letter bodies
            cv2.rectangle(img, (cx, 100), (cx + 16, 124), (10, 10, 10), -1)
        for cx in range(66, 350, 24):                 # their pale hairlines
            cv2.rectangle(img, (cx, 126), (cx + 4, 140), (170, 170, 170), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 420, 260],
             "block": [60, 100, 306, 44]}
        mask = np.zeros((260, 420), np.uint8)
        cv2.ellipse(mask, (210, 130), (195, 120), 0, 0, 360, 255, -1)
        self.fit.clean_bubble(img, b, mask)

        band = img[126:140, 66:350].min(axis=2)
        self.assertGreater(band.min(), 225,
                           "pale strokes survived as a faint ghost")

    def test_art_outside_the_balloon_is_never_repainted(self):
        """The softer cut-off must stay inside the balloon's own FILL.

        Bounding it to the text block is not enough: the block swallows
        artwork beside the balloon and the mask leaks past the outline. Art
        too light for the strict ink cut-off but within the soft one was
        then caught as lettering and the row median painted it with the fill
        colour — white rectangles across the page (17 artifact blobs on 9
        pages, where the shipped book had 2)."""
        # art at min-channel 190: BELOW the soft cut-off (ref-30 = 220) but
        # above the strict one (150), so only the soft term can reach it
        img = np.full((300, 700, 3), (215, 205, 190), np.uint8)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(150, 340, 24):
            cv2.rectangle(img, (cx, 138), (cx + 16, 162), (10, 10, 10), -1)
        before = img.copy()
        # the mask leaks far past the balloon and the block reaches the art
        b = {"kind": "bubble", "bbox": [0, 0, 700, 300],
             "block": [150, 138, 350, 40]}
        mask = np.full((300, 700), 255, np.uint8)
        self.fit.clean_bubble(img, b, mask)
        art = img[138:178, 410:500].astype(int)
        self.assertLess(np.abs(art - before[138:178, 410:500]).max(), 40,
                        "artwork outside the balloon was repainted")

    def test_shading_outside_the_text_block_is_kept(self):
        """The softer cut-off must not flatten a grey-shaded balloon."""
        img = self._balloon()
        img[190:230, 60:350] = 205                    # a shaded band, no text
        for cx in range(60, 350, 24):
            cv2.rectangle(img, (cx, 100), (cx + 16, 124), (10, 10, 10), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 420, 260],
             "block": [60, 100, 306, 26]}
        mask = np.zeros((260, 420), np.uint8)
        cv2.ellipse(mask, (210, 130), (195, 120), 0, 0, 360, 255, -1)
        self.fit.clean_bubble(img, b, mask)

        shade = img[195:225, 80:330].min(axis=2).mean()
        self.assertLess(shade, 225, "the balloon's shading was flattened")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class SplitCaptionBoxIsCleanedWhole(unittest.TestCase):
    """A coloured caption box must be cleaned as ONE box.

    detection's letter_mask is a "dark text on light paper" ring test. On a
    yellow->white gradient the bright half fails it, so the box comes back as
    TWO stacked entries — a `tint` strip and a `bubble` strip, split exactly
    where the fill crosses LIGHT_MIN — each with a RAGGED mask that follows
    the gradient rather than the box. Letters straddle the mask edge, the
    band at the junction is never cleaned, and the new line is typeset on top
    of the old one: 28 boxes on 19 pages, p147 and p170 unreadable.

    The two entries' bboxes together span every line of the box, so their
    union is all the cleaner needs to wipe it as one."""

    def setUp(self):
        self.fit = load_fit()

    def _box(self):
        img = np.full((260, 700, 3), (90, 70, 60), np.uint8)
        x0, y0, x1, y1 = 50, 50, 650, 180
        for yy in range(y0, y1):                      # yellow -> white
            t = (yy - y0) / float(y1 - y0)
            img[yy, x0:x1] = (255, int(230 + 25 * t), int(90 + 165 * t))
        cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (20, 20, 20), 5)
        for cx in range(80, 600, 26):                 # line 1 (bright half)
            cv2.rectangle(img, (cx, 70), (cx + 18, 96), (15, 15, 15), -1)
        for cx in range(80, 600, 26):                 # line 2 (pale half)
            cv2.rectangle(img, (cx, 120), (cx + 18, 146), (15, 15, 15), -1)
        return img, (x0, y0, x1, y1)

    def test_every_line_of_the_box_is_wiped(self):
        img, box = self._box()
        self.fit.clean_caption_box(img, box)
        for lo, hi in ((70, 96), (120, 146)):
            left = (img[lo:hi, 80:600].min(axis=2) < 60).mean()
            self.assertLess(left, 0.02,
                            f"lettering at y {lo}..{hi} survived")

    def test_the_drawn_frame_survives(self):
        img, box = self._box()
        before = img.copy()
        self.fit.clean_caption_box(img, box)
        frame = (before.min(axis=2) < 60)
        frame[60:170, 60:640] = False                 # only the border ring
        kept = (img.min(axis=2) < 60) & frame
        self.assertGreater(kept.sum(), 0.9 * frame.sum(),
                           "the caption box's drawn frame was erased")

    def test_a_dense_row_is_not_blown_to_white(self):
        """On a row the lettering nearly fills, there are no fill pixels left
        to take a median from. Falling back to white painted a bright band
        right across the box at the strip junction."""
        img, box = self._box()
        cv2.rectangle(img, (50, 100), (650, 118), (15, 15, 15), -1)
        self.fit.clean_caption_box(img, box)
        dense = img[100:118, 60:640].min(axis=2).mean()
        above = img[85:95, 60:640].min(axis=2).mean()
        below = img[150:160, 60:640].min(axis=2).mean()
        self.assertLess(dense, max(above, below) + 18,
                        "a dense row was painted brighter than the fill")

    def test_a_fill_that_varies_ACROSS_the_row_survives(self):
        """These boxes are not uniform along a row either — a highlight, or
        yellow edges around a pale centre. A single per-row fill level then
        reads the bright part as the fill and everything yellow as ink, and
        the whole row is repainted WHITE: a band straight across the box
        (p101, p118, p124 in the artifact gate)."""
        img = np.full((260, 700, 3), (90, 70, 60), np.uint8)
        x0, y0, x1, y1 = 50, 50, 650, 180
        img[y0:y1, x0:x1] = (255, 230, 90)              # yellow box
        img[y0:y1, 300:430] = (255, 253, 245)           # a pale highlight
        cv2.rectangle(img, (x0 - 6, y0 - 6), (x1 + 6, y1 + 6), (20, 20, 20), 5)
        for cx in range(80, 600, 26):
            cv2.rectangle(img, (cx, 90, ), (cx + 18, 116), (15, 15, 15), -1)
        self.fit.clean_caption_box(img, (x0, y0, x1, y1))
        yellow = img[130:170, 80:280, 2].mean()         # still yellow?
        pale = img[130:170, 320:410, 2].mean()
        self.assertLess(yellow, 160,
                        "the yellow fill was repainted white")
        self.assertGreater(pale, 200, "the highlight was lost")

    def test_the_gradient_is_not_flattened(self):
        """A flat fill reads as a coloured slab with straight edges."""
        img, box = self._box()
        self.fit.clean_caption_box(img, box)
        top = int(img[60:68, 300:340, 2].mean())
        bot = int(img[160:172, 300:340, 2].mean())
        self.assertGreater(bot - top, 60,
                           "the box's vertical gradient was flattened")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CaptionBoxDiscovery(unittest.TestCase):
    """Which entries form a coloured caption box that must be cleaned whole.

    The signature of a box split by the gradient is precise: a `tint` strip
    with a `bubble` strip directly above or below it, sharing its x range.
    Keep it tight — a rectangularity or drawn-frame test fires on ordinary
    oval balloons and on leaked masks (p5 b08), which must never be wiped
    as boxes."""

    def setUp(self):
        self.fit = load_fit()

    @staticmethod
    def _b(kind, x, y, w, h):
        return {"kind": kind, "bbox": [x, y, w, h],
                "block": [x + 4, y + 4, w - 8, h - 8]}

    def test_tint_strip_and_the_bubble_below_it_make_one_box(self):
        bubs = [self._b("tint", 229, 117, 631, 34),
                self._b("bubble", 229, 149, 634, 60)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2}),
                         [((229, 117, 863, 209), {1, 2})])

    def test_strips_that_overlap_are_still_one_box(self):
        """Detection does not always split the box cleanly at the seam — the
        two strips can OVERLAP. p219's bubble strip starts 26px above the
        tint strip's bottom, fell outside the tolerance, and shipped with the
        old line still under the new."""
        bubs = [self._b("tint", 694, 1202, 314, 99),
                self._b("bubble", 696, 1275, 314, 126)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2}),
                         [((694, 1202, 1010, 1401), {1, 2})])

    def test_a_lone_tint_strip_is_a_box_on_its_own(self):
        bubs = [self._b("tint", 316, 158, 272, 35)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1}),
                         [((316, 158, 588, 193), {1})])

    def test_a_balloon_elsewhere_is_not_joined_to_the_strip(self):
        bubs = [self._b("tint", 229, 117, 631, 34),
                self._b("bubble", 60, 600, 200, 140)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2}),
                         [((229, 117, 860, 151), {1})])

    def test_ordinary_balloons_make_no_boxes(self):
        bubs = [self._b("bubble", 60, 60, 300, 180),
                self._b("bubble", 500, 400, 260, 150)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2}), [])

    def test_an_entry_with_no_layout_is_ignored(self):
        """A bubble the fit could not place keeps its original lettering, so
        its box must not be wiped."""
        bubs = [self._b("tint", 229, 117, 631, 34),
                self._b("bubble", 229, 149, 634, 60)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1}),
                         [((229, 117, 860, 151), {1})])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class InteriorStopsAtAColourChange(unittest.TestCase):
    """The balloon's fill must not flood out onto a pale background.

    bubble_interior walls the flood on DARK strokes only. Where a balloon
    sits on pale sky and its outline has a gap — a tail leaving the crop, a
    stroke the mask clipped — the fill escapes and the "interior" becomes
    balloon + page. Round 3 clipped the fit's row profile to that interior
    to undo leaked masks, which does nothing when the interior leaks too:
    p65 b07 kept 283 of 311 rows, so the fit laid out against a 250-row
    profile for a 165-row oval and set two lines above the balloon's top
    arc, on bare art.

    The page is a different COLOUR from the fill even where nothing dark
    separates them (there, white 252,252,251 against sky 254,241,163), and
    that is a wall the flood can respect."""

    def setUp(self):
        self.fit = load_fit()

    def _leaky(self):
        img = np.full((300, 420, 3), (254, 241, 163), np.uint8)   # pale sky
        cv2.ellipse(img, (210, 150), (150, 90), 0, 0, 360, (252, 252, 251), -1)
        cv2.ellipse(img, (210, 150), (150, 90), 0, 30, 330, (0, 0, 0), 3)
        for cx in range(120, 300, 24):
            cv2.rectangle(img, (cx, 140), (cx + 16, 162), (10, 10, 10), -1)
        mask = np.full((300, 420), 1, np.uint8)       # the leak: whole bbox
        minch = img.min(axis=2)
        stroke = (minch <= self.fit.DARK_MAX - 20).astype(np.uint8)
        return img, stroke, mask

    def test_interior_keeps_to_the_balloon(self):
        img, stroke, mask = self._leaky()
        keep = cv2.erode(mask, np.ones((3, 3), np.uint8))
        inter = self.fit.bubble_interior(stroke, keep, (120, 140, 300, 162),
                                         img=img)
        ellipse = np.pi * 150 * 90
        self.assertLess(inter.sum(), 1.5 * ellipse,
                        "the fill flooded out onto the page")
        self.assertGreater(inter.sum(), 0.5 * ellipse,
                           "the balloon's own fill was lost")

    def test_grey_lettering_rims_stay_inside_the_fill(self):
        """The wall is a colour SHIFT, not a brightness one.

        Comparing absolute channel difference carves the old lettering's own
        grey rims (same hue, just darker) out of the interior. Everything
        gated on the fill then skips them, and the rims survive as hollow
        ghost letters (p105 b02's single caption word)."""
        img = np.full((300, 420, 3), 252, np.uint8)
        cv2.ellipse(img, (210, 150), (150, 90), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (210, 150), (150, 90), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(120, 300, 24):               # letter cores...
            cv2.rectangle(img, (cx, 140), (cx + 16, 162), (10, 10, 10), -1)
            cv2.rectangle(img, (cx - 2, 138), (cx + 18, 140), (185, 185, 185), -1)
        mask = np.full((300, 420), 1, np.uint8)
        keep = cv2.erode(mask, np.ones((3, 3), np.uint8))
        stroke = (img.min(axis=2) <= self.fit.DARK_MAX - 20).astype(np.uint8)
        inter = self.fit.bubble_interior(stroke, keep, (120, 138, 300, 162),
                                         img=img)
        rim = np.zeros(inter.shape, bool)
        for cx in range(120, 300, 24):
            rim[138:140, cx - 2:cx + 18] = True
        self.assertGreater(
            (inter & rim).sum(), 0.6 * rim.sum(),
            "the lettering's grey rims were walled out of the fill")

    def test_a_balloon_on_white_page_is_unchanged(self):
        """Same colour inside and out: the dark outline is the only wall,
        and it must still work."""
        img = np.full((300, 420, 3), 255, np.uint8)
        cv2.ellipse(img, (210, 150), (150, 90), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (210, 150), (150, 90), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(120, 300, 24):
            cv2.rectangle(img, (cx, 140), (cx + 16, 162), (10, 10, 10), -1)
        mask = np.full((300, 420), 1, np.uint8)
        keep = cv2.erode(mask, np.ones((3, 3), np.uint8))
        stroke = (img.min(axis=2) <= self.fit.DARK_MAX - 20).astype(np.uint8)
        inter = self.fit.bubble_interior(stroke, keep, (120, 140, 300, 162),
                                         img=img)
        ellipse = np.pi * 150 * 90
        self.assertLess(inter.sum(), 1.4 * ellipse,
                        "the closed outline no longer walls the fill")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class RowSpanFollowsOneRun(unittest.TestCase):
    """A row's available width is ONE connected run, never first-to-last.

    Low in a balloon the fill splits into the body and the mouth of the
    TAIL, with drawn outline between them. Taking the row span as
    (leftmost, rightmost) spans that gap, so the wrap believed it had the
    full width and set the last lines out into the tail, hard against the
    outline — p127's "text outside the bubble". Detection already follows
    the single connected run (follow_lobe); the fit's own profile has to do
    the same."""

    def setUp(self):
        self.fit = load_fit()

    def test_a_narrow_tail_mouth_is_dropped(self):
        row = np.zeros(300, bool)
        row[20:44] = True                 # the tail's mouth
        row[60:280] = True                # the balloon body
        self.assertEqual(self.fit.widest_run(row), (60, 280))

    def test_gaps_between_words_do_not_shrink_the_span(self):
        """A row crossing lettering is broken into the gaps between words.
        Those are all of a size, so the span must still run end to end —
        taking only the widest gave p101's caption a word gap for a profile
        and it could not be typeset at all."""
        row = np.zeros(400, bool)
        for a in range(20, 380, 40):      # fill, broken by letters
            row[a:a + 30] = True
        self.assertEqual(self.fit.widest_run(row), (20, 370))

    def test_a_single_run_is_returned_whole(self):
        row = np.zeros(300, bool)
        row[30:250] = True
        self.assertEqual(self.fit.widest_run(row), (30, 250))

    def test_an_empty_row_has_no_run(self):
        self.assertIsNone(self.fit.widest_run(np.zeros(300, bool)))

    def test_a_run_reaching_the_edge_is_measured(self):
        row = np.zeros(300, bool)
        row[0:120] = True
        self.assertEqual(self.fit.widest_run(row), (0, 120))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LetterWeldedToOutlineDoesNotShrinkTheBalloon(unittest.TestCase):
    """The original lettering must not cost the balloon its own interior.

    Detection walls its flood with the drawn ink CLOSED by a 7x7 kernel, so
    that a gap in the outline cannot leak. Wherever a letter sits within that
    distance of the outline the close WELDS the two: the letter stops being
    an ENCLOSED hole, the hole-fill leaves a bite out of the mask, and a line
    of text that nearly spans the balloon seals whole regions off the
    interior altogether.

    The mask then under-reports the balloon in BOTH directions that matter:
    the fit sees a fraction of the width the balloon really offers (p65 b08:
    127px of a 252px balloon) and drops several size steps, setting short
    narrow lines; and the cleaning, which is confined to the mask, never
    reaches the letters inside the bite, so they survive as pale remnants —
    the "little almost white small objects" the user reported in the same
    balloons as the small text.

    Re-flooding with the RAW ink as the only wall keeps the 2-3px gap between
    letter and outline, so the interior stays ONE component and the letters
    are enclosed holes again. That flood has no seal against a real gap in
    the outline, so what it recovers is trusted only inside the letterer's
    own text block: past a gap it runs onto the page, and a joined balloon's
    sibling lobe is a deliberate split that must never be undone.
    """

    def setUp(self):
        self.fit = load_fit()

    # --- the page detection saw -------------------------------------
    def _balloon(self, gap=False, sibling=False, ground=(215, 170, 125)):
        """White balloon, black outline, a line of letters whose end letters
        all but touch the outline. `gap` opens the outline onto the page;
        `sibling` adds a second balloon sharing the crop; `ground` sets the
        page colour, which is what the flood's colour wall keys on."""
        img = np.full((260, 460, 3), ground, np.uint8)
        cv2.ellipse(img, (200, 130), (150, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (200, 130), (150, 95), 0, 0, 360, (12, 12, 12), 4)
        if gap:                      # a break in the outline, onto the page
            cv2.ellipse(img, (200, 130), (150, 95), 0, 84, 96, (255, 255, 255), 7)
        if sibling:
            cv2.ellipse(img, (415, 130), (95, 80), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, (415, 130), (95, 80), 0, 0, 360, (12, 12, 12), 4)
        # three lines of letter blobs set the way a letterer sets them: hard
        # against the balloon, each end letter 3px clear of the drawn outline
        # over its WHOLE height — a gap the 7x7 close bridges and the raw ink
        # does not. Measured off the drawn balloon rather than the ellipse's
        # algebra: a 4px stroke on a sloping arc covers twice that in x.
        fill = img.min(axis=2) >= 200
        self.letters = np.zeros(img.shape[:2], bool)
        for cy in (100, 130, 160):
            lo, hi = 0, img.shape[1]
            for yy in range(cy - 11, cy + 12):
                xs = np.flatnonzero(fill[yy])
                xs = xs[(xs > 60) & (xs < 350)]
                lo, hi = max(lo, int(xs[0])), min(hi, int(xs[-1]))
            lo, hi = lo + 3, hi - 3
            for cx in list(range(lo, hi - 16, 24)) + [hi - 16]:
                cv2.rectangle(img, (cx, cy - 11), (cx + 16, cy + 11),
                              (15, 15, 15), -1)
                self.letters[cy - 11:cy + 12, cx:cx + 17] = True
        return img

    def _detection_mask(self, img, block):
        """The mask detection's strict path builds: ink CLOSED with a 7x7
        kernel as the barrier, the free component holding the text block,
        then holes filled."""
        bx0, by0, bx1, by1 = block
        barrier = (img.min(axis=2) <= self.fit.DARK_MAX).astype(np.uint8)
        barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE,
                                   np.ones((7, 7), np.uint8))
        n, lab = cv2.connectedComponents((1 - barrier).astype(np.uint8), 4)
        win = lab[by0:by1, bx0:bx1]
        vals, cnt = np.unique(win[win > 0], return_counts=True)
        comp = (lab == vals[cnt.argmax()])
        return self.fit.fill_holes(comp).astype(np.uint8)

    def _block(self):
        return (55, 89, 346, 172)          # the letterer's text block

    def _width(self, m, row):
        xs = np.flatnonzero(m[row])
        return int(xs[-1] - xs[0] + 1) if len(xs) else 0

    # --- the bite is real -------------------------------------------
    def _interior(self, img=None):
        """The balloon's true interior — the ellipse inside its own stroke."""
        ref = np.zeros((260, 460), np.uint8)
        cv2.ellipse(ref, (200, 130), (147, 92), 0, 0, 360, 1, -1)
        return ref

    def test_the_welded_barrier_really_chews_the_mask(self):
        """Guard on the premise: without this the repair tests prove nothing."""
        img = self._balloon()
        mask = self._detection_mask(img, self._block())
        inter = self._interior(img)
        lost = max(self._width(inter, r) - self._width(mask, r)
                   for r in (100, 130, 160))
        self.assertGreater(lost, 40,
                           "premise broken: the mask was never chewed")

    # --- the repair --------------------------------------------------
    def test_the_balloon_gets_its_width_back(self):
        img = self._balloon()
        block = self._block()
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        inter = self._interior(img)
        for row in (100, 130, 160):
            self.assertGreaterEqual(
                self._width(fixed, row), 0.95 * self._width(inter, row),
                f"row {row} did not get the balloon's width back")

    def test_the_letters_come_back_inside_the_mask(self):
        """The cleaning is confined to the mask, so a letter outside it is
        never wiped — new text then lands on top of the old."""
        img = self._balloon()
        block = self._block()
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        inside = np.zeros(self.letters.shape, bool)
        inside[block[1]:block[3], block[0]:block[2]] = True
        want = self.letters & inside
        missed_before = (want & (mask == 0)).sum()
        missed_after = (want & (fixed == 0)).sum()
        self.assertGreater(missed_before, 0, "premise broken: nothing missed")
        self.assertLess(missed_after, 0.05 * missed_before,
                        "old lettering is still outside the mask")

    def test_a_healthy_mask_is_left_alone(self):
        """Letters well clear of the outline weld to nothing."""
        img = np.full((260, 460, 3), (200, 150, 90), np.uint8)
        cv2.ellipse(img, (200, 130), (150, 95), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (200, 130), (150, 95), 0, 0, 360, (12, 12, 12), 4)
        for cx in range(120, 270, 24):
            cv2.rectangle(img, (cx, 119), (cx + 16, 141), (15, 15, 15), -1)
        block = (120, 119, 286, 141)
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        grew = ((fixed > 0) & (mask == 0)).sum()
        self.assertLess(grew, 0.02 * mask.sum(),
                        "a healthy mask was grown")

    # --- and must not run away ---------------------------------------
    def test_a_sibling_lobe_is_never_swallowed(self):
        """Joined balloons are separate entries with a deliberate split; the
        repair must not hand one of them the other's lobe."""
        img = self._balloon(sibling=True)
        block = self._block()
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        self.assertEqual(int(fixed[:, 340:].sum()), int(mask[:, 340:].sum()),
                         "the repair reached into the sibling balloon")

    def test_a_gap_INSIDE_the_block_is_not_followed_either(self):
        """The real shape of the failure: the break in the outline is behind
        the lettering, so the escape lands inside the block rectangle and the
        block bound alone does not hold it. p104's balloon sits on pale sky
        and its outline is broken where a sword crosses it: the flood ran out
        onto the sky, the cleaner then measured the fill on sky and painted
        blue patches over the artwork AND left the old lettering standing.

        A bite is a POCKET the mask and the drawn ink already enclose. A leak
        is not enclosed by anything — that is the difference to key on."""
        # a PALE ground, as p104's sky is: the flood's colour wall has
        # nothing to hold on to when the page is nearly the fill's own colour
        img = self._balloon(ground=(233, 239, 250))
        block = self._block()
        # break the outline where the text block reaches it, on both sides
        cv2.ellipse(img, (200, 130), (150, 95), 0, 170, 190, (255, 255, 255), 8)
        cv2.ellipse(img, (200, 130), (150, 95), 0, -10, 10, (255, 255, 255), 8)
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        page = np.zeros(mask.shape, bool)          # ground well outside it
        page[:, :30] = True
        page[:, 430:] = True
        self.assertEqual(int((fixed[page] > 0).sum()),
                         int((mask[page] > 0).sum()),
                         "the repair followed the leak onto the page")
        self.assertLess(int((fixed > 0).sum()), 1.25 * int((mask > 0).sum()),
                        "the repair swallowed the page around the balloon")

    def test_a_leak_through_an_outline_gap_is_not_followed(self):
        """Past a break in the outline the raw flood runs onto the page. The
        page is outside the letterer's block, which is what bounds it."""
        img = self._balloon(gap=True)
        block = self._block()
        mask = self._detection_mask(img, block)
        fixed = self.fit.repair_letter_bites(mask, img, (0, 0), block)
        outside = np.ones(mask.shape, bool)
        outside[block[1]:block[3], block[0]:block[2]] = False
        self.assertEqual(int((fixed[outside] > 0).sum()),
                         int((mask[outside] > 0).sum()),
                         "the repair followed the leak past the outline")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class RowProfileKeepsTheBalloonsOwnWidth(unittest.TestCase):
    """The profile the fit wraps into must not carry the mask's bites either.

    `rows` comes from detection as the mask ERODED for clearance, so it
    inherits every bite; the letter-union repair there only reaches rows that
    carry letter pixels, and the 6x6 erosion smears a bite three rows past
    the letter that caused it. Those neighbouring rows are what a line box
    lands on, and `rows_span` takes the NARROWEST row of the band — one
    chewed row vetoes the whole line (p88 b06: 2 rows cost it 3 size steps).

    Inside the letterer's own text block the balloon's own fill is the
    authority: where `rows` sits further in than the profile's own clearance,
    that is damage, not clearance."""

    def setUp(self):
        self.fit = load_fit()

    def test_a_bitten_row_is_given_the_fills_width_back(self):
        fill = {y: (100, 400) for y in range(100, 200)}
        rows = {y: (106, 394) for y in range(100, 200)}
        rows[150] = (180, 394)                    # the bite
        out = self.fit.resolve_rows(rows, fill, (100, 200), pad=6)
        self.assertEqual(out[150][0], 106,
                         "the bitten row kept the bite")
        self.assertEqual(out[149], (106, 394), "a healthy row moved")

    def test_clearance_is_never_traded_away(self):
        """The profile's own inset must survive: it is what keeps the new
        lettering off the drawn outline."""
        fill = {y: (100, 400) for y in range(100, 200)}
        rows = {y: (106, 394) for y in range(100, 200)}
        out = self.fit.resolve_rows(rows, fill, (100, 200), pad=6)
        self.assertEqual(out[150], (106, 394))

    def test_a_row_outside_the_block_is_only_clipped(self):
        """Past the block the fill is not trusted to widen anything."""
        fill = {y: (100, 400) for y in range(100, 200)}
        rows = {y: (106, 394) for y in range(100, 200)}
        rows[105] = (180, 394)
        out = self.fit.resolve_rows(rows, fill, (120, 200), pad=6)
        self.assertEqual(out[105], (180, 394))

    def test_a_leaked_profile_is_still_clipped_to_the_fill(self):
        """The old job — a mask covering balloon AND page gives a RECTANGLE
        of a profile — still has to be done."""
        fill = {y: (150, 350) for y in range(100, 200)}
        rows = {y: (10, 450) for y in range(100, 200)}
        out = self.fit.resolve_rows(rows, fill, (100, 200), pad=6)
        self.assertEqual(out[150], (150, 350))

    def test_rows_the_fill_does_not_reach_are_dropped(self):
        fill = {y: (100, 400) for y in range(100, 150)}
        rows = {y: (106, 394) for y in range(100, 200)}
        out = self.fit.resolve_rows(rows, fill, (100, 200), pad=6)
        self.assertNotIn(170, out)
        self.assertIn(120, out)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CaptionBoxRepaintMatchesTheFillItCovers(unittest.TestCase):
    """What replaces the old lettering must be the fill that was under it.

    The fill under a word can only be estimated, and the estimate was a wide
    horizontal grayscale CLOSE — which takes the BRIGHTEST value in its
    window. The window has to be as wide as a whole word (a narrower one
    leaves the word's interior dark and the estimate tracks the ink), so on a
    box whose fill grades ALONG the row it reads the bright end of that
    gradient as the fill everywhere and repaints the whole line as a flat,
    too-bright slab with straight edges — "yellow square very bad,
    rectangle". Interpolate across the lettering from the fill either side of
    it instead, and every gradient the box carries survives the wipe.

    KNOWN, not fixed here: the box rectangle is a detection bbox and can
    overhang the drawn frame by a few px, so a thin tab of fill colour ends up
    outside the box. It happens because the frame is broken into short pieces
    by the rows where a full-width rule cannot register against a per-row fill
    estimate, and the pieces are too short to be spared as long thin curves —
    so the frame is wiped and haloed like lettering. Bounding the repaint
    cannot fix that while the frame is still read as ink; sparing the frame
    reliably is a change to what gets wiped, book-wide."""

    def setUp(self):
        self.fit = load_fit()

    # p42's own proportions: 272x35, one line of type filling most of its
    # height, and a fill that grades BOTH ways. The height is what makes it
    # bite — there is barely any clean fill inside the box to measure from,
    # so the row close reaches along the gradient for its estimate.
    X0, Y0, X1, Y1 = 40, 40, 312, 75
    TOP, BOT = 52, 72                     # the line of type

    def _box(self, letters=True, overhang=0):
        x0, y0, x1, y1 = self.X0, self.Y0, self.X1, self.Y1
        img = np.full((140, 400, 3), (96, 76, 62), np.uint8)
        gx = np.linspace(0.0, 1.0, x1 - x0)[None, :]
        gy = np.linspace(0.0, 1.0, y1 - y0)[:, None]
        g = 0.45 * gx + 0.55 * gy
        img[y0:y1, x0:x1, 0] = 255
        img[y0:y1, x0:x1, 1] = (206 + 49 * g).astype(np.uint8)
        img[y0:y1, x0:x1, 2] = (55 + 200 * g).astype(np.uint8)
        cv2.rectangle(img, (x0 - 5, y0 - 5), (x1 + 4, y1 + 4), (18, 18, 18), 4)
        if letters:
            for cx in range(x0 + 8, x1 - 20, 21):
                cv2.rectangle(img, (cx, self.TOP), (cx + 15, self.BOT),
                              (14, 14, 14), -1)
        box = (x0 - 5 - overhang, y0 - 5 - overhang,
               x1 + 5 + overhang, y1 + 5 + overhang)
        return img, box

    def test_the_repaint_reproduces_the_fill_it_covered(self):
        img, box = self._box()
        ref, _ = self._box(letters=False)
        self.fit.clean_caption_box(img, box)
        band = (slice(self.TOP - 4, self.BOT + 4),
                slice(self.X0 + 4, self.X1 - 4))
        err = np.abs(img[band].astype(np.int16)
                     - ref[band].astype(np.int16)).max(axis=2)
        self.assertLess(int(np.percentile(err, 95)), 26,
                        "the lettering was replaced with the wrong fill")

    def test_it_is_not_flattened_along_the_row(self):
        """The giveaway of the bad estimate: the repainted band is one flat
        colour where the fill it covers grades."""
        img, box = self._box()
        ref, _ = self._box(letters=False)
        self.fit.clean_caption_box(img, box)
        rows = slice(self.TOP + 3, self.BOT - 3)
        want = (float(ref[rows, self.X1 - 60:self.X1 - 10, 2].mean())
                - float(ref[rows, self.X0 + 10:self.X0 + 60, 2].mean()))
        got = (float(img[rows, self.X1 - 60:self.X1 - 10, 2].mean())
               - float(img[rows, self.X0 + 10:self.X0 + 60, 2].mean()))
        self.assertGreater(got, 0.7 * want,
                           "the repainted band lost the row's gradient")

    def test_the_lettering_is_still_gone(self):
        img, box = self._box()
        self.fit.clean_caption_box(img, box)
        band = img[self.TOP:self.BOT, self.X0 + 8:self.X1 - 8]
        self.assertLess((band.min(axis=2) < 60).mean(), 0.02,
                        "lettering survived")



@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CapHeightMatchesWhatIsRendered(unittest.TestCase):
    """`cap_height` must report the height the glyph ACTUALLY renders at.

    It fed two things: `new_cap_h` in layout.json (compared against
    `old_cap_h`, which is MEASURED off the original's pixels) and
    `orig_px = old_cap_h / ratio`, the original letterer's size expressed in
    our font's px. PIL's `getbbox("H")` is neither — it is the layout box
    from the ascender origin, rounded out to whole pixels, so at 23px it
    reported 22 where GIMP renders 19. Measured against the printed PDFs the
    ratio was 0.864 book-wide, and the "is the new text at least as big as
    the original?" gate therefore passed on every one of 56 sampled bubbles
    while 5 of them printed SMALLER than the original. A gate that cannot
    fail is worse than no gate: it is why four rounds of "verified fixed"
    reached a reader who could still see the fault.

    The truth is in the font: the cap glyph's own ink height, scaled by the
    em. Measured at run time from the font file, never hardcoded — the book
    is re-lettered with whatever font the user supplies.
    """

    def setUp(self):
        self.F = load_fit()
        if not (REPO / "relettering" / "fonts" / "regular.ttf").is_file():
            self.skipTest("user-supplied font not installed")
        self.fit = self.F.Fitter()

    def _true_cap(self, size):
        """The cap glyph's ink height, straight from the font's outlines."""
        from fontTools.pens.boundsPen import BoundsPen
        from fontTools.ttLib import TTFont
        t = TTFont(str(REPO / "relettering" / "fonts" / "regular.ttf"))
        upm = t["head"].unitsPerEm
        gs = t.getGlyphSet()
        pen = BoundsPen(gs)
        gs[t.getBestCmap()[ord("H")]].draw(pen)
        return (pen.bounds[3] - pen.bounds[1]) / float(upm) * size

    def test_cap_height_tracks_the_rendered_glyph(self):
        for size in (18, 19, 20, 21, 22, 23):
            got, want = self.fit.cap_height(size), self._true_cap(size)
            self.assertLess(
                abs(got - want), 1.0,
                f"cap_height({size}) = {got}, but the glyph renders "
                f"{want:.1f}px — a gate built on this cannot fail")

    def test_cap_height_is_proportional_to_size(self):
        """No rounding cliffs: 22px must not jump 3px over 21px."""
        caps = [self.fit.cap_height(s) for s in range(16, 26)]
        steps = [b - a for a, b in zip(caps, caps[1:])]
        self.assertLess(max(steps), 1.8, f"cap heights jump: {caps}")

    def test_the_size_that_matches_the_original_is_not_overstated(self):
        """orig_px: the original's measured cap in our font's px."""
        ratio = self.fit.cap_height(100) / 100.0
        # a 16px measured cap needs ~18.7px of our font, not 16.7
        self.assertGreaterEqual(round(16 / ratio), 18)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class GroupedLobesSizeIndependently(unittest.TestCase):
    """Each lobe of a joined balloon fills its OWN balloon.

    Grouped lobes used to share their group's smallest maximum, because the
    original letterer used one size per joined pair. Faithful — and fine
    while the original ran uniformly at ~16px cap across a page. Ours runs
    at the book size, so the pair took the tightest lobe's size and sat at
    18px beside neighbours at 23px: reported as "text small, not occupying
    the whole bubble" on p24, p79 and p101 in three consecutive rounds.
    The user's call (Sep 16 2026) is that each lobe fills its own balloon,
    accepting that a joined pair may now show two sizes.
    """

    def setUp(self):
        self.F = load_fit()

    def test_a_lobe_is_not_held_back_by_its_partner(self):
        caps = self.F.lobe_size_caps({1: 19, 2: 21}, {1: 19, 2: 19}, book=23)
        self.assertEqual(caps, {1: 19, 2: 21})

    def test_each_lobe_is_still_capped_at_the_book_size(self):
        caps = self.F.lobe_size_caps({1: 30, 2: 21}, {1: 16, 2: 16}, book=23)
        self.assertEqual(caps[1], 23)

    def test_a_shout_larger_than_the_book_keeps_its_own_size(self):
        """max(book, the original's own size) — a display shout stays big."""
        caps = self.F.lobe_size_caps({1: 30, 2: 21}, {1: 28, 2: 16}, book=23)
        self.assertEqual(caps[1], 28)

    def test_never_exceeds_a_lobes_own_maximum(self):
        caps = self.F.lobe_size_caps({1: 19, 2: 21}, {1: 25, 2: 25}, book=23)
        self.assertEqual(caps, {1: 19, 2: 21})


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class TrackingBuysASizeInATightBalloon(unittest.TestCase):
    """Slight negative tracking where the balloon is too tight for the size.

    Every bubble the user reported was width-bound within a pixel or two:
    CC WildWords runs about 1.5x wider per unit of cap height than the font
    the original edition used, so at the original's cap height our text
    overflows and the fit drops a size (or two). Comic letterers tighten
    tracking to fill a balloon; GIMP applies it as a text-layer property, so
    the XCF layers stay editable and the glyphs keep their shapes.

    Measured in GIMP: the width reduction is exactly spacing x (chars - 1),
    which is the model used here. Tracking is only ever reached when the
    size does not fit untracked, so roomy balloons are never touched.
    """

    def setUp(self):
        self.F = load_fit()
        if not (REPO / "relettering" / "fonts" / "regular.ttf").is_file():
            self.skipTest("user-supplied font not installed")
        self.fit = self.F.Fitter()

    def test_tracking_narrows_a_line_by_spacing_times_gaps(self):
        runs = [("regular", "PARALELISMO")]
        wide = self.fit.line_width(runs, 21)
        self.fit.track = -1.0
        try:
            tight = self.fit.line_width(runs, 21)
        finally:
            self.fit.track = 0.0
        self.assertAlmostEqual(wide - tight, len("PARALELISMO") - 1, places=6)

    def test_zero_tracking_is_the_untouched_width(self):
        runs = [("regular", "COSAS...")]
        self.fit.track = 0.0
        self.assertAlmostEqual(self.fit.line_width(runs, 20),
                               self.fit.font("regular", 20).getlength("COSAS..."),
                               places=6)

    def test_the_cap_scales_with_the_font_not_a_fixed_pixel(self):
        """A 5%-of-em limit: bigger type may tighten by more px, and the
        same book re-set in another font gets the same proportion."""
        self.assertLess(self.F.track_steps(30)[-1], self.F.track_steps(15)[-1])
        for size in (15, 20, 30):
            worst = self.F.track_steps(size)[-1]
            self.assertAlmostEqual(worst / float(size), -0.05, places=6)

    def test_the_first_step_is_no_tracking_at_all(self):
        self.assertEqual(self.F.track_steps(21)[0], 0.0)

    def test_tracking_never_widens(self):
        for size in (15, 20, 30):
            self.assertTrue(all(t <= 0 for t in self.F.track_steps(size)))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CaptionBoxKeepsItsDrawnFrame(unittest.TestCase):
    """The box's drawn frame must survive, and nothing outside it may move.

    A caption box is wiped WHOLE, and the frame is meant to be spared as a
    "long thin curve". A full-width rule is not reliably found that way: the
    fill estimate is a horizontal close, which a rule spanning the whole
    width survives, so the rule only registers as ink where the gradient
    happens to push it past the threshold — and the pieces are then too
    short to look like a curve. The frame was wiped and haloed, and the
    repaint reached a few px PAST it onto the artwork, which is what left
    p42 reading as a bare rectangle of colour ("yellow square very bad,
    rectangle") through three rounds.

    The box rectangle is known, so the frame is found where it actually is —
    near-solid dark rows and columns at the box's edges — and the repaint is
    confined inside it.
    """

    def setUp(self):
        self.F = load_fit()
        # a graded yellow box, black 3px frame, a word of lettering inside
        self.img = np.full((120, 300, 3), 240, np.uint8)
        # the DETECTED box does not coincide with the DRAWN frame — it runs
        # a few px wide of it, which is what let the repaint wipe the frame
        # and leave a tab of fill colour outside it on p42
        self.X0, self.Y0, self.X1, self.Y1 = 40, 30, 260, 90
        self.FX0, self.FY0 = self.X0 + 7, self.Y0 + 4
        self.FX1, self.FY1 = self.X1 - 5, self.Y1 - 3
        for r in range(self.FY0, self.FY1):
            t = (r - self.FY0) / float(self.FY1 - self.FY0)
            self.img[r, self.FX0:self.FX1] = (255, int(215 + 40 * t),
                                              int(40 + 180 * t))
        self.img[self.FY0:self.FY0 + 3, self.FX0:self.FX1] = 20
        self.img[self.FY1 - 3:self.FY1, self.FX0:self.FX1] = 20
        self.img[self.FY0:self.FY1, self.FX0:self.FX0 + 3] = 20
        self.img[self.FY0:self.FY1, self.FX1 - 3:self.FX1] = 20
        for cx in range(70, 235, 22):       # letters
            self.img[48:70, cx:cx + 13] = 15
        self.before = self.img.copy()

    def test_the_frame_survives_the_wipe(self):
        self.F.clean_caption_box(self.img, (self.X0, self.Y0, self.X1, self.Y1))
        for name, sl in (
                ("top", (slice(self.FY0, self.FY0 + 3), slice(self.FX0 + 6, self.FX1 - 6))),
                ("bottom", (slice(self.FY1 - 3, self.FY1), slice(self.FX0 + 6, self.FX1 - 6))),
                ("left", (slice(self.FY0 + 6, self.FY1 - 6), slice(self.FX0, self.FX0 + 3))),
                ("right", (slice(self.FY0 + 6, self.FY1 - 6), slice(self.FX1 - 3, self.FX1)))):
            self.assertLess(self.img[sl].max(), 90,
                            f"{name} rule of the frame was wiped")

    def test_nothing_outside_the_box_is_repainted(self):
        self.F.clean_caption_box(self.img, (self.X0, self.Y0, self.X1, self.Y1))
        out = np.abs(self.img.astype(int) - self.before.astype(int)).max(axis=2)
        out[self.FY0:self.FY1, self.FX0:self.FX1] = 0
        self.assertLess(out.max(), 12,
                        "the repaint reached past the drawn frame")

    def test_the_lettering_is_still_wiped(self):
        self.F.clean_caption_box(self.img, (self.X0, self.Y0, self.X1, self.Y1))
        inner = self.img[48:70, 70:235]
        self.assertLess((inner.min(axis=2) < 60).mean(), 0.02,
                        "lettering survived")

    def test_the_fill_keeps_its_gradient(self):
        self.F.clean_caption_box(self.img, (self.X0, self.Y0, self.X1, self.Y1))
        col = self.img[self.FY0 + 6:self.FY1 - 6, 150, 2].astype(int)
        self.assertGreater(col[-1] - col[0], 80,
                           "the graded fill was flattened into a slab")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CaptionCentresWhereTheLettererSetIt(unittest.TestCase):
    """A caption box is centred on the letterer's block, not on its rectangle.

    Balloons go through the anchor ladder, whose first rung is the original
    letterer's own text-block centre. Captions (`tint`/`dark`/`open`) took
    the width-weighted centroid of their row profile instead. For a graded
    caption box those differ: `caption_boxes` finds the box by its FILL, so
    on a yellow->white box the rectangle is only the coloured part and its
    centre sits above where the letterer set the type.

    On p42 the box centre is 175.5 and the letterer's block centre 181.5;
    the printed caption landed on rows 167-183 against the original's
    174-189, leaving 7px of flat colour underneath that read as a "yellow
    rectangle" in three consecutive reviews.
    """

    def setUp(self):
        self.F = load_fit()

    def test_block_anchor_gives_the_letterers_centre(self):
        rows = {y: (316, 588) for y in range(158, 193)}
        # p42 b01 exactly: the fill box is 158..193, the block 170..193
        _, cy = self.F.block_anchor((316, 170, 272, 23), rows,
                                    (316, 158, 272, 35))
        self.assertAlmostEqual(cy, 181.5, delta=1.0)

    def test_the_rectangle_centroid_is_the_wrong_target(self):
        """Guards the regression: the box centroid is 6px higher."""
        rows = {y: (316, 588) for y in range(158, 193)}
        wsum = sum(x1 - x0 for x0, x1 in rows.values())
        centroid = sum(y * (x1 - x0) for y, (x0, x1) in rows.items()) / wsum
        _, cy = self.F.block_anchor((316, 170, 272, 23), rows,
                                    (316, 158, 272, 35))
        self.assertGreater(cy - centroid, 4.0,
                           "the two anchors should differ on a graded box")

    def test_an_honest_block_is_not_moved(self):
        """A caption whose block already centres in its box stays put."""
        rows = {y: (100, 400) for y in range(100, 160)}
        _, cy = self.F.block_anchor((100, 115, 300, 30), rows,
                                    (100, 100, 300, 60))
        self.assertAlmostEqual(cy, 130.0, delta=1.0)


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LetteringWeldedToASparedCurveIsStillWiped(unittest.TestCase):
    """A letter drawn against the outline must not be spared WITH it.

    The wipe spares long thin curves so that the balloon outline and panel
    lines survive a leaked mask. But the original lettering runs right up to
    the outline, and at the ink threshold a letter that touches it becomes
    ONE component with it — so the curve test spared the letter too. It then
    stood on the page with the new text beside it: p215's balloon printed a
    stray "L" (the L of "EL", welded to the right arc: component 111x152,
    sparse 0.26) out on the artwork, and p147's caption boxes kept whole
    line-ends the same way (1554 px on one box).

    Neither gate could see it. leftover.py runs detection's letter_mask,
    which is a "dark text on light paper" RING test — a letter fused to the
    outline has no ring — and qa_scan only sees pixels that CHANGED.

    The outline is the wall around the fill; lettering lies inside it. So
    strip the component back to the part inside the balloon's own fill and
    ask the same question of that: a letter is compact, a panel line
    crossing the balloon still spans it."""

    def setUp(self):
        self.fit = load_fit()

    def _welded(self):
        """A balloon whose last letter touches the outline's right arc."""
        img = np.full((260, 440, 3), 255, np.uint8)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(img, (210, 130), (195, 120), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(70, 340, 26):               # a line of letters
            cv2.rectangle(img, (cx, 118), (cx + 18, 142), (10, 10, 10), -1)
        # ...and one more, hard against the arc (x=405 at this row)
        cv2.rectangle(img, (384, 118), (407, 142), (10, 10, 10), -1)
        mask = np.zeros((260, 440), np.uint8)
        cv2.ellipse(mask, (210, 130), (195, 120), 0, 0, 360, 255, -1)
        b = {"kind": "bubble", "bbox": [0, 0, 440, 260],
             "block": [70, 118, 330, 24]}
        return img, b, mask

    def test_the_welded_letter_is_wiped(self):
        img, b, mask = self._welded()
        self.fit.clean_bubble(img, b, mask)
        welded = np.zeros((260, 440), bool)
        welded[118:142, 384:401] = True
        self.assertLess((img.min(axis=2) < 100)[welded].mean(), 0.15,
                        "a letter welded to the outline survived the wipe")

    def test_the_outline_still_survives(self):
        img, b, mask = self._welded()
        self.fit.clean_bubble(img, b, mask)
        ring = np.zeros((260, 440), np.uint8)
        cv2.ellipse(ring, (210, 130), (195, 120), 0, 0, 360, 255, 3)
        ring[100:160, 360:440] = 0            # not where the weld was
        self.assertGreater((img.min(axis=2) < 100)[ring > 0].mean(), 0.8,
                           "the balloon outline was erased")

    def test_a_panel_line_crossing_the_balloon_is_kept(self):
        """The part inside the fill is re-tested, not blindly wiped: a rule
        that crosses the balloon still spans it and must stay."""
        img, b, mask = self._welded()
        cv2.line(img, (20, 210), (420, 210), (0, 0, 0), 3)
        self.fit.clean_bubble(img, b, mask)
        line = np.zeros((260, 440), bool)
        line[207:214, 120:300] = True
        self.assertGreater((img.min(axis=2) < 100)[line].mean(), 0.5,
                           "a panel line through the balloon was wiped")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LetteringWeldedToTheBoxFrameIsStillWiped(unittest.TestCase):
    """The same weld, in a caption box: the frame keeps the line-ends.

    p147's boxes printed "NTRA", "OLA" and a stray "E" — the tails of the
    original's longest lines — standing on top of the new text, because each
    touched the drawn frame and the frame is spared as a curve."""

    def setUp(self):
        self.fit = load_fit()

    def _box(self):
        """A framed yellow box whose middle line runs into the right rule.

        The frame sits AT the box edge, as a drawn one does, and the welded
        run is a whole line-end ("NTRA"): three letters touching each other
        and the rule, so the component stays sparse (0.18, as p147's 0.14)
        and the curve test spares the lot."""
        img = np.full((260, 520, 3), (90, 210, 245), np.uint8)   # yellow fill
        cv2.rectangle(img, (2, 2), (517, 257), (0, 0, 0), 3)     # drawn frame
        for cy in (40, 100, 160, 210):
            for cx in range(30, 430, 26):
                cv2.rectangle(img, (cx, cy), (cx + 18, cy + 24), (20, 20, 20), -1)
        for cx in (460, 486):             # the line-end, into the rule
            cv2.rectangle(img, (cx, 100), (cx + 29, 124), (20, 20, 20), -1)
        return img

    def test_the_welded_line_end_is_wiped(self):
        img = self._box()
        self.fit.clean_caption_box(img, (0, 0, 520, 260))
        welded = np.zeros((260, 520), bool)
        welded[100:124, 460:512] = True
        dark = img.max(axis=2) < 120
        self.assertLess(dark[welded].mean(), 0.15,
                        "a line-end welded to the frame survived the wipe")

    def test_the_frame_still_survives(self):
        img = self._box()
        self.fit.clean_caption_box(img, (0, 0, 520, 260))
        dark = img.max(axis=2) < 120
        self.assertGreater(dark[20:240, 1:4].mean(), 0.8,
                           "the box's left frame was wiped")
        self.assertGreater(dark[1:4, 20:500].mean(), 0.8,
                           "the box's top frame was wiped")

    def test_the_rest_of_the_lettering_is_gone(self):
        img = self._box()
        self.fit.clean_caption_box(img, (0, 0, 520, 260))
        body = np.zeros((260, 520), bool)
        body[40:234, 30:430] = True
        self.assertLess((img.max(axis=2) < 120)[body].mean(), 0.05,
                        "the box's own lettering survived")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class OpenEntryNeverPaintsARectangle(unittest.TestCase):
    """An `open` entry over a drawn balloon must not white out its window.

    `open` means "lettering on light ground with no balloon to mask", so its
    mask is simply the letterer's block — a RECTANGLE. Where detection
    misreads a real balloon as `open` (4 entries in this book) that rectangle
    crosses the drawn outline, and the outline arcs inside it are small
    enough to pass the letter-size test: inpainting them floods the window
    from its own white surroundings, so the page prints a white RECTANGLE
    with the balloon's sides erased (p182, p242).

    The bubble branch already refuses a blob that touches its window border
    for exactly this reason ("the white rectangle behind the bubble"); the
    open branch never got the same guard."""

    def setUp(self):
        self.fit = load_fit()

    def _page(self):
        img = np.full((300, 520, 3), (120, 110, 150), np.uint8)    # artwork
        cv2.rectangle(img, (40, 40), (150, 260), (60, 50, 90), -1)  # dark art
        cv2.ellipse(img, (260, 150), (110, 70), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (260, 150), (110, 70), 0, 0, 360, (0, 0, 0), 4)
        for cx in range(190, 320, 26):
            cv2.rectangle(img, (cx, 138), (cx + 18, 162), (10, 10, 10), -1)
        # the rectangle runs WIDER than the balloon, as in p242
        b = {"kind": "open", "bbox": [120, 130, 290, 44],
             "block": [120, 130, 290, 44]}
        mask = np.full((44, 290), 255, np.uint8)
        return img, b, mask

    def test_the_artwork_beside_the_balloon_is_untouched(self):
        img, b, mask = self._page()
        before = img.copy()
        self.fit.clean_bubble(img, b, mask)
        art = (slice(130, 174), slice(120, 148))
        self.assertLess(np.abs(img[art].astype(int) - before[art]).max(), 25,
                        "the artwork beside the balloon was painted over")

    def test_the_balloon_outline_survives(self):
        img, b, mask = self._page()
        self.fit.clean_bubble(img, b, mask)
        ring = np.zeros((300, 520), np.uint8)
        cv2.ellipse(ring, (260, 150), (110, 70), 0, 0, 360, 255, 4)
        band = np.zeros((300, 520), bool)
        band[130:174, :] = True                 # only where the window is
        sel = (ring > 0) & band
        self.assertGreater((img.min(axis=2) < 100)[sel].mean(), 0.6,
                           "the balloon's sides were erased")

    def test_its_own_lettering_is_still_wiped(self):
        img, b, mask = self._page()
        self.fit.clean_bubble(img, b, mask)
        letters = np.zeros((300, 520), bool)
        letters[138:162, 190:320] = True
        self.assertLess((img.min(axis=2) < 100)[letters].mean(), 0.15,
                        "the lettering the entry is for survived")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DarkArtworkSurvivesALeakedMask(unittest.TestCase):
    """Dark artwork reached through a leaked mask must not be repainted.

    The soft cut-off is already bounded to the balloon's own fill, but the
    STRICT one was bounded only by the mask — so where a mask spilled past
    the outline (p149: 558 px of mask for a 254 px text block) the artwork's
    own line work was read as old lettering and painted with the balloon's
    fill colour: white streaks and blobs across the ice.

    Lettering is what the letterer wrote, so the strict term belongs inside
    the text block or inside the fill, never merely inside the mask."""

    def setUp(self):
        self.fit = load_fit()

    def test_dark_art_outside_the_balloon_is_kept(self):
        img = np.full((300, 700, 3), (170, 140, 120), np.uint8)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(150, 340, 24):
            cv2.rectangle(img, (cx, 138), (cx + 16, 162), (10, 10, 10), -1)
        for cx in range(470, 660, 30):          # art detail beyond the balloon
            cv2.rectangle(img, (cx, 120), (cx + 14, 140), (30, 25, 20), -1)
        before = img.copy()
        b = {"kind": "bubble", "bbox": [0, 0, 700, 300],
             "block": [150, 138, 190, 24]}
        mask = np.full((300, 700), 255, np.uint8)       # the leak
        self.fit.clean_bubble(img, b, mask)
        art = (slice(120, 140), slice(470, 674))
        self.assertLess(np.abs(img[art].astype(int) - before[art]).max(), 30,
                        "artwork outside the balloon was repainted")

    def test_the_lettering_is_still_wiped(self):
        img = np.full((300, 700, 3), (170, 140, 120), np.uint8)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (250, 150), (150, 95), 0, 0, 360, (0, 0, 0), 3)
        for cx in range(150, 340, 24):
            cv2.rectangle(img, (cx, 138), (cx + 16, 162), (10, 10, 10), -1)
        b = {"kind": "bubble", "bbox": [0, 0, 700, 300],
             "block": [150, 138, 190, 24]}
        mask = np.full((300, 700), 255, np.uint8)
        self.fit.clean_bubble(img, b, mask)
        letters = np.zeros((300, 700), bool)
        letters[138:162, 150:356] = True
        self.assertLess((img.min(axis=2) < 100)[letters].mean(), 0.1,
                        "the balloon's own lettering survived")


def load_detect():
    """Import reletter_detect as a LIBRARY: configure() points it at a comic, so a
    test never has to fake sys.argv to get at the algorithms."""
    sys.path.insert(0, str(REPO / "relettering"))
    import reletter_detect
    reletter_detect.configure("unit-test-comic")
    return reletter_detect


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class BoxStripsMustShareOneXRange(unittest.TestCase):
    """An oval balloon must never be pulled into a caption box.

    The two halves of a gradient box are cut from ONE rectangle by a
    horizontal fill threshold, so their x extents agree to within a few px.
    Bounding the overlap against the WIDER member instead let a balloon of
    similar width but offset edges in: on p164 a 257x86 oval two panels up
    joined a caption box, which (a) grew the box from 233x73 to
    277x179, so the fit gave the caption 265px of writing width in a 230px
    box and both lines printed past the frame, and (b) left the box with a
    non-typeset member, so the whole box was skipped by the wipe and its two
    strips were cleaned as separate entries — a flat yellow slab with a
    white band at the seam and a tab of fill on the artwork.

    Measured over the book, the x-range IoU of the 31 real pairs runs from
    0.893 to 0.998 and the two false ones sit at 0.716 and 0.769."""

    def setUp(self):
        self.fit = load_fit()

    @staticmethod
    def _b(kind, x, y, w, h):
        return {"kind": kind, "bbox": [x, y, w, h],
                "block": [x + 4, y + 4, w - 8, h - 8]}

    def test_an_oval_above_the_strip_is_not_a_box_half(self):
        # p164: b01 is an ordinary balloon, b03/b04 the box's two strips
        bubs = [self._b("bubble", 142, 4, 257, 86),
                self._b("tint", 122, 110, 233, 37),
                self._b("bubble", 124, 145, 225, 38)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2, 3}),
                         [((122, 110, 355, 183), {2, 3})])

    def test_an_unplaced_tint_below_a_balloon_is_not_a_box(self):
        # p232: an oval balloon with a tint region 4px below it
        bubs = [self._b("bubble", 274, 852, 233, 108),
                self._b("tint", 243, 964, 220, 86)]
        self.assertEqual(self.fit.caption_boxes(bubs, {1, 2}),
                         [((243, 964, 463, 1050), {2})])

    def test_the_real_strips_still_pair(self):
        """The tightest real pair in the book (p270, IoU 0.893) and the
        widest overlap (p219, 26px) must both survive."""
        p270 = [self._b("tint", 60, 1000, 292, 40),
                self._b("bubble", 42, 1017, 327, 79)]
        self.assertEqual(self.fit.caption_boxes(p270, {1, 2}),
                         [((42, 1000, 369, 1096), {1, 2})])
        p219 = [self._b("tint", 694, 1202, 314, 99),
                self._b("bubble", 696, 1275, 314, 126)]
        self.assertEqual(self.fit.caption_boxes(p219, {1, 2}),
                         [((694, 1202, 1010, 1401), {1, 2})])


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class CaptionBoxIsOneBlockOfType(unittest.TestCase):
    """A split caption box must print as one continuous run of lines.

    Detection cuts a gradient box into a `tint` strip and a `bubble` strip,
    and each was centred in its OWN strip — so the box printed with a blank
    line at the seam: "too much space between paragraphs, as if it were 2
    bubbles not one" (p147's bottom-left box, p219). The box is one piece of
    type: the letterer set its lines on one grid, centred on his own block.
    """

    def setUp(self):
        self.fit = load_fit()

    def test_the_lines_run_continuously_across_the_seam(self):
        # p147 b14 (2 lines) + b15 (2 lines), lh 21, letterer's block
        # y 1839..1970 -> centre 1904.5
        got = self.fit.stack_box_lines([(14, 2), (15, 2)], 21, 1904.5)
        self.assertEqual(got, {14: [1862.5, 1883.5], 15: [1904.5, 1925.5]})

    def test_the_block_is_centred_on_the_letterers_centre(self):
        got = self.fit.stack_box_lines([(1, 3), (2, 2)], 20, 500.0)
        ys = got[1] + got[2]
        self.assertEqual(ys, [450.0, 470.0, 490.0, 510.0, 530.0])
        self.assertAlmostEqual((ys[0] + ys[-1] + 20) / 2.0, 500.0)

    def test_a_lone_member_is_still_centred(self):
        self.assertEqual(self.fit.stack_box_lines([(7, 2)], 10, 100.0),
                         {7: [90.0, 100.0]})


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class OpenEntryFindsTheBalloonDetectionMissed(unittest.TestCase):
    """`open` is also detection's fallback when its flood fails on a balloon.

    Then nothing bounds the new type: p242's shout was set 141px wide
    in a 132px oval and printed across the outline on both sides. Its block
    could not catch it (the blob cluster had swallowed both arcs, so the
    block was 148px) and neither could the size ceiling (the arcs carried
    the measured cap height from ~25px to 36px). Flooding out from the block
    finds the balloon, and that it comes back NARROWER than the block is the
    proof that it IS one — an escaped flood comes back wider (p222 b04's ran
    down the tail and returned 194px for a 132px block)."""

    def setUp(self):
        self.fit = load_fit()

    def _shout(self):
        """A balloon detection called `open`: its two arcs cross the block."""
        img = np.full((300, 500, 3), (180, 150, 130), np.uint8)
        cv2.ellipse(img, (250, 150), (76, 40), 0, 0, 360, (252, 252, 252), -1)
        cv2.ellipse(img, (250, 150), (76, 40), 0, 0, 360, (0, 0, 0), 5)
        cv2.rectangle(img, (0, 146), (178, 155), (0, 0, 0), -1)   # panel rule
        cv2.rectangle(img, (322, 146), (500, 155), (0, 0, 0), -1)
        for cx in range(200, 300, 26):
            cv2.rectangle(img, (cx, 138), (cx + 16, 160), (10, 10, 10), -1)
        return img

    def test_the_balloon_is_found_and_is_narrower_than_the_block(self):
        img = self._shout()
        b = {"kind": "open", "bbox": [174, 128, 152, 44],
             "block": [174, 128, 152, 44]}
        got = self.fit.unfound_balloon(img, b)
        self.assertIsNotNone(got, "the drawn balloon was not found")
        x0, x1 = got
        self.assertLess(x1 - x0, 152)               # narrower than the block
        self.assertGreater(x0, 168)                 # and it is the oval
        self.assertLess(x1, 332)

    def test_text_on_open_artwork_finds_nothing(self):
        """A real `open` entry has no balloon: the flood escapes and comes
        back wider than the block, which is how it is rejected."""
        img = np.full((300, 500, 3), (230, 228, 225), np.uint8)
        for cx in range(200, 300, 26):
            cv2.rectangle(img, (cx, 138), (cx + 16, 160), (10, 10, 10), -1)
        b = {"kind": "open", "bbox": [196, 136, 108, 26],
             "block": [196, 136, 108, 26]}
        self.assertIsNone(self.fit.unfound_balloon(img, b))

    def test_a_flood_that_escapes_is_refused(self):
        """p222 b04: the flood ran out along the tail. Wider than the block
        is not an interior, whatever else it is."""
        img = np.full((300, 500, 3), (120, 90, 40), np.uint8)
        cv2.ellipse(img, (250, 150), (70, 38), 0, 0, 360, (252, 252, 252), -1)
        # the tail: an unwalled corridor out to the left, so the flood leaves
        cv2.rectangle(img, (20, 140), (250, 160), (252, 252, 252), -1)
        for cx in range(205, 290, 26):
            cv2.rectangle(img, (cx, 138), (cx + 16, 160), (10, 10, 10), -1)
        b = {"kind": "open", "bbox": [201, 136, 104, 26],
             "block": [201, 136, 104, 26]}
        self.assertIsNone(self.fit.unfound_balloon(img, b))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class DiagonalLobesSplitOnTheirSeam(unittest.TestCase):
    """Two joined balloons set DIAGONALLY have no straight cut between them.

    The joined-pair split picks a row or a column between the two text
    blocks. When the blocks overlap on BOTH axes there is no such line: on
    p294 b07's block reaches down and right into b12's first line, so the
    row window ran from b07's block bottom (1971) UP to b12's block top
    (1938) and the cut landed at 1954 — straight through b07's own last
    line. The mask lost that line, which the cleaner then could not reach,
    and the profile kept the part of b12's balloon the flood had filled, so
    the last new line was set 274px wide out of a 220px balloon.

    Equidistant assignment from each lobe's own lettering splits them the
    way the artist drew them, and never crosses either lobe's letters."""

    def setUp(self):
        self.det = load_detect()

    def _pair(self):
        """Two overlapping ovals, upper-left and lower-right, one region."""
        union = np.zeros((400, 500), np.uint8)
        cv2.ellipse(union, (160, 150), (130, 95), 0, 0, 360, 1, -1)
        cv2.ellipse(union, (330, 270), (120, 85), 0, 0, 360, 1, -1)
        a = np.zeros((400, 500), np.uint8)
        a[110:200, 60:280] = 1          # the upper-left lobe's lettering
        b = np.zeros((400, 500), np.uint8)
        b[240:310, 240:430] = 1         # the lower-right lobe's lettering
        return union, a, b

    def test_each_lobe_keeps_its_own_letters(self):
        union, sa, sb = self._pair()
        ma, mb = self.det.seam_split(union, (sa, sb))
        own_a, own_b = (sa > 0) & (union > 0), (sb > 0) & (union > 0)
        self.assertTrue((ma[own_a] > 0).all(),
                        "the upper lobe lost part of its own lettering")
        self.assertTrue((mb[own_b] > 0).all(),
                        "the lower lobe lost part of its own lettering")

    def test_the_two_halves_tile_the_region(self):
        union, sa, sb = self._pair()
        ma, mb = self.det.seam_split(union, (sa, sb))
        self.assertTrue(((ma > 0) | (mb > 0) == (union > 0)).all())
        self.assertFalse(((ma > 0) & (mb > 0)).any())

    def test_the_seam_is_not_a_straight_row(self):
        """A row cut would hand the upper lobe the whole top band, including
        the right-hand lobe's top arc."""
        union, sa, sb = self._pair()
        ma, _ = self.det.seam_split(union, (sa, sb))
        rows = np.flatnonzero(ma.any(axis=1))
        widths = [(np.flatnonzero(ma[r])[-1] - np.flatnonzero(ma[r])[0])
                  for r in rows]
        self.assertLess(max(widths), 300,
                        "the upper lobe swallowed its sibling's balloon")


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class LeakedMaskRescuedByRectangles(unittest.TestCase):
    """The per-comic escape hatch for a mask that leaked past recognition.

    p276 b04 is TWO balloons drawn on pale sky, returned as one 1269x528
    entry whose mask covers 79% of the panel: mask_lobes sees one blob, the
    fit places nothing, cleaning it would repaint sky, and splitting the
    entry in detection would renumber the page and invalidate its positional
    transcripts. One hand-measured rectangle per balloon, each flooded
    inside itself, recovers both."""

    def setUp(self):
        self.fit = load_fit()

    def _panel(self):
        img = np.full((500, 900, 3), (150, 200, 240), np.uint8)   # pale sky
        for cxy, ab in (((250, 250), (170, 130)), ((650, 300), (110, 90))):
            cv2.ellipse(img, cxy, ab, 0, 0, 360, (252, 252, 252), -1)
            cv2.ellipse(img, cxy, ab, 0, 0, 360, (0, 0, 0), 4)
        return img

    def test_both_balloons_come_back_as_their_own_lobe(self):
        img = self._panel()
        got = self.fit.lobes_from_rects(
            img, [[60, 100, 440, 400], [520, 195, 780, 405]],
            [0, 0, 900, 500])
        self.assertIsNotNone(got)
        mask, lobes, _ = got
        self.assertEqual(len(lobes), 2)
        # each lobe is its own balloon, not the rectangle and not the sky
        for (rows, cx) in zip(lobes, (250, 650)):
            xs = [v for sp in rows.values() for v in sp]
            self.assertAlmostEqual((min(xs) + max(xs)) / 2, cx, delta=12)
        self.assertLess(mask.mean(), 0.25, "the sky came with it")

    def test_the_sky_is_never_inside_the_mask(self):
        img = self._panel()
        mask, _, _ = self.fit.lobes_from_rects(
            img, [[60, 100, 440, 400], [520, 195, 780, 405]],
            [0, 0, 900, 500])
        sky = np.zeros((500, 900), bool)
        sky[440:500, 0:900] = True          # well below both balloons
        self.assertEqual(int(mask[sky].sum()), 0)

    def test_a_rectangle_that_misses_is_refused(self):
        """Better ignored than trusted: a rect off the balloon floods sky."""
        img = self._panel()
        self.assertIsNone(self.fit.lobes_from_rects(
            img, [[60, 420, 440, 495]], [0, 0, 900, 500]))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class OwnBlockStopsAtTheSiblingsClaim(unittest.TestCase):
    """"A lobe's own text block" may not include its sibling's lettering.

    Every guard in detection protects the block a lobe's text has to go in.
    But a block is a blob cluster, and p294 b07's reached over b12's first
    line — so the guard held "SI DESEAS" inside b07's mask, and with it the
    slice of b12's balloon b07's flood had filled."""

    def setUp(self):
        self.det = load_detect()

    def test_the_overlap_belongs_to_neither(self):
        m = self.det.own_block_mask((200, 300), (0, 0),
                                    [10, 10, 200, 100], ([150, 60, 120, 80],))
        self.assertTrue(m[20, 20])            # its own, uncontested
        self.assertFalse(m[80, 200])          # inside the sibling's block
        self.assertTrue(m[20, 200])           # above the sibling's block

    def test_no_sibling_keeps_the_whole_block(self):
        m = self.det.own_block_mask((200, 300), (0, 0), [10, 10, 200, 100])
        self.assertEqual(int(m.sum()), 200 * 100)

    def test_the_frame_origin_is_honoured(self):
        m = self.det.own_block_mask((100, 100), (50, 40), [60, 50, 20, 20])
        self.assertTrue(m[15, 15])            # block (60,50) -> frame (10,10)
        self.assertFalse(m[5, 5])

    def test_an_overlap_with_no_sibling_lettering_is_kept(self):
        """p294 b04's block overlaps b06's by a 30px sliver that holds none
        of b06's type and is where b04's own last line ends. Dropping it
        cost b04 five size steps."""
        letters = np.zeros((200, 300), bool)
        letters[100:120, 240:290] = True      # the sibling's type, well clear
        m = self.det.own_block_mask((200, 300), (0, 0), [10, 10, 200, 100],
                                    ([180, 60, 120, 80],), letters=letters)
        self.assertTrue(m[80, 200], "a sliver with no sibling type was cut")
        self.assertEqual(int(m.sum()), 200 * 100)

    def test_the_siblings_own_line_is_still_dropped(self):
        letters = np.zeros((200, 300), bool)
        letters[70:90, 190:280] = True        # the sibling's line, in the overlap
        m = self.det.own_block_mask((200, 300), (0, 0), [10, 10, 200, 100],
                                    ([180, 60, 120, 80],), letters=letters)
        self.assertFalse(m[80, 200], "the sibling's own line stayed protected")
        self.assertTrue(m[20, 20])            # and the rest is untouched


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class ACaptionBoxHasNoOverhang(unittest.TestCase):
    """The first and last line of a BALLOON may overhang its rows a little —
    glyphs fill about 70% of the line box, and the arc forgives it. A drawn
    caption FRAME forgives nothing: the fill IS the writing area.

    p214's recovered box is 47px of interior; two lines at the book size are
    48px of line box, and the 0.16-of-a-line-height allowance at each outer
    edge let them "fit" — the type printed across the top and bottom rules.
    """

    def setUp(self):
        self.fit = load_fit()

    @staticmethod
    def _rows(y0, y1, x0=20, x1=180):
        return {y: (x0, x1) for y in range(y0, y1)}

    def test_two_lines_do_not_fit_a_box_that_is_a_line_short(self):
        fit = self.fit.Fitter()
        words = self.fit.parse_runs("CIERTA MAÑANA...")
        rows = self._rows(112, 153)          # 41px of interior
        got = self.fit.wrap_at(fit, words, rows, 22, 132.5, set(),
                               hold_cy=True, tight=True)
        self.assertTrue(got is None or len(got) < 2
                        or got[-1][0] + 24 <= 153,
                        "the last line hangs past the box")

    def test_a_size_that_does_fit_is_still_accepted(self):
        fit = self.fit.Fitter()
        words = self.fit.parse_runs("CIERTA MAÑANA...")
        rows = self._rows(112, 153)
        got = self.fit.wrap_at(fit, words, rows, 18, 132.5, set(),
                               hold_cy=True, tight=True)
        self.assertIsNotNone(got, "18px fits 41px of interior and was refused")
        self.assertEqual(len(got), 2)
        self.assertGreaterEqual(got[0][0], 112)
        self.assertLessEqual(got[-1][0] + 20, 153)

    def test_a_balloon_keeps_its_overhang(self):
        """Unchanged for every other kind: tight defaults off."""
        fit = self.fit.Fitter()
        words = self.fit.parse_runs("CIERTA MAÑANA...")
        rows = self._rows(112, 153)
        self.assertIsNotNone(
            self.fit.wrap_at(fit, words, rows, 22, 132.5, set(),
                             hold_cy=True))


@unittest.skipUnless(HAVE_CV2, "cv2 only in .venv-reletter")
class RescuedLobesAreCleanedOneByOne(unittest.TestCase):
    """A `lobes` entry's own block spans every balloon and the artwork
    between them, and the cleaner reads that block — so cleaning the entry
    as one repainted p276's green sphere and the darts behind its left
    balloon. `lobes_from_rects` hands back one cleaning job per lobe, with
    the original lettering inside that interior as its block."""

    def setUp(self):
        self.fit = load_fit()

    def _panel(self):
        img = np.full((500, 900, 3), (150, 200, 240), np.uint8)   # pale sky
        cv2.circle(img, (80, 300), 90, (90, 170, 90), -1)         # art: sphere
        for cy in range(120, 260, 26):                            # art: darts
            cv2.rectangle(img, (30, cy), (70, cy + 14), (40, 30, 120), -1)
        for cxy, ab in (((250, 250), (150, 120)), ((650, 300), (110, 90))):
            cv2.ellipse(img, cxy, ab, 0, 0, 360, (252, 252, 252), -1)
            cv2.ellipse(img, cxy, ab, 0, 0, 360, (0, 0, 0), 4)
        for cx in range(160, 340, 26):
            cv2.rectangle(img, (cx, 240), (cx + 16, 262), (10, 10, 10), -1)
        for cx in range(580, 720, 26):
            cv2.rectangle(img, (cx, 290), (cx + 16, 312), (10, 10, 10), -1)
        return img

    def test_one_job_per_lobe_each_bounded_to_its_own_balloon(self):
        img = self._panel()
        got = self.fit.lobes_from_rects(
            img, [[80, 110, 420, 390], [520, 195, 780, 405]],
            [0, 0, 900, 500])
        self.assertIsNotNone(got)
        _, lobes, jobs = got
        self.assertEqual(len(jobs), 2)
        for (bb, blk, m) in jobs:
            self.assertEqual(m.shape, (bb[3], bb[2]))
            # the block is the lettering inside the interior, not the rect
            self.assertGreaterEqual(blk[0], bb[0])
            self.assertLessEqual(blk[0] + blk[2], bb[0] + bb[2])
            self.assertLess(blk[2] * blk[3], bb[2] * bb[3])

    def test_the_artwork_behind_the_left_balloon_survives(self):
        img = self._panel()
        before = img.copy()
        b = {"kind": "bubble", "bbox": [0, 0, 900, 500],
             "block": [40, 110, 700, 260]}        # the polluted block
        _, _, jobs = self.fit.lobes_from_rects(
            img, [[80, 110, 420, 390], [520, 195, 780, 405]], b["bbox"])
        for bb, blk, m in jobs:
            self.fit.clean_bubble(img, dict(b, bbox=bb, block=blk), m,
                                  set(range(bb[1] - 6, bb[1] + bb[3] + 7)))
        art = (slice(110, 260), slice(25, 75))    # the darts, left of it all
        self.assertLess(np.abs(img[art].astype(int) - before[art]).max(), 30,
                        "the artwork behind the balloon was repainted")

    def test_both_balloons_lettering_is_still_wiped(self):
        img = self._panel()
        b = {"kind": "bubble", "bbox": [0, 0, 900, 500],
             "block": [40, 110, 700, 260]}
        _, _, jobs = self.fit.lobes_from_rects(
            img, [[80, 110, 420, 390], [520, 195, 780, 405]], b["bbox"])
        for bb, blk, m in jobs:
            self.fit.clean_bubble(img, dict(b, bbox=bb, block=blk), m,
                                  set(range(bb[1] - 6, bb[1] + bb[3] + 7)))
        for sl in ((slice(240, 262), slice(160, 356)),
                   (slice(290, 312), slice(580, 736))):
            self.assertLess((img.min(axis=2) < 100)[sl].mean(), 0.12,
                            "a balloon kept its own lettering")
