"""Unit tests for untext.py (text removal from one image).
Needs cv2, which lives only in .venv-untext (`just setup-untext`):
  .venv-untext/bin/python -m unittest tests.test_untext
Skipped under the base .venv."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import cv2  # noqa: F401
except ImportError:  # base .venv: no cv2
    untext = None
else:
    import untext

SKY = (56, 112, 141)
SNOW = (253, 252, 248)
OLIVE = (130, 134, 65)


def letters(size, ground, ink, rects):
    """A flat ground with block 'letters' (rectangles) painted on it."""
    im = Image.new("RGB", size, ground)
    d = ImageDraw.Draw(im)
    for r in rects:
        d.rectangle(r, fill=ink)
    return im


WORD = [(40 + 30 * i, 40, 60 + 30 * i, 80) for i in range(6)]  # 6 letters


def shadowed(size, ground, ink, rects, radius):
    """Letters with a soft dark drop shadow `radius` px wide round them."""
    shape = Image.new("L", size, 0)
    d = ImageDraw.Draw(shape)
    for r in rects:
        d.rectangle(r, fill=255)
    halo = shape.filter(ImageFilter.MaxFilter(2 * (radius // 2) + 1)).filter(
        ImageFilter.GaussianBlur(radius / 2))
    base = np.array(Image.new("RGB", size, ground), float)
    a = np.array(halo, float)[..., None] / 255 * 0.6
    base = base * (1 - a)
    base[np.array(shape) > 0] = ink
    return Image.fromarray(base.astype(np.uint8))


ART = [(235, 235, 230), (5, 5, 8), (125, 118, 114)]
GOLD = (235, 162, 37)


def busy(size):
    """Artwork with no ground to speak of: stripes of white, black, grey."""
    a = np.zeros((size[1], size[0], 3), np.uint8)
    for x in range(size[0]):
        a[:, x] = ART[(x // 6) % 3]
    return a


def outlined(a, rects, ink, edge, width):
    """'Sticker' letters: flat `ink` with a flat `edge` outline round them."""
    for x0, y0, x1, y1 in rects:
        a[y0 - width:y1 + width + 1, x0 - width:x1 + width + 1] = edge
    for x0, y0, x1, y1 in rects:
        a[y0:y1 + 1, x0:x1 + 1] = ink
    return a


def pixels(path):
    with Image.open(path) as im:
        return np.array(im)


def box_of(rects, pad=4):
    return (min(r[0] for r in rects) - pad, min(r[1] for r in rects) - pad,
            max(r[2] for r in rects) + pad, max(r[3] for r in rects) + pad)


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class OutputPaths(unittest.TestCase):
    def test_outputs_sit_beside_the_input(self):
        out, mask = untext.output_paths(Path("/x/Series 13 - Title 012 cover.jpg"))
        self.assertEqual(out, Path("/x/Series 13 - Title 012 cover-untexted.jpg"))
        self.assertEqual(mask, Path("/x/Series 13 - Title 012 cover-untexted-mask.png"))

    def test_refuses_its_own_output(self):
        for name in ("a-untexted.jpg", "a-untexted-mask.png"):
            with self.assertRaises(ValueError):
                untext.output_paths(Path("/x") / name)


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class BlockLetters(unittest.TestCase):
    def test_light_letters_on_dark_ground(self):
        im = np.array(letters((260, 120), SKY, (255, 255, 255), WORD))
        m = untext.text_mask(im, [box_of(WORD)])
        self.assertTrue(m[60, 50])        # inside a letter
        self.assertFalse(m[110, 10])      # far ground

    def test_dark_letters_on_light_ground(self):
        im = np.array(letters((260, 120), SNOW, OLIVE, WORD))
        m = untext.text_mask(im, [box_of(WORD)])
        self.assertTrue(m[60, 50])
        self.assertFalse(m[110, 10])

    def test_artwork_of_another_colour_is_not_text(self):
        # a grey rock inside the text box, beside olive letters on snow
        rock = (220, 85, 238, 95)
        im = letters((260, 120), SNOW, OLIVE, WORD)
        ImageDraw.Draw(im).rectangle(rock, fill=(105, 111, 113))
        m = untext.text_mask(np.array(im), [box_of(WORD + [rock])])
        self.assertTrue(m[60, 50])
        self.assertFalse(m[90, 229])

    def test_a_box_taller_than_wide_is_not_a_line_of_text(self):
        # the detector fires on artwork (face markings, a hilt) as tall
        # boxes; horizontal text is never taller than wide
        mark = [(100, 20, 130, 50), (100, 70, 130, 100), (100, 120, 130, 150)]
        im = np.array(letters((260, 180), SKY, (255, 255, 255), mark))
        m = untext.text_mask(im, [box_of(mark)])
        self.assertFalse(m.any())

    def test_low_contrast_marks_are_not_text(self):
        # face markings on skin, a signature in ink on dark art: lettering
        # is set off strongly from the art, these are not
        im = np.array(letters((260, 120), (70, 25, 25), (120, 41, 43), WORD))
        self.assertFalse(untext.text_mask(im, [box_of(WORD)]).any())

    def test_painted_marks_are_not_text(self):
        # strong contrast but a PAINTED fill (texture, brush strokes):
        # lettering is flat colour
        rng = np.random.default_rng(1)
        im = np.array(letters((260, 120), SKY, (220, 220, 220), WORD)).astype(float)
        shape = np.array(letters((260, 120), (0, 0, 0), (255, 255, 255), WORD))[..., 0] > 0
        im[shape] += rng.normal(0, 25, (shape.sum(), 3))
        m = untext.text_mask(np.clip(im, 0, 255).astype(np.uint8), [box_of(WORD)])
        self.assertFalse(m.any())

    def test_faint_thin_mark_is_text(self):
        # a tiny anti-aliased mark (a registered sign): a BLEND of the
        # letters' white and the sky, 70 levels off pure white
        mark = (226, 30, 230, 34)
        im = letters((260, 120), SKY, (255, 255, 255), WORD)
        ImageDraw.Draw(im).rectangle(mark, fill=(197, 221, 232))
        m = untext.text_mask(np.array(im), [box_of(WORD + [mark])])
        self.assertTrue(m[32, 228])

    def test_hollow_thin_mark_is_taken_whole(self):
        # a 1px open ring (a TM's counter, an R's circle) beside big letters:
        # its middle is ground, but the mark goes as one piece
        big = [(40 + 50 * i, 40, 80 + 50 * i, 100) for i in range(4)]
        im = letters((340, 140), SKY, (255, 255, 255), big)
        d = ImageDraw.Draw(im)
        c = (205, 221, 232)
        d.line([(250, 110), (262, 110)], fill=c)
        d.line([(250, 110), (250, 122)], fill=c)
        d.line([(250, 122), (262, 122)], fill=c)
        m = untext.text_mask(np.array(im), [box_of(big + [(250, 110, 262, 122)])])
        self.assertTrue(m[116, 257])


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class BusyGround(unittest.TestCase):
    """A title set straight on the artwork: the box's border is art, not a
    ground, and the letters are read by an outline of their own."""

    def test_outlined_letters_on_busy_art_are_text(self):
        im = outlined(busy((300, 140)), WORD, GOLD, (252, 252, 252), 6)
        m = untext.text_mask(im, [box_of(WORD, pad=8)])
        self.assertTrue(m[60, 50])        # inside a letter
        self.assertTrue(m[60, 36])        # on the outline, 4 px out
        self.assertTrue(m[60, 65])        # outline between two letters
        self.assertFalse(m[60, 12])       # art, well clear of the outline
        self.assertFalse(m[125, 150])     # art below the title

    def test_busy_art_alone_is_not_text(self):
        self.assertFalse(untext.text_mask(busy((300, 140)), [box_of(WORD)]).any())

    def test_flat_patch_with_no_outline_on_busy_art_is_not_text(self):
        # a flat piece of art inside the box (a buckle, a light) has no
        # lettering's outline round it
        im = busy((300, 140))
        for x0, y0, x1, y1 in WORD:
            im[y0:y1 + 1, x0:x1 + 1] = GOLD
        self.assertFalse(untext.text_mask(im, [box_of(WORD, pad=8)]).any())


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class Halo(unittest.TestCase):
    def test_mask_covers_the_drop_shadow(self):
        im = np.array(shadowed((300, 160), SKY, (255, 255, 255), WORD, 10))
        m = untext.text_mask(im, [box_of(WORD)])
        self.assertTrue(m[60, 31])        # 9 px left of the first letter
        self.assertFalse(m[60, 5])        # 35 px out: plain sky

    def test_busy_art_round_the_letters_is_no_halo(self):
        # the ground's brightness rises and falls with the distance from the
        # letters (folds, hair, a face round a false detection): a halo
        # fades ONE way, so the walk must stop at the first turn
        im = letters((300, 160), SKY, (255, 255, 255), WORD)
        a = np.array(im, float)
        shape = np.zeros((160, 300), np.uint8)
        for x0, y0, x1, y1 in WORD:
            shape[y0:y1 + 1, x0:x1 + 1] = 1
        dist = cv2.distanceTransform(1 - shape, cv2.DIST_L2, 5)
        wave = 40 * np.sin(dist / 3)[..., None]
        a = np.where(shape[..., None] > 0, a, np.clip(a + wave, 0, 255))
        m = untext.text_mask(a.astype(np.uint8), [box_of(WORD)])
        self.assertTrue(m[60, 38])
        self.assertFalse(m[60, 22])       # 18 px out: art, not a halo

    def test_flat_ground_keeps_the_mask_tight(self):
        im = np.array(letters((300, 160), SKY, (255, 255, 255), WORD))
        m = untext.text_mask(im, [box_of(WORD)])
        self.assertTrue(m[60, 38])        # just outside: anti-alias margin
        self.assertFalse(m[60, 28])       # 12 px out: nothing to remove


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class Fill(unittest.TestCase):
    def test_paste_back_keeps_every_unmasked_pixel(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (50, 60, 3), dtype=np.uint8)
        fill = np.zeros_like(img)
        mask = np.zeros((50, 60), np.uint8)
        mask[10:20, 10:30] = 255
        res = untext.paste_back(img, fill, mask)
        self.assertTrue((res[mask == 0] == img[mask == 0]).all())
        self.assertTrue((res[mask > 0] == 0).all())

    def test_windows_cover_every_masked_pixel(self):
        mask = np.zeros((3000, 2000), np.uint8)
        mask[100:300, 200:1800] = 255     # a logo at the top
        mask[2800:2850, 600:1400] = 255   # a title at the bottom
        wins = untext.windows(mask)
        covered = np.zeros_like(mask)
        for x0, y0, x1, y1 in wins:
            self.assertTrue(0 <= x0 < x1 <= 2000 and 0 <= y0 < y1 <= 3000)
            covered[y0:y1, x0:x1] = 255
        self.assertTrue((covered[mask > 0] > 0).all())
        # two blocks this far apart are filled separately, not as one
        # window the size of the page
        self.assertEqual(len(wins), 2)

    def test_small_image_is_one_window(self):
        mask = np.zeros((1286, 900), np.uint8)
        mask[10:300, 150:750] = 255
        mask[1190:1240, 190:710] = 255
        self.assertEqual(untext.windows(mask), [(0, 0, 900, 1286)])


def fake_fill(img, mask):
    out = img.copy()
    out[mask > 0] = (1, 2, 3)
    return out


@unittest.skipIf(untext is None, "needs .venv-untext (cv2)")
class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.src = self.dir / "Title 012 cover.jpg"
        letters((260, 120), SKY, (255, 255, 255), WORD).save(self.src, quality=95)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def detect(self, img):
        self.calls += 1
        return [box_of(WORD)]

    def run_untext(self, src=None, detect=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return untext.run(src or self.src, detect=detect or self.detect,
                              fill=fake_fill)

    def test_writes_image_and_mask_beside_the_input(self):
        self.run_untext()
        out = self.dir / "Title 012 cover-untexted.jpg"
        mask = self.dir / "Title 012 cover-untexted-mask.png"
        self.assertTrue(out.exists() and mask.exists())
        with Image.open(out) as im:
            self.assertEqual((im.format, im.size), ("JPEG", (260, 120)))

    def test_never_touches_the_input(self):
        before = self.src.read_bytes()
        self.run_untext()
        self.assertEqual(self.src.read_bytes(), before)

    def test_an_existing_mask_is_reused(self):
        mask = self.dir / "Title 012 cover-untexted-mask.png"
        hand = np.zeros((120, 260), np.uint8)
        hand[0:10, 0:10] = 255            # a hand-painted mask
        Image.fromarray(hand).save(mask)
        self.run_untext()
        self.assertEqual(self.calls, 0)
        res = pixels(self.dir / "Title 012 cover-untexted.jpg")
        self.assertLess(int(res[5, 5].sum()), 30)       # filled
        self.assertGreater(int(res[60, 50].sum()), 700)  # letter untouched

    def test_a_deleted_mask_is_detected_again(self):
        self.run_untext()
        (self.dir / "Title 012 cover-untexted-mask.png").unlink()
        self.run_untext()
        self.assertEqual(self.calls, 2)

    def test_no_text_writes_an_empty_mask_and_no_image(self):
        self.run_untext(detect=lambda img: [])
        self.assertFalse((self.dir / "Title 012 cover-untexted.jpg").exists())
        mask = pixels(self.dir / "Title 012 cover-untexted-mask.png")
        self.assertEqual(mask.shape, (120, 260))
        self.assertFalse(mask.any())

    def test_transparency_survives(self):
        src = self.dir / "Title 012 logo.png"
        im = letters((260, 120), SKY, (255, 255, 255), WORD).convert("RGBA")
        alpha = np.full((120, 260), 255, np.uint8)
        alpha[:, :20] = 0
        im.putalpha(Image.fromarray(alpha))
        im.save(src)
        self.run_untext(src=src)
        out = pixels(self.dir / "Title 012 logo-untexted.png")
        self.assertEqual(out.shape, (120, 260, 4))
        self.assertTrue((out[..., 3] == alpha).all())


if __name__ == "__main__":
    unittest.main()
