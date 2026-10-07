#!/usr/bin/env python3
"""Webcomic pipeline: upscale vertical pages to the template canvas size
(Real-ESRGAN locally by default, or iloveimg.com x2 with --engine iloveimg),
then compose each one onto template.xcf with GIMP (fit to canvas, keep
proportions, white background, hide template layers) and export a PDF per page.

Steps (each is resumable — already-produced files are skipped):
  scan     report what will be processed (horizontal spreads split into -1/-2)
  upscale  scale each page to exactly its size on the canvas; horizontal
           spreads go through the AI whole, then split into -1/-2 halves
  compose  GIMP: place each upscaled image on the template -> xcf/ + pdf/

Layout (relative to this repo):
  <comic folder>/            source images (jpg/png)
  upscaled/<comic>/          x2 images downloaded from iloveimg
  xcf/<comic>/               one .xcf per page (template + page layer)
  pdf/<comic>/               one .pdf per page
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent
TEMPLATE = REPO / "template.xcf"
GIMP = "/Applications/GIMP-2.10.app/Contents/MacOS/gimp"
UPSCALE_URL = "https://www.iloveimg.com/upscale-image"
REALESRGAN = REPO / "tools" / "realesrgan" / "realesrgan-ncnn-vulkan"
# line-art model: keeps lettering intact AND is color-faithful
# (realesrgan-x4plus-anime brightens/desaturates the artwork noticeably)
REALESRGAN_MODEL = "realesr-animevideov3"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
MAX_UPSCALE_PIXELS = 6_000_000  # pages this large are reported, not upscaled
BLEED_IN = 0.125  # standard print bleed per side
GEOMETRY_CACHE = REPO / "template_geometry.json"

# Python-Fu snippet that measures the template: canvas size and, for each
# guide layer, its inner white box (skip the leading white background, cross
# the colored guide band, the box starts at the next white pixel).
GEOMETRY_PROBE = r"""
from gimpfu import *
import sys
img = pdb.gimp_file_load('%s', 't')
sys.stdout.write('CANVAS %%d %%d\n' %% (img.width, img.height))
sys.stdout.write('RES %%f\n' %% pdb.gimp_image_get_resolution(img)[0])
for i in range(len(img.layers)):
    dup = pdb.gimp_image_duplicate(img)
    for j, l2 in enumerate(dup.layers):
        pdb.gimp_item_set_visible(l2, j == i)
    pdb.gimp_image_flatten(dup)
    d = dup.layers[0]
    w, h = d.width, d.height
    def white(x, y):
        return min(pdb.gimp_drawable_get_pixel(d, x, y)[1][:3]) > 240
    def inner(coords):
        it = iter(coords)
        for c in it:
            if not white(*c[1]):
                break
        else:
            return None
        for c in it:
            if white(*c[1]):
                return c[0]
        return None
    cy, cx = h // 2, w // 2
    L = inner([(x, (x, cy)) for x in range(w)])
    R = inner([(x, (x, cy)) for x in range(w - 1, -1, -1)])
    T = inner([(y, (cx, y)) for y in range(h)])
    B = inner([(y, (cx, y)) for y in range(h - 1, -1, -1)])
    if None not in (L, R, T, B):
        sys.stdout.write('BOX %%d %%d %%d %%d\n' %% (L, T, R, B))
    sys.stdout.flush()
    pdb.gimp_image_delete(dup)
"""


def template_geometry() -> dict:
    """Canvas size and guide boxes measured from template.xcf (cached; the
    probe reruns whenever the template file changes)."""
    import json

    stat = TEMPLATE.stat()
    key = [stat.st_mtime, stat.st_size]
    if GEOMETRY_CACHE.is_file():
        cached = json.loads(GEOMETRY_CACHE.read_text())
        if cached.get("key") == key:
            return cached
    print("Measuring template geometry (template changed or first run)...")
    proc = subprocess.run(
        [GIMP, "-i", "-d", "-f", "--batch-interpreter", "python-fu-eval",
         "-b", GEOMETRY_PROBE % str(TEMPLATE), "-b", "pdb.gimp_quit(0)"],
        capture_output=True, text=True)
    canvas, boxes, res = None, [], 300.0
    for line in proc.stdout.splitlines():
        parts = line.split()
        if parts[:1] == ["CANVAS"]:
            canvas = [int(parts[1]), int(parts[2])]
        elif parts[:1] == ["RES"]:
            res = float(parts[1])
        elif parts[:1] == ["BOX"]:
            boxes.append([int(v) for v in parts[1:5]])
    if not canvas:
        raise RuntimeError("template geometry probe failed:\n"
                           + (proc.stdout + proc.stderr)[-500:])
    geo = {"key": key, "canvas": canvas, "boxes": boxes, "res": res}
    GEOMETRY_CACHE.write_text(json.dumps(geo))
    return geo


def canvas_size() -> tuple:
    return tuple(template_geometry()["canvas"])


def story_box() -> tuple:
    """The box numbered story pages' (cropped) content must fit, symmetric so
    plain canvas-centering is exact: horizontally the canvas minus half the
    safety band per side; vertically the canvas minus the full bottom safety
    band on both sides — the bottom one doubles as the page-number strip."""
    geo = template_geometry()
    cw, ch = geo["canvas"]
    if not geo["boxes"]:
        return cw, ch
    l, t, r, b = min(geo["boxes"], key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]))
    margin_x = (l + (cw - 1 - r)) // 4          # half a safety band per side
    # 1.5x the bottom safety band: room for the page number to sit clearly
    # above the trim line
    margin_y = round((ch - 1 - b) * 1.5)
    return cw - 2 * margin_x, ch - 2 * margin_y


def safety_bottom_y() -> int:
    """Bottom edge of the safety area = top of the page-number strip."""
    geo = template_geometry()
    if not geo["boxes"]:
        return geo["canvas"][1]
    return min(b[3] for b in geo["boxes"])


def strip_bottom_y() -> int:
    """Bottom of the page-number strip: the outermost guide's inner bottom
    edge (the trim line)."""
    geo = template_geometry()
    if not geo["boxes"]:
        return geo["canvas"][1]
    return max(b[3] for b in geo["boxes"])


def cover_box() -> tuple:
    """The box cover images must fit in: the outermost guide layer's inner
    box (the trim border), inset again by that guide band's own width so the
    art clearly does not touch the border. Falls back to the canvas when the
    template has no guide layers."""
    geo = template_geometry()
    cw, ch = geo["canvas"]
    if not geo["boxes"]:
        return cw, ch
    l, t, r, b = max(geo["boxes"], key=lambda bx: (bx[2] - bx[0]) * (bx[3] - bx[1]))
    band_l, band_t = l, t
    band_r, band_b = cw - 1 - r, ch - 1 - b
    return (r - l + 1) - band_l - band_r, (b - t + 1) - band_t - band_b
# a horizontal image is a double-page spread (split into two pages) only if it
# is wide enough that each half makes a sensible portrait page: two comic
# pages side by side give w/h >= ~1.3, while single landscape/square art
# (concept pages etc.) stays well below
SPREAD_MIN_RATIO = 1.25


def should_split(path: Path) -> bool:
    """True if a horizontal image should be split into two pages."""
    with Image.open(path) as im:
        return im.width > im.height and im.width / im.height >= SPREAD_MIN_RATIO


def fit_scale(w: int, h: int, box: tuple = None) -> float:
    """Scale factor at which an image exactly fits the given box (default:
    the template canvas), i.e. the size GIMP will place it at."""
    bw, bh = box or canvas_size()
    return min(bw / w, bh / h)


def image_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def is_vertical(path: Path) -> bool:
    with Image.open(path) as im:
        return im.height > im.width


def megapixels(path: Path) -> float:
    with Image.open(path) as im:
        return im.width * im.height / 1_000_000


def classify(comic_dir: Path, engine: str = "realesrgan"):
    """Split the comic's images into (processable, too_big, horizontal).
    The local engine processes everything: oversized images are downscaled to
    placement size and horizontal spreads are split into two pages. iloveimg
    has a size limit and no splitting, so those images are only reported."""
    processable, too_big, horizontal = [], [], []
    for p in image_files(comic_dir):
        if not is_vertical(p):
            horizontal.append(p)
            if engine != "iloveimg":
                processable.append(p)
        elif (engine == "iloveimg"
                and megapixels(p) * 1_000_000 >= MAX_UPSCALE_PIXELS):
            too_big.append(p)
        else:
            processable.append(p)
    return processable, too_big, horizontal


def report_scan(comic_dir: Path, out_dir: Path,
                engine: str = "realesrgan") -> list[Path]:
    processable, too_big, horizontal = classify(comic_dir, engine)
    split_note = ("skipped" if engine == "iloveimg"
                  else "auto: spreads split into -1/-2, other art kept whole")
    print(f"{comic_dir.name}: {len(processable)} to process, "
          f"{len(too_big)} too big (>=6MP), "
          f"{len(horizontal)} horizontal ({split_note})")
    if engine == "iloveimg":
        for p in horizontal:
            print(f"  HORIZONTAL (skipped): {p.name}")
    if engine != "iloveimg":
        stale = out_dir / "TOO_BIG.txt"
        stale.unlink(missing_ok=True)
    if too_big:
        print("\nToo big for iloveimg upscaler (>=6MP) — decide manually for each:")
        for p in too_big:
            print(f"  TOO BIG: {p.name} ({megapixels(p):.1f}MP)")
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "TOO_BIG.txt"
        report.write_text("".join(f"{p.name}\t{megapixels(p):.1f}MP\n" for p in too_big))
        print(f"  (list written to {report})")
        print(f"  To include one manually, put your chosen version in {out_dir}/ "
              f"and rerun 'just compose'.")
    return processable


FETCH_BLOB_JS = """async (url) => {
    const r = await fetch(url);
    const b = await r.blob();
    return await new Promise(res => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result.split(',')[1]);
        fr.readAsDataURL(b);
    });
}"""


def ext_for(data: bytes, fallback: str) -> str:
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    return fallback


def upscale_one(page, src: Path, out_dir: Path) -> Path:
    """Run one iloveimg upscale task (x2 is the default multiplier) and save
    the result. The file is taken from the result preview (the
    img-comparison-slider), not from the download flow, which the site cuts
    off after a couple of uses."""
    import base64

    page.goto(UPSCALE_URL, timeout=60_000)
    page.wait_for_selector("input[type=file]", state="attached", timeout=30_000)
    # dismiss the cookie banner if it shows up (it can cover the process button)
    accept = page.query_selector("#okck") or page.query_selector("button:has-text('ACCEPT ALL')")
    if accept:
        try:
            accept.click(timeout=2_000)
        except Exception:
            pass
    page.set_input_files("input[type=file]", str(src))
    page.wait_for_selector("#processTask", state="visible", timeout=60_000)
    # make sure the x2 multiplier is the one selected
    x2 = page.query_selector("#upscaleimage-options :text-is('2x')")
    if x2:
        try:
            x2.click(timeout=2_000)
        except Exception:
            pass
    page.click("#processTask")
    page.wait_for_selector(".img-comparison-slider__second img", timeout=300_000)
    url = page.get_attribute(".img-comparison-slider__second img", "src")
    if not url:
        raise RuntimeError("result preview has no src")
    if url.startswith(("blob:", "data:")):
        data = base64.b64decode(page.evaluate(FETCH_BLOB_JS, url))
    else:
        resp = page.request.get(url)
        if resp.status != 200:
            raise RuntimeError(f"fetching result gave HTTP {resp.status}")
        data = resp.body()
    dest = out_dir / (src.stem + ext_for(data, src.suffix.lower()))
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.rename(dest)
    return dest


def transfer_color(img: Image.Image, reference: Image.Image) -> Image.Image:
    """Pin the image's local color/tone to the reference while keeping the
    model's high-frequency detail: result = lowpass(reference) +
    (img - lowpass(img)). Real-ESRGAN drifts tone locally (e.g. it smooths a
    character's shading away, leaving them lighter/desaturated); taking every
    low-frequency component from the original makes such drift impossible,
    while lines and lettering stay as sharp as the model rendered them."""
    import numpy as np
    from PIL import ImageFilter

    blur = ImageFilter.GaussianBlur(4)
    alpha = img.getchannel("A") if "A" in img.getbands() else None
    rgb = img.convert("RGB")
    ref = reference.convert("RGB").resize(rgb.size, Image.LANCZOS)

    a = np.asarray(rgb, dtype=np.int16)
    out = (np.asarray(ref.filter(blur), dtype=np.int16)
           + a - np.asarray(rgb.filter(blur), dtype=np.int16))
    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    if alpha is not None:
        result.putalpha(alpha)
    return result


def save_page(result: Image.Image, src: Path, out_dir: Path) -> Path:
    """Save a processed page next to its siblings, keeping jpg/png kind."""
    if src.suffix.lower() in (".jpg", ".jpeg"):
        dest = out_dir / (src.stem + ".jpg")
        # subsampling=0 (4:4:4): default 4:2:0 chroma subsampling visibly
        # washes out the colors of line art
        fmt, save_kwargs = "JPEG", {"quality": 95, "subsampling": 0}
        result = result.convert("RGB")
    else:
        dest = out_dir / (src.stem + ".png")
        fmt, save_kwargs = "PNG", {}
    tmp = dest.with_suffix(dest.suffix + ".part")
    result.save(tmp, format=fmt, **save_kwargs)
    tmp.rename(dest)
    return dest


def ai_scale(img: Image.Image, target: tuple, scale: float) -> Image.Image:
    """Return img scaled to exactly target: Lanczos only when shrinking
    (nothing to gain from AI), otherwise Real-ESRGAN at its smallest covering
    native scale, Lanczos down to target, and local color pinned to the
    input via transfer_color."""
    import tempfile

    if scale <= 1:
        return img.resize(target, Image.LANCZOS)
    if scale > 4:
        print(f"  WARNING: needs {scale:.2f}x, model gives 4x; "
              f"the rest is plain resampling")
    model_scale = 2 if scale <= 2 else 3 if scale <= 3 else 4

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_in:
        tmp_in = Path(tf_in.name)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf_out:
        tmp_out = Path(tf_out.name)
    try:
        img.save(tmp_in, format="PNG")
        proc = subprocess.run(
            [str(REALESRGAN), "-i", str(tmp_in), "-o", str(tmp_out),
             "-n", REALESRGAN_MODEL, "-s", str(model_scale),
             "-m", str(REALESRGAN.parent / "models")],
            capture_output=True, text=True)
        if proc.returncode != 0 or not tmp_out.stat().st_size:
            raise RuntimeError("realesrgan failed: "
                               + (proc.stderr.strip() + proc.stdout.strip())[-300:])
        with Image.open(tmp_out) as big:
            result = big.resize(target, Image.LANCZOS)
        return transfer_color(result, img)
    finally:
        tmp_in.unlink(missing_ok=True)
        tmp_out.unlink(missing_ok=True)


def crop_to_content(im: Image.Image, tolerance: int = 26,
                    min_frac: float = 0.002) -> Image.Image:
    """Crop away the page's own margins: everything outside the bounding box
    of non-paper content (paper = border median color). The background layer
    recreates the margins on the canvas, so nothing is lost visually."""
    import numpy as np

    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    h, w = a.shape[:2]
    ring = max(2, round(min(w, h) * 0.02))
    border = np.concatenate([a[:ring].reshape(-1, 3), a[-ring:].reshape(-1, 3),
                             a[:, :ring].reshape(-1, 3), a[:, -ring:].reshape(-1, 3)])
    paper = np.median(border, axis=0)
    mask = drop_edge_stripes(np.abs(a - paper).max(axis=2) > tolerance,
                             min_frac)
    rows = np.flatnonzero(mask.mean(axis=1) > min_frac)
    cols = np.flatnonzero(mask.mean(axis=0) > min_frac)
    if not len(rows) or not len(cols):
        return im
    return im.crop((int(cols[0]), int(rows[0]),
                    int(cols[-1]) + 1, int(rows[-1]) + 1))


def drop_edge_stripes(mask, min_frac: float = 0.002):
    """The content mask with the scan's edge stripes cleared.

    A stripe is the gutter shadow or the scanner lid's edge: a run of
    content columns (or rows) at most 1% of the page wide, touching the
    scan's own edge, a real gap of paper away from the art. Kept, it holds
    the crop open twice over — on its own axis, out to the edge, and on the
    OTHER axis over every row it runs down (book 3 p205: 8px of grey at
    x=0 down the top 555 rows held both the left and the top margin, the
    art printed 4% smaller). Clearing one stripe can expose another on the
    other axis, so repeat until none is left. Art that bleeds to the edge
    is no stripe: it is wider than 1%, or no gap separates it."""
    import numpy as np

    mask = mask.copy()
    for _ in range(4):
        found = False
        for axis in (0, 1):                  # 0: columns, 1: rows
            n = mask.shape[1 - axis]
            idx = np.flatnonzero(mask.mean(axis=axis) > min_frac)
            if not len(idx):
                return mask
            runs = np.split(idx, np.flatnonzero(
                np.diff(idx) > max(5, round(0.01 * n))) + 1)
            width = max(3, round(0.01 * n))
            stripes = []
            if len(runs) > 1 and runs[0][0] == 0 and len(runs[0]) <= width:
                stripes.append(slice(0, int(runs[0][-1]) + 1))
            if len(runs) > 1 and runs[-1][-1] == n - 1 and len(runs[-1]) <= width:
                stripes.append(slice(int(runs[-1][0]), n))
            for s in stripes:
                if axis == 0:
                    mask[:, s] = False
                else:
                    mask[s] = False
            found = found or bool(stripes)
        if not found:
            break
    return mask


def trim_folio_band(cropped: Image.Image, page: Image.Image,
                    tolerance: int = 26, min_frac: float = 0.002,
                    speck_rows: float = 2.0) -> Image.Image:
    """The content crop of a page whose folio was erased, with the band the
    folio sat in trimmed off.

    crop_to_content keeps every row that holds a few px of non-paper, and on
    a page whose folio was erased what is left in that band is scan-edge dust:
    a 4px speck on the last row, a smudge at x=0. It held the crop open down
    to the bottom of the scan, so the page kept a strip of empty paper 50-90px
    tall and its art printed several % smaller than its neighbours'. Below a
    real gap, a run of rows holding less ink than `speck_rows` full lines is
    dust, and so are empty rows. Bottom edge only, folio pages only: that is
    where the folio was, and the rest of every page is as approved."""
    import numpy as np
    a = np.asarray(cropped.convert("RGB"), dtype=np.int16)
    h, w = a.shape[:2]
    b = np.asarray(page.convert("RGB"), dtype=np.int16)
    ring = max(2, round(min(b.shape[:2]) * 0.02))
    paper = np.median(np.concatenate(
        [b[:ring].reshape(-1, 3), b[-ring:].reshape(-1, 3),
         b[:, :ring].reshape(-1, 3), b[:, -ring:].reshape(-1, 3)]), axis=0)
    mask = np.abs(a - paper).max(axis=2) > tolerance
    idx = np.flatnonzero(mask.mean(axis=1) > min_frac)
    if not len(idx):
        return cropped
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > max(5, round(0.01 * h)))
                    + 1)
    while len(runs) > 1 and (mask[runs[-1][0]:runs[-1][-1] + 1].sum()
                             < speck_rows * w):
        runs.pop()
    bottom = int(runs[-1][-1]) + 1
    # no early return when the bottom is already tight: the sides below need
    # checking all the same (crop_to_content's edge-stripe rule can have
    # tightened the bottom first, leaving a speck in from the edge)
    # the same dust held the SIDES open too — a smudge at x=0 in the band,
    # or the dark edge of the scan itself: a stripe a few px wide down the
    # page's edge, a real gap away from the art. Re-measure the columns on
    # what is left and drop such an edge run.
    cols = np.flatnonzero(mask[:bottom].mean(axis=0) > min_frac)
    if not len(cols):
        return cropped.crop((0, 0, w, bottom))
    cruns = np.split(cols, np.flatnonzero(np.diff(cols) > max(5, round(0.01 * w)))
                     + 1)
    edge = max(3, round(0.01 * w))
    if len(cruns) > 1 and cruns[0][0] == 0 and len(cruns[0]) <= edge:
        cruns.pop(0)
    if len(cruns) > 1 and cruns[-1][-1] == w - 1 and len(cruns[-1]) <= edge:
        cruns.pop()
    return cropped.crop((int(cruns[0][0]), 0, int(cruns[-1][-1]) + 1, bottom))


def paper_color(src: Path) -> tuple:
    """The page's paper color: median of the source image's border ring
    (measured pre-crop — the cropped content's border is art, not paper).
    When the border is not uniform (full-bleed art: no real margins or frame
    anywhere), there is no paper color to match — fall back to white, the
    book's paper, instead of an arbitrary average of art edges."""
    import numpy as np

    im = Image.open(src)
    im.draft("RGB", (400, 800))
    im = im.convert("RGB")
    a = np.asarray(im, dtype=np.int16)
    h, w = a.shape[:2]
    ring = max(2, round(min(w, h) * 0.02))
    border = np.concatenate([a[:ring].reshape(-1, 3), a[-ring:].reshape(-1, 3),
                             a[:, :ring].reshape(-1, 3), a[:, -ring:].reshape(-1, 3)])
    median = np.median(border, axis=0)
    if (np.abs(border - median).max(axis=1) <= 26).mean() < 0.7:
        return (255, 255, 255)
    return tuple(int(v) for v in median)


def pick_split_x(im: Image.Image, window: float = 0.05,
                 tolerance: int = 24) -> int:
    """Choose where to split a spread. A real panel gutter is a column of
    paper-colored pixels running vertically for a long stretch; within
    +/-window of the center, find such columns and split at the middle of the
    gutter band nearest the center. Cutting through borderless art (sky,
    backgrounds) is fine, but a cut near a panel edge leaves a sliver of that
    panel on the wrong page — so only real gutters move the split; with no
    gutter in the window, split exactly in half."""
    import numpy as np

    rgb = np.asarray(im.convert("RGB"), dtype=np.int16)
    h, w = rgb.shape[:2]
    center = w // 2
    half = max(2, int(w * window))

    # paper color = median of the outer border ring (same idea as the
    # background layer in the compose plugin)
    ring = max(2, round(min(w, h) * 0.02))
    border = np.concatenate([rgb[:ring].reshape(-1, 3), rgb[-ring:].reshape(-1, 3),
                             rgb[:, :ring].reshape(-1, 3), rgb[:, -ring:].reshape(-1, 3)])
    paper = np.median(border, axis=0)

    cols = (np.abs(rgb[:, center - half:center + half] - paper).max(axis=2)
            <= tolerance)
    # longest contiguous vertical paper run per column
    runs = np.zeros(cols.shape[1], dtype=int)
    current = np.zeros(cols.shape[1], dtype=int)
    for row in cols:
        current = (current + 1) * row
        runs = np.maximum(runs, current)

    best = runs.max()
    if best < h * 0.1:  # no meaningful gutter in the window: exact half
        return center
    good = np.flatnonzero(runs >= best * 0.8)
    groups = np.split(good, np.where(np.diff(good) > 1)[0] + 1)
    gutter = min(groups, key=lambda g: abs((g[0] + g[-1]) / 2 - half))
    return int(center - half + round((gutter[0] + gutter[-1]) / 2))


# --- the source's own printed page numbers (folios) -------------------------
# Many sources carry their original edition's page number at the bottom
# centre (under each half of a spread). The book draws its own, so the
# original is erased before the crop — which also stops it stretching the
# content box. Detection keys on the ISSUE, never on one page: a folio is a
# group of 1-3 digit-shaped marks, and an issue counts as numbered only when
# many of its pages agree on the marks' height and position (stray specks,
# scan dust and lettering never do). The printed value restarts per issue and
# is not the file number, so it is never predicted — only position and look.
FOLIO_TOP = 0.92            # folios sit below this fraction of page height
FOLIO_H = (0.007, 0.018)    # digit height range, fraction of page height
FOLIO_CONTRAST = 60         # ink vs local background (luma)
FOLIO_VOTE = (3, 0.2)       # pages that must agree: at least N and this share
FOLIO_CACHE = "folios.json"
FOLIO_VERSION = 1


def _label(mask):
    """8-connected components of a bool mask as (ys, xs) arrays — labelled
    on row runs with union-find (no scipy in the base venv)."""
    import numpy as np
    h, w = mask.shape
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = mask
    d = np.diff(padded, axis=1)
    sy, sx = np.nonzero(d == 1)
    ex = np.nonzero(d == -1)[1]
    parent = list(range(len(sy)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    row = np.searchsorted(sy, np.arange(h + 1))
    for y in range(1, h):
        i, j, a1, b1 = row[y - 1], row[y], row[y], row[y + 1]
        while i < a1 and j < b1:
            if sx[i] <= ex[j] and sx[j] <= ex[i]:   # touch incl. diagonals
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
            if ex[i] < ex[j]:
                i += 1
            else:
                j += 1
    groups = {}
    for k in range(len(sy)):
        groups.setdefault(find(k), []).append(k)
    return [(np.concatenate([np.full(ex[k] - sx[k], sy[k]) for k in ks]),
             np.concatenate([np.arange(sx[k], ex[k]) for k in ks]))
            for ks in groups.values()]


def _dilate(mask, r: int):
    out = mask.copy()
    for _ in range(r):
        m = out.copy()
        m[1:] |= out[:-1]
        m[:-1] |= out[1:]
        m[:, 1:] |= out[:, :-1]
        m[:, :-1] |= out[:, 1:]
        out = m
    return out


def _luma(rgb):
    import numpy as np
    return (rgb.astype(np.int32) @ np.array([299, 587, 114]) // 1000
            ).astype(np.int16)


def _folio_ink(rgb, page_h: int):
    """(luma, {"dark": mask, "light": mask}) for a region: ink = pixels
    well off a local median background, which is flat paper, a flat band,
    or the artwork itself."""
    import numpy as np
    from PIL import ImageFilter
    luma = _luma(rgb)
    k = min(31, int(FOLIO_H[1] * page_h) | 1)
    bg = np.asarray(Image.fromarray(luma.astype(np.uint8)).filter(
        ImageFilter.MedianFilter(k)), dtype=np.int16)
    return luma, {"dark": luma - bg < -FOLIO_CONTRAST,
                  "light": luma - bg > FOLIO_CONTRAST,
                  "any": np.abs(luma - bg) > FOLIO_CONTRAST}


def _digit_groups(rgb, page_h: int, strict: bool) -> list:
    """Groups of 1-3 digit-shaped marks in one search window (region-
    relative boxes). strict: alone on a plain surround — the evidence an
    issue's folio band is built from. Relaxed: may sit on artwork, but must
    be the flat near-black / near-white ink a folio is printed in."""
    import numpy as np
    luma, inks = _folio_ink(rgb, page_h)
    lo, hi = FOLIO_H[0] * page_h, FOLIO_H[1] * page_h
    rh, rw = rgb.shape[:2]
    out = []
    for pol in ("dark", "light"):
        ink = inks[pol]
        digits = []
        for ys, xs in _label(ink):
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            ch, cw = y1 - y0, x1 - x0
            fill = len(ys) / (ch * cw)
            # a "1" is a bar: solid fill is fine when the mark is that thin
            if (lo <= ch <= hi and 0.12 * ch <= cw <= ch
                    and 0.15 <= fill <= (1.0 if cw <= 0.3 * ch else 0.9)
                    and x0 > 0 and x1 < rw and y1 < rh):
                digits.append({"box": (x0, y0, x1, y1), "h": ch, "fill": fill,
                               "lum": float(np.median(luma[ys, xs])),
                               "ys": ys, "xs": xs})
        digits.sort(key=lambda c: c["box"][0])
        used = set()
        for i, c in enumerate(digits):
            if i in used:
                continue
            grp = [i]
            for j in range(i + 1, len(digits)):
                d, last = digits[j], digits[grp[-1]]
                if (abs(d["h"] - c["h"]) <= 0.2 * c["h"]
                        and abs(d["box"][1] + d["box"][3]
                                - c["box"][1] - c["box"][3]) <= 0.5 * c["h"]
                        and d["box"][0] - last["box"][2] <= 0.6 * c["h"]):
                    grp.append(j)
            used.update(grp)
            if len(grp) > 3:
                continue        # a line of lettering, not a page number
            members = [digits[g] for g in grp]
            x0 = min(m["box"][0] for m in members)
            y0 = min(m["box"][1] for m in members)
            x1 = max(m["box"][2] for m in members)
            y1 = max(m["box"][3] for m in members)
            gh = y1 - y0
            tone = float(np.median(np.concatenate(
                [luma[m["ys"], m["xs"]] for m in members])))
            # complete the number: a grey digit's pale strokes break off,
            # and a digit touching art of its own colour never comes back as
            # a digit-shaped piece — take in same-tone ink in the number's
            # rows, up to a digit width either side, while it stays contiguous
            cols = (ink & (np.abs(luma - tone) <= 40))[y0:y1].any(axis=0)
            reach, gap = int(round(0.65 * gh)), int(round(0.3 * gh))
            nx0 = x0
            while nx0 > max(0, x0 - reach):
                seg = cols[max(0, nx0 - gap):nx0]
                if not seg.any():
                    break
                nx0 = max(0, nx0 - gap) + int(np.argmax(seg))
            nx1 = x1
            while nx1 < min(rw, x1 + reach):
                seg = cols[nx1:nx1 + gap]
                if not seg.any():
                    break
                nx1 += len(seg) - int(np.argmax(seg[::-1]))
            x0, x1 = nx0, nx1
            mine = np.zeros((rh, rw), bool)
            mine[y0:y1, x0:x1] = ink[y0:y1, x0:x1]
            # a number stands alone on its line: marks continuing it sideways
            # make it lettering (a panel edge just above is fine)
            other = inks["any"].copy()
            other[y0:y1, x0:x1] = False
            side = int(round(0.6 * gh))
            stray = int(other[y0:y1, max(0, x0 - side):x1 + side].sum())
            pad = int(round(0.35 * gh))
            ring = np.zeros((rh, rw), bool)
            ring[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad] = True
            ring = rgb[ring & ~_dilate(mine, 2)]
            # share of the surround at its own colour: a plain margin or band
            # even with a bit of art reaching into one edge
            plain = (float((np.abs(ring - np.median(ring, axis=0)).max(axis=1)
                            <= 24).mean()) if len(ring) else 0.0)
            if strict:
                if stray > 0.05 * gh * gh or plain < 0.85:
                    continue
                if pol == "light" and tone < 160:
                    continue    # light folios are printed white
            elif not all((m["lum"] <= 70 if pol == "dark" else m["lum"] >= 200)
                         and m["fill"] >= 0.25 for m in members):
                continue
            out.append({"box": (x0, y0, x1, y1), "pol": pol})
    return out


def folio_candidates(im: Image.Image, strict: bool) -> list:
    """Digit groups in a page's folio windows, in page coordinates, with the
    window's expected centre (want: 0.5 for a page, 0.25/0.75 under each half
    of a spread) and the group's position as page fractions."""
    import numpy as np
    a = np.asarray(im.convert("RGB"), dtype=np.int16)
    h, w = a.shape[:2]
    top = int(h * FOLIO_TOP)
    windows = ([(0.25, 0.19, 0.31), (0.75, 0.69, 0.81)]
               if w / h >= SPREAD_MIN_RATIO else [(0.5, 0.40, 0.60)])
    found = []
    for want, f0, f1 in windows:
        x0 = int(w * f0)
        for g in _digit_groups(a[top:, x0:int(w * f1)], h, strict):
            bx0, by0, bx1, by1 = g["box"]
            box = [x0 + bx0, top + by0, x0 + bx1, top + by1]
            found.append({"box": [int(v) for v in box], "pol": g["pol"],
                          "want": want, "cx": (box[0] + box[2]) / 2 / w,
                          "cy": (box[1] + box[3]) / 2 / h,
                          "hf": (box[3] - box[1]) / h})
    return found


def _folio_issue(stem: str) -> str:
    """The issue a page belongs to: its name minus the trailing page number
    (and any tag after it, e.g. 'cover')."""
    import re
    return re.sub(r"\s+\d+[a-z]*(\s+\D*)?$", "", stem)


def _folio_parity(stem: str):
    import re
    m = re.search(r"(\d+)[a-z]*(\s+\D*)?$", stem)
    return int(m.group(1)) % 2 if m else None


def _folio_pick(cands, band, want=None, cx_tol=0.04, cy_tol=0.02,
                hf_tol=0.25) -> list:
    """At most one folio per window: the band match nearest the expected x
    (want overrides the window centre)."""
    out = {}
    for g in cands:
        x = (want or {}).get(g["want"], g["want"])
        if (abs(g["hf"] - band[1]) <= hf_tol * band[1]
                and abs(g["cy"] - band[0]) <= cy_tol
                and abs(g["cx"] - x) <= cx_tol):
            k = g["want"]
            if k not in out or abs(g["cx"] - x) < abs(out[k]["cx"] - x):
                out[k] = g
    return list(out.values())


def find_folios(pages: list) -> tuple:
    """({filename: [{"box", "pol", "art"}]}, report lines) for one comic's
    story pages. Per issue: strict candidates vote a band (height, position);
    every page then gets its band match — strict first, else relaxed (on
    art) at the side the issue prints that page parity's number."""
    import numpy as np
    from collections import defaultdict
    issues = defaultdict(list)
    for p in pages:
        issues[_folio_issue(p.stem)].append(p)
    plan, report = {}, []
    for issue, files in sorted(issues.items()):
        strict = {}
        for p in files:
            with Image.open(p) as im:
                strict[p] = folio_candidates(im, True)
        # the vote is tighter than the per-page match: a real folio band
        # agrees within a few % of height and ~1% of position
        allc = [g for cs in strict.values() for g in cs
                if abs(g["cx"] - g["want"]) <= 0.03]
        best, best_pages = [], 0
        for g in allc:
            agree = [o for o in allc if abs(o["hf"] - g["hf"]) <= 0.1 * g["hf"]
                     and abs(o["cy"] - g["cy"]) <= 0.012]
            ids = {id(o) for o in agree}
            n = sum(1 for cs in strict.values() if any(id(o) in ids for o in cs))
            if n > best_pages:
                best, best_pages = agree, n
        if best_pages < max(FOLIO_VOTE[0], FOLIO_VOTE[1] * len(files)):
            continue
        band = (float(np.median([g["cy"] for g in best])),
                float(np.median([g["hf"] for g in best])))
        picked = {p: _folio_pick(strict[p], band) for p in files}
        sides = defaultdict(list)
        for p, gs in picked.items():
            sides[_folio_parity(p.stem)] += [g["cx"] for g in gs
                                             if g["want"] == 0.5]
        want = {k: {0.5: float(np.median(v))} for k, v in sides.items()
                if len(v) >= 3}
        on_art = []
        for p in files:
            gs, art = picked[p], False
            if not gs:
                with Image.open(p) as im:
                    gs = _folio_pick(folio_candidates(im, False), band,
                                     want.get(_folio_parity(p.stem)),
                                     cx_tol=0.015, hf_tol=0.08)
                art = bool(gs)
            if gs:
                plan[p.name] = [{"box": g["box"], "pol": g["pol"], "art": art}
                                for g in gs]
                if art:
                    on_art.append(p.stem)
        n = sum(1 for p in files if p.name in plan)
        report.append(f"  {issue}: {n} of {len(files)} pages"
                      + (f" ({len(on_art)} over art)" if on_art else ""))
    return plan, report


def _inpaint(a, mask, iters: int = 400):
    """Harmonic (Laplace) fill of mask from its surroundings, solved
    coarse-to-fine so a patch a few dozen px wide converges in a few hundred
    sweeps."""
    import numpy as np
    h, w = mask.shape
    if min(h, w) >= 16:
        hh, ww = h // 2 * 2, w // 2 * 2
        known = ~mask[:hh, :ww].reshape(h // 2, 2, w // 2, 2)
        cnt = known.sum(axis=(1, 3))
        blocks = a[:hh, :ww].reshape(h // 2, 2, w // 2, 2, 3)
        coarse = ((blocks * known[..., None]).sum(axis=(1, 3))
                  / np.maximum(cnt, 1)[..., None])
        coarse = _inpaint(coarse, cnt == 0, iters)
        full = np.empty_like(a)
        full[:] = a[~mask].mean(axis=0)
        full[:hh, :ww] = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
        a = np.where(mask[..., None], full, a)
    else:
        a = np.where(mask[..., None], a[~mask].mean(axis=0), a)
    for _ in range(iters):
        p = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
        avg = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4
        a = np.where(mask[..., None], avg, a)
    return a


def erase_folios(im: Image.Image, folios: list) -> Image.Image:
    """The page with its folios erased. On a plain margin or band the digits
    and their halo (anti-aliasing, JPEG ringing) take the surround's colour,
    keeping anything far from it (a panel border, line art); over artwork
    the patch is filled in from its edges."""
    import numpy as np
    from PIL import ImageFilter
    if not folios:
        return im
    a = np.asarray(im.convert("RGB")).astype(np.float64)
    h, w = a.shape[:2]
    for f in folios:
        bx0, by0, bx1, by1 = f["box"]
        gh = by1 - by0
        pad = max(6, int(round(0.3 * gh)))
        # the ink mask, re-derived from the box like detection derived it
        # (a background window wide enough for the median to see past it)
        m = 40
        rx0, ry0 = max(0, bx0 - m), max(0, by0 - m)
        rx1, ry1 = min(w, bx1 + m), min(h, by1 + m)
        _, inks = _folio_ink(a[ry0:ry1, rx0:rx1].astype(np.int16), h)
        digits = np.zeros((h, w), bool)
        digits[by0:by1, bx0:bx1] = inks[f["pol"]][by0 - ry0:by1 - ry0,
                                                  bx0 - rx0:bx1 - rx0]
        y0, y1 = max(0, by0 - pad), min(h, by1 + pad)
        x0, x1 = max(0, bx0 - pad), min(w, bx1 + pad)
        digits = digits[y0:y1, x0:x1]
        patch = a[y0:y1, x0:x1]
        # what directly surrounds the number decides plain vs art; a border
        # or art a few px further out must not
        ring = patch[_dilate(digits, 6) & ~_dilate(digits, 3)]
        med = np.median(ring, axis=0)
        if (np.abs(ring - med).max(axis=1) <= 36).mean() >= 0.85:
            # the surround's colour, plus the paper's own drift (grain, a
            # gradient) interpolated in from the patch edge — a flat median
            # shows as a faint square; things far from it (a border) count
            # as plain paper so they cannot bleed in
            near = np.abs(patch - med).max(axis=2) <= 80
            fill = _dilate(digits, 2) | (near & _dilate(digits, 5))
            drift = np.where(near[..., None], patch - med, 0.0)
            patch[fill] = (med + _inpaint(drift, fill))[fill]
        else:
            # the halo is whatever near the number still differs from the
            # local background — left in, it seeds the fill with a ghost
            k = min(31, max(3, int(round(1.2 * gh))) | 1)
            bg = np.asarray(Image.fromarray(np.clip(patch, 0, 255).astype(
                np.uint8)).filter(ImageFilter.MedianFilter(k)), np.float64)
            halo = np.abs(patch - bg).max(axis=2) > 16
            near = np.zeros_like(digits)
            near[max(0, by0 - y0 - 4):by1 - y0 + 4,
                 max(0, bx0 - x0 - 4):bx1 - x0 + 4] = True
            patch[:] = _inpaint(patch, _dilate(digits, 3)
                                | (_dilate(halo, 1) & near))
    return Image.fromarray(np.clip(a + 0.5, 0, 255).astype(np.uint8))


_FOLIO_PLANS = {}


def folio_plan(comic_dir: Path, cache_dir: Path = None) -> dict:
    """{source filename: folios} for a comic, computed once and cached in
    upscaled/<comic>/folios.json (keyed on every story source's size and
    mtime, so adding or replacing a source re-detects)."""
    import json
    comic_dir = Path(comic_dir)
    if comic_dir in _FOLIO_PLANS:
        return _FOLIO_PLANS[comic_dir]
    cache_dir = cache_dir or REPO / "upscaled" / comic_dir.name
    pages = [p for p in image_files(comic_dir) if "cover" not in p.stem.lower()]
    key = {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in pages}
    cache = cache_dir / FOLIO_CACHE
    try:
        cached = json.loads(cache.read_text())
        if cached.get("version") == FOLIO_VERSION and cached["sources"] == key:
            _FOLIO_PLANS[comic_dir] = cached["pages"]
            return cached["pages"]
    except (OSError, ValueError, KeyError):
        pass
    print(f"Looking for the sources' own page numbers ({len(pages)} pages)...",
          flush=True)
    plan, report = find_folios(pages)
    print("\n".join(report) if report else "  none found")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"version": FOLIO_VERSION, "sources": key,
                                 "pages": plan, "report": report},
                                ensure_ascii=False, indent=0))
    _FOLIO_PLANS[comic_dir] = plan
    return plan


def upscale_spec(src: Path) -> dict:
    """How this page will be produced: the (possibly cropped) content image,
    kind ('single' or 'spread'), scale, target size and split point. Shared
    by the upscale step and the resume check so they always agree. Story
    pages are cropped to their content bounding box (their own margins are
    recreated by the background layer) and fit the symmetric story box, which
    reserves the page-number strip; covers fit inside the trim; index pages
    fit the full canvas."""
    img = Image.open(src)
    img.load()
    if "cover" in src.stem.lower():
        box, note, is_cover = cover_box(), ", cover: fit inside trim", True
    elif img.height > img.width and is_index_page(src):
        box, note, is_cover = canvas_size(), ", index: full canvas", False
    else:
        folios = folio_plan(src.parent).get(src.name, [])
        img = erase_folios(img, folios)
        cropped = crop_to_content(img)
        if folios:
            cropped = trim_folio_band(cropped, img)
        note = (f", cropped from {img.width}x{img.height}"
                if cropped.size != img.size else "")
        if folios:
            note += ", original page number erased"
        img = cropped
        box, is_cover = story_box(), False
    w, h = img.size
    if is_cover or h >= w or w / h < SPREAD_MIN_RATIO:
        scale = fit_scale(w, h, box)
        return {"kind": "single", "img": img, "scale": scale, "note": note,
                "target": (round(w * scale), round(h * scale))}
    split_x = pick_split_x(img)
    scale = fit_scale(max(split_x, w - split_x), h, story_box())
    return {"kind": "spread", "img": img, "scale": scale, "note": note,
            "target": (round(w * scale), round(h * scale)), "split_x": split_x}


def upscale_realesrgan(src: Path, out_dir: Path) -> Path:
    """Produce the page(s) for one source image at exactly the size GIMP will
    place them (see upscale_spec). A spread goes through the AI process whole
    (seam consistency), then is split at the detected gutter into <name>-1
    (left) and <name>-2 (right)."""
    spec = upscale_spec(src)
    img, scale, target = spec["img"], spec["scale"], spec["target"]
    result = ai_scale(img, target, scale)
    if spec["kind"] == "single":
        dest = save_page(result, src, out_dir)
        with Image.open(dest) as check:
            if (check.width, check.height) != target:
                raise RuntimeError(f"output is {check.size}, expected {target}")
        print(f"  {img.width}x{img.height} -> {target[0]}x{target[1]} "
              f"({scale:.2f}x{spec['note']})")
        return dest
    split_x = spec["split_x"]
    mid = min(max(1, round(split_x * scale)), target[0] - 1)
    save_page(result.crop((0, 0, mid, target[1])),
              src.with_stem(src.stem + "-1"), out_dir)
    dest = save_page(result.crop((mid, 0, target[0], target[1])),
                     src.with_stem(src.stem + "-2"), out_dir)
    print(f"  spread split at x={split_x} -> {mid}x{target[1]} + "
          f"{target[0] - mid}x{target[1]} ({scale:.2f}x{spec['note']})")
    return dest


def check_x2(src: Path, dest: Path, tolerance: int = 4) -> None:
    with Image.open(src) as a, Image.open(dest) as b:
        if (abs(b.width - a.width * 2) > tolerance
                or abs(b.height - a.height * 2) > tolerance):
            print(f"  WARNING: {dest.name} is {b.width}x{b.height}, "
                  f"expected ~{a.width * 2}x{a.height * 2}")


def cmd_upscale(comic_dir: Path, out_dir: Path, engine: str = "realesrgan",
                headed: bool = False, retries: int = 2,
                cleanup_dirs: tuple = ()) -> int:
    processable = report_scan(comic_dir, out_dir, engine)
    done = {p.stem: p for p in image_files(out_dir)} if out_dir.is_dir() else {}

    def matches(stem: str, height: int, width: int = None) -> bool:
        f = done.get(stem)
        if f is None:
            return False
        with Image.open(f) as im:
            return im.height == height and (width is None or im.width == width)

    def is_done(p: Path) -> bool:
        """Done = output exists AT the currently expected size — a template
        or sizing-rule change invalidates stale outputs automatically."""
        if p.stem not in done and not (
                {p.stem + "-1", p.stem + "-2"} <= set(done)):
            return False  # nothing there: skip the (costly) spec computation
        spec = upscale_spec(p)
        tw, th = spec["target"]
        outs = ([p.stem] if spec["kind"] == "single"
                else [p.stem + "-1", p.stem + "-2"])
        # erasing a folio changes the pixels but often not the size (the
        # crop box only moves when the number stretched it), so a size check
        # alone kept 15 pages of one book with their numbers still printed.
        # An output older than the folio plan that erases something on it
        # predates the erasure.
        cache = out_dir / FOLIO_CACHE
        if (folio_plan(p.parent).get(p.name) and cache.is_file()
                and any(done[s].stat().st_mtime < cache.stat().st_mtime
                        for s in outs if s in done)):
            return False
        if spec["kind"] == "single":
            return matches(p.stem, th, tw)
        return matches(p.stem + "-1", th) and matches(p.stem + "-2", th)

    # if the split decision for a horizontal changed since an earlier run,
    # drop the now-stale outputs (single vs halves) from every output folder
    for p in processable:
        if is_vertical(p):
            continue
        stale_stems = ([p.stem] if should_split(p)
                       else [p.stem + "-1", p.stem + "-2"])
        for d in (out_dir, *cleanup_dirs):
            for stem in stale_stems:
                for stale in Path(d).glob(stem + ".*"):
                    stale.unlink()
                    print(f"  removed stale output: {stale}")

    pending = [p for p in processable if not is_done(p)]
    # once the fit has run, upscaled/ holds the CLEANED working pages, and
    # layout.json, the masks and the positional transcripts are all in that
    # page's geometry: a crop or sizing rule that moves an existing page must
    # not re-upscale it in place. Report it; rebuild it deliberately.
    if (RELETTER_DIR / comic_dir.name / "layout.json").is_file():
        moved = [p for p in pending
                 if p.stem in done or {p.stem + "-1", p.stem + "-2"} & set(done)]
        if moved:
            print(f"\nKEPT {len(moved)} re-lettered page(s) whose expected "
                  "upscale changed — re-upscaling would overwrite the cleaned "
                  "page under a layout in the old geometry. To rebuild one: "
                  "re-upscale, remap its transcripts through the source and "
                  "re-letter it (ARCHITECTURE.md, single-page recipe):")
            for p in moved:
                print(f"  KEPT: {p.name}")
            pending = [p for p in pending if p not in moved]
    already = len(processable) - len(pending)
    if already:
        print(f"{already} already upscaled, {len(pending)} to go")
    if not pending:
        print("Nothing to upscale.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    failures = []

    def loop(run_one, on_retry=None):
        for i, src in enumerate(pending, 1):
            print(f"UPSCALE {i}/{len(pending)} [{engine}]: {src.name}", flush=True)
            for attempt in range(1, retries + 2):
                try:
                    run_one(src)
                    break
                except Exception as e:
                    print(f"  attempt {attempt} failed: {e}")
                    if attempt > retries:
                        failures.append(src.name)
                    else:
                        time.sleep(5)
                        if on_retry:
                            on_retry()

    if engine == "realesrgan":
        if not REALESRGAN.is_file():
            print(f"Real-ESRGAN binary not found: {REALESRGAN}")
            return 1
        loop(lambda src: upscale_realesrgan(src, out_dir))
    else:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not headed)
            context = browser.new_context()
            state = {"page": context.new_page()}

            def fresh_page():
                state["page"] = context.new_page()

            def run_iloveimg(src):
                dest = upscale_one(state["page"], src, out_dir)
                check_x2(src, dest)

            loop(run_iloveimg, fresh_page)
            browser.close()

    if failures:
        print(f"\nFAILED ({len(failures)}): " + ", ".join(failures))
        print("Rerun 'just upscale' to retry the failed ones.")
        return 1
    print("Upscaling complete.")
    return 0


PDF_JPEG_QUALITY = 92


def postprocess_pdf(pdf_path: Path) -> None:
    """Fixes to GIMP's PDF export, in place: (1) recompress its lossless
    Flate images as high-quality JPEG — GIMP stores raw RGB, ~4x the size of
    the JPEG the page was composed from; (2) tag RGB images with an sRGB ICC
    profile so viewers and print RIPs interpret the colors as-is instead of
    guessing; (3) strip transparency declarations (/Group transparency dicts
    and CA/ca=1 graphics states) — the content is fully opaque, but the bare
    declarations trip print-shop preflight checks."""
    import io
    import pikepdf
    from PIL import ImageCms

    icc_bytes = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    with pikepdf.open(pdf_path, allow_overwriting_input=True) as pdf:
        icc_stream = pdf.make_stream(icc_bytes)
        icc_stream.N = 3
        srgb = pdf.make_indirect(pikepdf.Array([pikepdf.Name.ICCBased, icc_stream]))

        def fix_image(xobj):
            cs = xobj.get("/ColorSpace")
            is_rgb = cs == pikepdf.Name.DeviceRGB or (
                isinstance(cs, pikepdf.Array) and cs[0] == pikepdf.Name.ICCBased)
            if not is_rgb:
                return
            if (xobj.get("/Filter") == pikepdf.Name.FlateDecode
                    and xobj.get("/BitsPerComponent") == 8):
                im = pikepdf.PdfImage(xobj).as_pil_image().convert("RGB")
                buf = io.BytesIO()
                im.save(buf, format="JPEG",
                        quality=PDF_JPEG_QUALITY, subsampling=0)
                xobj.write(buf.getvalue(), filter=pikepdf.Name.DCTDecode)
            xobj.ColorSpace = srgb

        def strip_transparency(obj):
            if "/Group" in obj:
                del obj["/Group"]
            # empty the CA/ca (opaque anyway) rather than delete the
            # ExtGState, which content streams still reference by name
            resources = obj.get("/Resources", {})
            for _, gs in (resources.get("/ExtGState", {}) or {}).items():
                for key in ("/CA", "/ca", "/SMask"):
                    if key in gs:
                        del gs[key]

        def walk(resources):
            for _, xobj in resources.get("/XObject", {}).items():
                subtype = xobj.get("/Subtype")
                if subtype == pikepdf.Name.Image:
                    fix_image(xobj)
                elif subtype == pikepdf.Name.Form:
                    # GIMP nests each layer's image inside a Form XObject
                    strip_transparency(xobj)
                    walk(xobj.get("/Resources", {}))

        for page in pdf.pages:
            strip_transparency(page.obj)
            walk(page.obj.get("/Resources", {}))
        pdf.save(pdf_path)


def scm_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def source_of(f: Path, comic_dir: Path):
    """The source image an upscaled page came from (a spread half maps to
    the whole spread), or None."""
    stem = f.stem
    if stem.endswith(("-1", "-2")):
        stem = stem[:-2]
    return next(iter(comic_dir.glob(stem + ".*")), None)


def page_numbers(in_dir: Path, comic_dir: Path) -> tuple:
    """(files, {filename: number}). Page numbers = final book positions
    (blanks included); drawn only on story pages and spread halves — never
    on covers, indexes, art pages, or blanks."""
    import re
    print("Computing book layout for page numbers...")
    files, kinds, plan, _ = book_plan(in_dir)

    def is_art(f: Path) -> bool:
        # art pages get no number: the word 'art' in the filename (marker,
        # like 'cover'), or auto-detected on the SOURCE image (the upscaled
        # file is cropped, which strips the margins the detector needs)
        if re.search(r"\bart\b", f.stem, re.IGNORECASE):
            return True
        src = source_of(f, comic_dir)
        return src is not None and paper_page_kind(src) == "art"

    numbers = {}
    for pos, entry in enumerate(plan, 1):
        if (entry is not None
                and kinds[files.index(entry)] in ("story", "half1")
                and not is_art(entry)):
            numbers[entry.name] = pos
    return files, numbers


def write_numbers_file(files: list, numbers: dict, comic_dir: Path,
                       xcf_dir: Path) -> Path:
    """Per-page metadata for the plugin (xcf_dir/numbers.txt): page number
    ('-' = none) and the paper color measured from the SOURCE image (the
    cropped content's own border is art, so the plugin can no longer
    measure it itself). Header: the number zone runs from the lowest
    possible art edge to the safety line; the strip below it, down to the
    trim, is left empty because the printer's cut can reach into it."""
    lines = [f"ART_BOTTOM {(canvas_size()[1] + story_box()[1]) // 2}",
             f"SAFETY_BOTTOM {safety_bottom_y()}",
             f"STRIP_BOTTOM {strip_bottom_y()}"]
    for f in files:
        num = numbers.get(f.name, "-")
        src = source_of(f, comic_dir)
        rgb = ",".join(map(str, paper_color(src))) if src else "-"
        lines.append(f"{f.name}\t{num}\t{rgb}")
    numbers_file = xcf_dir / "numbers.txt"
    numbers_file.write_text("\n".join(lines) + "\n")
    return numbers_file


def run_gimp_plugin(call: str) -> list:
    """Run one plugin call in GIMP batch mode; returns the COMPOSE/ERROR
    lines. GIMP on macOS often exits non-zero after a successful batch
    run (gimp_wire_write_msg noise), so callers judge success from the
    plugin's own DONE line instead of the return code."""
    # no -f: fonts must load for the page-number text
    cmd = [GIMP, "-i", "-d", "-b", call, "-b", "(gimp-quit 0)"]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    lines = (proc.stdout + proc.stderr).splitlines()
    return [l for l in lines if l.startswith(("COMPOSE", "ERROR"))]


def cmd_compose(in_dir: Path, xcf_dir: Path, pdf_dir: Path,
                comic_dir: Path) -> int:
    if not TEMPLATE.is_file():
        print(f"Template not found: {TEMPLATE}")
        return 1
    if not in_dir.is_dir() or not image_files(in_dir):
        print(f"No upscaled images in {in_dir} — run 'just upscale' first.")
        return 1
    xcf_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    import json
    files, numbers = page_numbers(in_dir, comic_dir)

    # pages whose number changed since the last compose (or that have never
    # been numbered) must be recomposed
    manifest_path = xcf_dir / "numbers.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    stale = [f for f in files if old.get(f.name) != numbers.get(f.name)]
    if stale:
        print(f"{len(stale)} pages need (re)numbering")
        for f in stale:
            (xcf_dir / (f.stem + ".xcf")).unlink(missing_ok=True)
            (pdf_dir / (f.stem + ".pdf")).unlink(missing_ok=True)

    numbers_file = write_numbers_file(files, numbers, comic_dir, xcf_dir)

    call = "(python-fu-webcomics-compose RUN-NONINTERACTIVE {} {} {} {} {})".format(
        scm_quote(str(in_dir)), scm_quote(str(TEMPLATE)),
        scm_quote(str(xcf_dir)), scm_quote(str(pdf_dir)),
        scm_quote(str(numbers_file)))
    print("Running GIMP batch (this can take a while)...", flush=True)
    interesting = run_gimp_plugin(call)
    print("\n".join(interesting))
    done_line = next((l for l in interesting if l.startswith("COMPOSE DONE")), None)
    if done_line is None or "0 failed" not in done_line:
        print("Compose step did not finish cleanly — see output above.")
        return 1
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Post-processing {len(pdfs)} PDFs (JPEG recompress + sRGB tag)...")
    for pdf in pdfs:
        postprocess_pdf(pdf)
    manifest_path.write_text(json.dumps(numbers))
    return 0


# index/credits page detection: mostly paper-colored with dark ink only in
# thin text rows. Validated on this collection: credits pages score paper
# fraction >= 0.82 with ink runs <= 0.03; sketch/extras pages have runs
# >= 0.44; story pages have paper fraction <= 0.67.
INDEX_MIN_PAPER = 0.70
INDEX_MAX_INK_RUN = 0.15


def paper_page_kind(path: Path):
    """Classify mostly-paper pages: 'index' (credits/title: thin text rows
    only), 'art' (extras/sketch pages: tall runs of dark strokes), or None
    for regular pages."""
    import numpy as np

    im = Image.open(path)
    im.draft("RGB", (600, 1200))
    im = im.convert("RGB")
    im.thumbnail((600, 900))
    a = np.asarray(im, dtype=np.int16)
    h, w = a.shape[:2]
    ring = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3),
                           a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    paper = np.median(ring, axis=0)
    if (np.abs(a - paper).max(axis=2) <= 26).mean() < INDEX_MIN_PAPER:
        return None
    luma = a @ np.array([0.299, 0.587, 0.114])
    inky = (luma < 140).sum(axis=1) > 0.01 * w
    best = cur = 0
    for v in inky:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return "index" if best / h <= INDEX_MAX_INK_RUN else "art"


def is_index_page(path: Path) -> bool:
    """True for section/credits pages (comic title, authors, info)."""
    return paper_page_kind(path) == "index"


# section leaders start a new section and have a required side:
# pos % 2 == 0 is a left page, pos % 2 == 1 a right page
LEADER_SIDE = {"index": 0, "cover": 1}


def page_kind(path: Path) -> str:
    """Classify an upscaled page image: cover (filename marker), index
    (detected), spread halves, or story."""
    if path.stem.endswith("-1"):
        return "half1"
    if path.stem.endswith("-2"):
        return "story"
    if "cover" in path.stem.lower():
        return "cover"
    return "index" if is_index_page(path) else "story"


def book_plan(upscaled_dir: Path):
    """The book layout, computed from the upscaled pages (the single source
    of truth used by both compose — for page numbers — and merge)."""
    files = image_files(upscaled_dir)
    kinds = [page_kind(p) for p in files]
    plan, warnings = plan_layout(files, kinds)
    return files, kinds, plan, warnings


def plan_layout(pdfs: list, kinds: list):
    """Assign book positions (1 = right page, even = left, odd = right) and
    decide where to insert blank pages. Blanks only go at section boundaries:
    before a section leader (index pages must land on a left page, covers on
    a right page) or directly after one (to shift the section's story parity
    so every -1 spread half lands on a left page, facing its -2). Returns
    (plan, warnings) where plan entries are Paths or None for a blank."""
    segments, cur = [], []
    for p, k in zip(pdfs, kinds):
        if k in LEADER_SIDE and cur:
            segments.append(cur)
            cur = []
        cur.append((p, k))
    if cur:
        segments.append(cur)

    plan, warnings = [], []
    pos = 1
    for seg in segments:
        if seg[0][1] in LEADER_SIDE:
            if pos % 2 != LEADER_SIDE[seg[0][1]]:
                plan.append(None)
                pos += 1
            story = seg[1:]

            def bad_halves(shift):
                q = pos + 1 + shift
                bad = []
                for p, k in story:
                    if k == "half1" and q % 2 == 1:
                        bad.append(p.name)
                    q += 1
                return bad

            shift = 0 if len(bad_halves(0)) <= len(bad_halves(1)) else 1
            warnings.extend(f"{n} lands on a right page (section spreads "
                            f"conflict)" for n in bad_halves(shift))
            plan.append(seg[0][0])
            pos += 1
            if shift:
                plan.append(None)
                pos += 1
        else:
            story = seg
        for p, k in story:
            if k == "half1" and pos % 2 == 1 and seg[0][1] not in LEADER_SIDE:
                warnings.append(f"{p.name} lands on a right page "
                                f"(no allowed insertion point before it)")
            plan.append(p)
            pos += 1
    return plan, warnings


def cmd_merge(pdf_dir: Path, out_path: Path, upscaled_dir: Path) -> int:
    """Concatenate all page PDFs (alphabetical = page order, spread halves
    -1/-2 included) into one book PDF named after the comic folder, inserting
    white pages so index pages land on left-hand pages and split spreads on
    facing left-right pages."""
    import pikepdf

    if not upscaled_dir.is_dir() or not image_files(upscaled_dir):
        print(f"No upscaled images in {upscaled_dir} — run 'just upscale' first.")
        return 1

    print("Classifying pages...")
    files, kinds, plan, warnings = book_plan(upscaled_dir)
    missing = [e.stem for e in plan
               if e is not None and not (pdf_dir / (e.stem + ".pdf")).is_file()]
    if missing:
        print(f"Missing page PDFs ({len(missing)}) — run 'just compose' first: "
              + ", ".join(missing[:5]))
        return 1

    print("Layout plan (R = right page, L = left page):")
    for pos, entry in enumerate(plan, 1):
        side = "R" if pos % 2 else "L"
        if entry is None:
            print(f"  {pos:3d} {side}  [white page]")
        else:
            mark = {"index": " [index]", "cover": " [cover]",
                    "half1": " [spread left]"}.get(kinds[files.index(entry)], "")
            print(f"  {pos:3d} {side}  {entry.stem}.pdf{mark}")
    for wmsg in warnings:
        print(f"  WARNING: {wmsg}")

    geo = template_geometry()
    cw, ch = geo["canvas"]
    res = geo.get("res", 300.0)
    blank_size = (cw * 72 / res, ch * 72 / res)
    out = pikepdf.new()
    for entry in plan:
        if entry is None:
            out.add_blank_page(page_size=blank_size)
        else:
            with pikepdf.open(pdf_dir / (entry.stem + ".pdf")) as src:
                out.pages.extend(src.pages)
    out.save(out_path)
    blanks = sum(1 for e in plan if e is None)
    print(f"MERGED {len(plan) - blanks} pages + {blanks} white pages "
          f"-> {out_path} ({len(plan)} total)")
    return 0


# ---------------------------------------------------------------------------
# Re-lettered book: `process.py <comic> all --relettering` chains the base
# pipeline with the relettering/ scripts, pausing for the human
# transcription step. Every step is skipped by a filesystem marker.
# ---------------------------------------------------------------------------


def transcript_gaps(work: Path) -> dict:
    """Bubbles the transcription hasn't covered yet: {short page key:
    [bubble indices]} from sheets/manifest.json vs the keys present across
    parts/*.json (any value counts, "" included — presence is what matters,
    same as merge_transcripts.py)."""
    import json
    manifest = json.loads((work / "sheets" / "manifest.json").read_text())
    have = set()
    for part in sorted((work / "parts").glob("*.json")):
        have.update(json.loads(part.read_text()).keys())
    gaps = {}
    for key, info in manifest.items():
        missing = [bi for bi in range(1, info["count"] + 1)
                   if f"{key} b{bi:02d}" not in have]
        if missing:
            gaps[key] = missing
    return gaps


def parse_reletter_done(output: str) -> list:
    """Stems the GIMP text pass re-exported this run ("RELETTER DONE <stem>"
    lines from reletter_gimp.py) — exactly the PDFs that need postprocessing."""
    prefix = "RELETTER DONE "
    return [line[len(prefix):].rstrip() for line in output.splitlines()
            if line.startswith(prefix)]


def parse_reletter_output(output: str) -> dict:
    """Digest of a reletter_gimp.py run: stems re-exported ("done"), pages
    skipped as already lettered ("skipped"), whether the script reached its
    final "RELETTER ALL DONE" ("finished" — false = it died midway), and
    every non-RELETTER line ("other": GIMP noise, or the traceback)."""
    lines = output.splitlines()
    return {
        "done": parse_reletter_done(output),
        "skipped": sum(1 for l in lines if l.startswith("RELETTER SKIP ")),
        "finished": any(l.rstrip() == "RELETTER ALL DONE" for l in lines),
        "other": [l for l in lines if not l.startswith("RELETTER ")],
    }


def gimp_pending_stems(work: Path, xcf_dir: Path) -> list:
    """Laid-out pages whose XCF still lacks the text pass: the pass writes
    qa/<stem>.png last, so png mtime >= xcf mtime marks a page done (compose
    recomposing the page makes the XCF newer again). Pages with no XCF yet
    are not pending — compose hasn't produced them."""
    import json
    layout = json.loads((work / "layout.json").read_text())
    pending = []
    for stem in sorted(layout):
        xcf = xcf_dir / (stem + ".xcf")
        if not xcf.is_file():
            continue
        png = work / "qa" / (stem + ".png")
        if not png.is_file() or png.stat().st_mtime < xcf.stat().st_mtime:
            pending.append(stem)
    return pending


RELETTER_DIR = REPO / "relettering"
RELETTER_PY = REPO / ".venv-reletter" / "bin" / "python"
FONT_FILES = ("regular.ttf", "bolditalic.ttf")


def reletter_preflight(fonts_dir: Path = RELETTER_DIR / "fonts",
                       reletter_py: Path = RELETTER_PY,
                       gimp: Path = Path(GIMP)) -> list:
    """What the re-lettering stage needs beyond the base pipeline; checked
    up front so a missing font doesn't surface after hours of upscaling."""
    problems = []
    for f in FONT_FILES:
        if not (fonts_dir / f).is_file():
            problems.append(f"missing font {fonts_dir / f} — see "
                            f"relettering/fonts/README.md")
    if not reletter_py.is_file():
        problems.append(f"missing {reletter_py} — create it as described in "
                        f"README 'Setup'")
    if not gimp.is_file():
        problems.append(f"missing GIMP at {gimp}")
    return problems


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def wait_for_transcripts(work: Path, interactive: bool = None,
                         read_line=input) -> bool:
    """The human step. True once every manifest bubble has a transcript.
    Interactive (a terminal): print what to fill in and block on Enter,
    re-checking each time; Ctrl-C stops (False). Non-interactive: print the
    same and return False — re-running the command resumes."""
    if interactive is None:
        interactive = sys.stdin.isatty()
    while True:
        gaps = transcript_gaps(work)
        if not gaps:
            return True
        sheets = sorted((work / "sheets").glob("sheet-*.jpg"))
        bubbles = sum(len(v) for v in gaps.values())
        print("\n== TRANSCRIPTION NEEDED ==")
        if sheets:
            print(f"Sheets:   {_rel(sheets[0])} … {sheets[-1].name}  "
                  f"({len(sheets)} sheets)")
        print(f"Fill in:  {_rel(work / 'parts')}/sheet-NNN.json   "
              f"(one string per bubble key; \"\" = leave untouched)")
        print(f"Missing:  {bubbles} bubbles on {len(gaps)} pages")
        if not interactive:
            print("Re-run the same command to resume once they are filled.")
            return False
        try:
            read_line("Press Enter when done  (Ctrl-C to stop; re-run the "
                      "same command to resume) ")
        except (KeyboardInterrupt, EOFError):
            print("\nStopped. Re-run the same command to resume.")
            return False


def ensure_pristine(upscaled_dir: Path, work: Path) -> None:
    """Untouched copies of the upscaled pages for qa_scan's pixel diff — taken
    BEFORE the fit cleans upscaled/. Only ever fills gaps: a copy that exists
    is the pristine one by definition and is never overwritten."""
    import shutil
    pristine = work / "pristine"
    pristine.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in image_files(upscaled_dir):
        dst = pristine / src.name
        if dst.is_file():
            continue
        shutil.copy2(src, dst)      # keeps the mtime, like cp -p
        copied += 1
    print(f"pristine copies: {copied} added, "
          f"{len(image_files(upscaled_dir)) - copied} already present")


def reletter_status(work: Path, xcf_dir: Path) -> list:
    """The upfront table: (step, state) per relettering marker, so a resumed
    run shows where it stands before doing anything. States are "pending",
    "done", or a short detail (transcription gap, pages left to letter)."""
    fitted = (work / "layout.json").is_file()
    manifest = work / "sheets" / "manifest.json"
    pristine = work / "pristine"
    rows = [
        ("pristine copies",
         "n/a (fit already ran)" if fitted
         else "done" if pristine.is_dir() and image_files(pristine)
         else "pending"),
        ("bubble detection",
         "done" if any((work / "bubbles").glob("*.json")) else "pending"),
        ("contact sheets", "done" if manifest.is_file() else "pending"),
    ]
    if manifest.is_file():
        gaps = transcript_gaps(work)
        rows.append(("transcription", "complete" if not gaps else
                     f"{sum(len(v) for v in gaps.values())} bubbles missing "
                     f"on {len(gaps)} pages"))
    else:
        rows.append(("transcription", "pending"))
    rows.append(("fit + clean", "done" if fitted else "pending"))
    if fitted:
        left = gimp_pending_stems(work, xcf_dir)
        rows.append(("GIMP text layers",
                     "done" if not left else f"{len(left)} pages pending"))
    else:
        rows.append(("GIMP text layers", "pending"))
    return rows


def run_reletter_script(script: str, name: str) -> int:
    """One relettering/ script under its own venv (cv2 lives only there),
    output streamed through."""
    cmd = [str(RELETTER_PY), str(RELETTER_DIR / script), name]
    print(f"$ .venv-reletter/bin/python relettering/{script} \"{name}\"",
          flush=True)   # keep our output ahead of the child's in a log file
    return subprocess.call(cmd, cwd=REPO)


def run_gimp_text_pass(name: str) -> list:
    """reletter_gimp.py under GIMP batch (adds the text layers, re-exports
    the PDFs). Returns the stems it re-exported this run."""
    import os
    env = dict(os.environ, RELETTER_COMIC=name, RELETTER_REPO=str(REPO))
    cmd = [GIMP, "-i", "-d", "--batch-interpreter", "python-fu-eval",
           "-b", "execfile('relettering/reletter_gimp.py')",
           "-b", "pdb.gimp_quit(0)"]
    proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            errors="replace")
    lines = []
    for line in proc.stdout:
        # progress: one line per lettered page; the SKIP flood is summarized
        if line.startswith("RELETTER ") and not line.startswith("RELETTER SKIP "):
            sys.stdout.write(line)
            sys.stdout.flush()
        lines.append(line)
    proc.wait()
    result = parse_reletter_output("".join(lines))
    if result["skipped"]:
        print(f"{result['skipped']} pages already lettered (skipped)")
    if not result["finished"]:
        # GIMP on macOS often exits non-zero after a SUCCESSFUL batch, so the
        # exit code means nothing; a missing ALL DONE line does — show why
        print(f"!! reletter_gimp.py did not finish (GIMP exit {proc.returncode})"
              " — last lines of its output:")
        for l in result["other"][-40:]:
            print("   " + l)
    return result["done"]


def cmd_reletter_book(comic_dir: Path, upscaled_dir: Path, xcf_dir: Path,
                      pdf_dir: Path, merged: Path, engine: str = "realesrgan",
                      headed: bool = False) -> int:
    """upscale -> pristine -> detect -> sheets -> [transcription pause] ->
    merge -> fit -> compose -> GIMP text -> postprocess -> merge -> qa.
    Resumable: every step is skipped by a filesystem marker, so the same
    command continues after the pause (or after any failure)."""
    name = comic_dir.name
    work = RELETTER_DIR / name
    problems = reletter_preflight()
    if problems:
        print("Re-lettering needs, before anything runs:")
        for msg in problems:
            print(f"  - {msg}")
        return 1
    fitted = (work / "layout.json").is_file()

    print(f"Re-lettering status for {name} (upscale/compose/merge check "
          "their own outputs):")
    for label, state in reletter_status(work, xcf_dir):
        print(f"  {label:<18} {state}")

    def step(title: str) -> None:
        print(f"\n=== {title} ===", flush=True)   # visible in redirected logs

    step("upscale")
    rc = cmd_upscale(comic_dir, upscaled_dir, engine=engine, headed=headed,
                     cleanup_dirs=(xcf_dir, pdf_dir))
    if rc != 0:
        print("Upscale had failures — fix/retry, then re-run the same command.")
        return rc

    step("pristine copies")
    if fitted:
        print("skipped: the fit already cleaned upscaled/ — too late to take "
              "pristine copies (regenerate one from the source if needed)")
    else:
        ensure_pristine(upscaled_dir, work)

    step("bubble detection")
    if any((work / "bubbles").glob("*.json")):
        print(f"done: {_rel(work / 'bubbles')} exists. Detection is never "
              "re-run automatically (it could reorder bubbles under the "
              "transcripts); re-detect single pages per ARCHITECTURE.md.")
    else:
        rc = run_reletter_script("reletter_detect.py", name)
        if rc != 0:
            return rc

    step("contact sheets")
    if (work / "sheets" / "manifest.json").is_file():
        print(f"done: {_rel(work / 'sheets' / 'manifest.json')} exists")
    else:
        rc = run_reletter_script("make_sheets.py", name)
        if rc != 0:
            return rc

    step("transcription")
    if not wait_for_transcripts(work):
        return 0
    print("complete")
    rc = run_reletter_script("merge_transcripts.py", name)
    if rc != 0:
        return rc

    step("fit + clean")
    if fitted:
        print(f"done: {_rel(work / 'layout.json')} exists. A global refit is "
              "never automatic (it re-encodes every working page); to redo "
              "one page follow the single-page recipe in ARCHITECTURE.md.")
    else:
        rc = run_reletter_script("reletter_fit.py", name)
        if rc != 0:
            return rc

    step("compose")
    rc = cmd_compose(upscaled_dir, xcf_dir, pdf_dir, comic_dir)
    if rc != 0:
        return rc

    step("GIMP text layers")
    pending = gimp_pending_stems(work, xcf_dir)
    if not pending:
        print("done: every laid-out page has a QA render newer than its XCF")
    else:
        print(f"{len(pending)} pages to letter")
        done = run_gimp_text_pass(name)
        step("PDF postprocess")
        for stem in done:
            postprocess_pdf(pdf_dir / (stem + ".pdf"))
        print(f"{len(done)} PDFs recompressed + sRGB-tagged")
        still = gimp_pending_stems(work, xcf_dir)
        if still:
            print(f"!! {len(still)} pages still pending after the GIMP pass "
                  f"(see the RELETTER lines above): " + ", ".join(still[:5]))
            return 1

    step("merge")
    rc = cmd_merge(pdf_dir, merged, upscaled_dir)
    if rc != 0:
        return rc

    step("QA scan (report only)")
    if run_reletter_script("qa_scan.py", name) != 0:
        print("qa_scan flagged pages — review them per ARCHITECTURE.md "
              "'Verification tooling' (the book PDF is built regardless)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("comic", help="comic folder (name inside this repo, or a path)")
    ap.add_argument("step", nargs="?", default="all",
                    choices=["scan", "upscale", "compose", "merge", "all"])
    ap.add_argument("--engine", default="realesrgan",
                    choices=["realesrgan", "iloveimg"],
                    help="upscaler: local Real-ESRGAN (default, keeps comic "
                         "lettering intact) or iloveimg.com")
    ap.add_argument("--headed", action="store_true",
                    help="show the browser while upscaling (iloveimg only)")
    ap.add_argument("--relettering", action="store_true",
                    help="full pipeline with the re-lettering stage: pauses "
                         "for the transcription, resumes when it is filled")
    args = ap.parse_args()
    if args.relettering and args.step != "all":
        print("--relettering only applies to the full pipeline "
              "(process.py <comic> all --relettering, i.e. 'just process').")
        return 2

    comic_dir = Path(args.comic)
    if not comic_dir.is_dir():
        comic_dir = REPO / args.comic
    if not comic_dir.is_dir():
        print(f"Comic folder not found: {args.comic}")
        return 1
    comic_dir = comic_dir.resolve()

    name = comic_dir.name
    upscaled_dir = REPO / "upscaled" / name
    xcf_dir = REPO / "xcf" / name
    pdf_dir = REPO / "pdf" / name

    if args.step == "scan":
        report_scan(comic_dir, upscaled_dir, args.engine)
        return 0
    if args.relettering:
        return cmd_reletter_book(comic_dir, upscaled_dir, xcf_dir, pdf_dir,
                                 REPO / "pdf" / f"{name}.pdf",
                                 engine=args.engine, headed=args.headed)
    if args.step in ("upscale", "all"):
        rc = cmd_upscale(comic_dir, upscaled_dir, engine=args.engine,
                         headed=args.headed,
                         cleanup_dirs=(xcf_dir, pdf_dir))
        if rc != 0 and args.step == "all":
            print("Upscale had failures — fix/retry before composing, "
                  "or run 'just compose' to compose what succeeded.")
            return rc
        if args.step == "upscale":
            return rc
    merged = REPO / "pdf" / f"{name}.pdf"
    if args.step == "merge":
        return cmd_merge(pdf_dir, merged, upscaled_dir)
    rc = cmd_compose(upscaled_dir, xcf_dir, pdf_dir, comic_dir)
    if rc != 0 or args.step == "compose":
        return rc
    return cmd_merge(pdf_dir, merged, upscaled_dir)


if __name__ == "__main__":
    sys.exit(main())
