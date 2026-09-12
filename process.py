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
    mask = np.abs(a - paper).max(axis=2) > tolerance
    rows = np.flatnonzero(mask.mean(axis=1) > min_frac)
    cols = np.flatnonzero(mask.mean(axis=0) > min_frac)
    if not len(rows) or not len(cols):
        return im
    return im.crop((int(cols[0]), int(rows[0]),
                    int(cols[-1]) + 1, int(rows[-1]) + 1))


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
        cropped = crop_to_content(img)
        note = (f", cropped from {img.width}x{img.height}"
                if cropped.size != img.size else "")
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

    # page numbers = final book positions (blanks included); drawn only on
    # story pages and spread halves — never on covers, indexes, or blanks
    import json
    import re
    print("Computing book layout for page numbers...")
    files, kinds, plan, _ = book_plan(in_dir)

    def source_of(f: Path) -> Path:
        stem = f.stem
        if stem.endswith(("-1", "-2")):
            stem = stem[:-2]
        return next(iter(comic_dir.glob(stem + ".*")), None)

    def is_art(f: Path) -> bool:
        # art pages get no number: the word 'art' in the filename (marker,
        # like 'cover'), or auto-detected on the SOURCE image (the upscaled
        # file is cropped, which strips the margins the detector needs)
        if re.search(r"\bart\b", f.stem, re.IGNORECASE):
            return True
        src = source_of(f)
        return src is not None and paper_page_kind(src) == "art"

    numbers = {}
    for pos, entry in enumerate(plan, 1):
        if (entry is not None
                and kinds[files.index(entry)] in ("story", "half1")
                and not is_art(entry)):
            numbers[entry.name] = pos

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

    # per-page metadata for the plugin: page number ('-' = none) and the
    # paper color measured from the SOURCE image (the cropped content's own
    # border is art, so the plugin can no longer measure it itself)
    # badge zone: between the lowest possible art edge and the trim line
    art_bottom = (canvas_size()[1] + story_box()[1]) // 2
    lines = [f"ART_BOTTOM {art_bottom}", f"STRIP_BOTTOM {strip_bottom_y()}"]
    for f in files:
        num = numbers.get(f.name, "-")
        src = source_of(f)
        rgb = ",".join(map(str, paper_color(src))) if src else "-"
        lines.append(f"{f.name}\t{num}\t{rgb}")
    numbers_file = xcf_dir / "numbers.txt"
    numbers_file.write_text("\n".join(lines) + "\n")

    call = "(python-fu-webcomics-compose RUN-NONINTERACTIVE {} {} {} {} {})".format(
        scm_quote(str(in_dir)), scm_quote(str(TEMPLATE)),
        scm_quote(str(xcf_dir)), scm_quote(str(pdf_dir)),
        scm_quote(str(numbers_file)))
    # no -f: fonts must load for the page-number text
    cmd = [GIMP, "-i", "-d", "-b", call, "-b", "(gimp-quit 0)"]
    print("Running GIMP batch (this can take a while)...", flush=True)
    # GIMP on macOS often exits non-zero after a successful batch run
    # (gimp_wire_write_msg noise), so success is judged from the plugin's
    # own "COMPOSE DONE ... 0 failed" line instead of the return code.
    proc = subprocess.run(cmd, text=True, capture_output=True)
    lines = (proc.stdout + proc.stderr).splitlines()
    interesting = [l for l in lines if l.startswith(("COMPOSE", "ERROR"))]
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
    args = ap.parse_args()

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
