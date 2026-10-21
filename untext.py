"""Remove the text from one image, keep everything else as it is.

  just untext "<image>"  -> <stem>-untexted<ext> + <stem>-untexted-mask.png
                            beside the input (the input is never touched)

The mask PNG (white = remove) is reused on the next run, so a mask fixed by
hand in GIMP sticks; delete it to detect again. Steps:
  1. CRAFT (EasyOCR's text detector, no recognition) finds the text blocks;
  2. inside each block the letters are the colour the block's border is not,
     plus anti-aliased marks that are a BLEND of that colour and the ground
     (a border that is busy ART: the flat colour off the border, proved by
     a flat outline round it, outline included);
  3. each block's mask is grown by its own halo (drop shadow / glow), measured
     by walking out from the letters until the surroundings stop changing;
  4. LaMa inpaints the mask, and only masked pixels are pasted back.
Runs in .venv-untext (`just setup-untext`); models live untracked in tools/.
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parent
LAMA = REPO / "tools" / "lama" / "big-lama.pt"
LAMA_URL = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
LAMA_MD5 = "e3aa4aaa15225a33ec84f9f4bc47e500"
CRAFT_DIR = REPO / "tools" / "easyocr"
SUFFIX = "-untexted"
MASK_SUFFIX = SUFFIX + "-mask"

# Pixel constants are tuned at this long side; larger images scale them
# (a halo, a pad, a speck all grow with the artwork's resolution).
BASE_SIDE = 1300
PAD = 6            # px added round each detected box before segmenting
MIN_AREA = 6       # px^2: smaller pieces are noise
BIG_FRAC = 0.2     # pieces >= this x the largest set the letters' colour
MARK_FRAC = 0.02   # pieces under this x the largest are thin marks (R, TM)
BLEND_MIN_T = 0.3  # how far along ground->ink a piece's colour must sit
BLEND_PERP = 30    # how far OFF the ground->ink line it may sit
MIN_CONTRAST = 150  # letters' colour vs the ground: lettering is set off
INK_FLAT = 9       # median colour spread inside the letters: lettering is flat
HALO_STEPS = 40    # how far (in scaled px) the halo walk looks
FLAT = 1.6         # luminance change per step that counts as flat
FLAT_RUN = 3       # ... for this many steps in a row
HALO_MARGIN = 2    # px beyond the measured halo
BUSY_RING = 40     # median distance of the border from its own median: no ground
CLUSTERS = 8       # colours a busy box is split into
OFF_BORDER = 0.01  # the letters' colour holds at most this share of the border
MIN_SHARE = 0.05   # ... and at least this share of the box
INK_TOL = 40       # colour distance that still counts as the letters' (or outline's)
EDGE_HOLD = 0.6    # share of a band round the letters that must be outline colour
EDGE_MAX = 12      # how far (in scaled px) the outline is followed
WHOLE_MAX = 1600   # images up to this long side are inpainted in one go
WINDOW_MARGIN = 128  # min context round a block when inpainted in windows


def output_paths(src):
    src = Path(src)
    if src.stem.endswith(SUFFIX) or src.stem.endswith(MASK_SUFFIX):
        raise ValueError(f"{src.name} is an untext output, not a source image")
    return (src.with_name(src.stem + SUFFIX + src.suffix),
            src.with_name(src.stem + MASK_SUFFIX + ".png"))


def scale_of(shape):
    return max(1, round(max(shape[:2]) / BASE_SIDE))


def otsu(v):
    hist = np.bincount(v.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (mu[-1] * w - mu) ** 2 / (w * (1 - w))
    return int(np.nanargmax(s))


def ring(a, k=3):
    return np.concatenate([a[:k].reshape(-1, *a.shape[2:]), a[-k:].reshape(-1, *a.shape[2:]),
                           a[:, :k].reshape(-1, *a.shape[2:]), a[:, -k:].reshape(-1, *a.shape[2:])])


def block_letters(reg, step=1):
    """Letter pixels of one text block (bool, reg's shape).

    The letters are what the block's border is not (distance from the border's
    colour), so light-on-dark and dark-on-light need no polarity guess. CRAFT
    boxes sit tight on the glyphs, hence PAD before this: the border has to be
    ground. A piece counts only if its colour is a
    BLEND of the ground and the letters' colour: an anti-aliased mark is,
    artwork of another colour (rocks, shadows) is off that line even when it
    is just as far from the ground."""
    reg = reg.astype(float)
    border = ring(reg)
    ground = np.median(border, axis=0)
    if np.median(np.sqrt(((border - ground) ** 2).sum(axis=1))) > BUSY_RING:
        return outlined_letters(reg, step)
    dist = np.sqrt(((reg - ground) ** 2).sum(axis=2))
    if dist.max() < 40:  # no contrast in the box: nothing to take
        return np.zeros(reg.shape[:2], bool)
    dist8 = np.clip(dist * 255 / dist.max(), 0, 255).astype(np.uint8)
    fg = dist8 > otsu(dist8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg.astype(np.uint8), connectivity=8)
    if n < 2:
        return np.zeros(reg.shape[:2], bool)
    areas = st[1:, cv2.CC_STAT_AREA]
    big = np.isin(lab, [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= BIG_FRAC * areas.max()])
    ink = np.median(reg[big], axis=0)
    d = ink - ground
    dd = max(float(np.dot(d, d)), 1.0)
    keep = np.zeros(fg.shape, bool)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a < MIN_AREA * step * step:
            continue
        col = np.median(reg[lab == i], axis=0)
        t = float(np.dot(col - ground, d)) / dd
        perp = float(np.sqrt(((col - (ground + t * d)) ** 2).sum()))
        if t < BLEND_MIN_T or perp > BLEND_PERP:
            continue
        if a < MARK_FRAC * areas.max():
            keep[y:y + h, x:x + w] = True  # a thin mark: its whole bbox
        else:
            keep |= lab == i
    if keep.any() and not printed(reg, keep, ground, step):
        return np.zeros(fg.shape, bool)
    return keep


def outlined_letters(reg, step=1):
    """Letters + outline of a block whose border is ARTWORK, not a ground (a
    title set straight on the art, read by an outline of its own): measured
    against that border, the art's whites and blacks are further off than the
    letters and pass for them. The letters are instead the flat colour that
    holds a fair share of the box and none of its border; the proof that it
    IS lettering is a flat outline all round it, which goes in the mask too
    (an outline is lettering). Artwork's flat patches have no outline."""
    h, w = reg.shape[:2]
    px = reg.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(0)  # same image, same mask
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, _, cen = cv2.kmeans(px[::max(1, len(px) // 20000)], CLUSTERS, None, crit, 3,
                           cv2.KMEANS_PP_CENTERS)
    lab = np.argmin(((px[:, None, :] - cen[None]) ** 2).sum(axis=2), axis=1).reshape(h, w)
    on_border = ring(lab)
    for c in sorted(range(len(cen)), key=lambda c: -(lab == c).sum()):
        if (lab == c).mean() < MIN_SHARE:
            break
        if (on_border == c).mean() > OFF_BORDER:
            continue
        ink = np.sqrt(((reg - cen[c]) ** 2).sum(axis=2)) < INK_TOL
        n, cl, st, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        letters = np.zeros((h, w), bool)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a >= MIN_AREA * step * step and x > 0 and y > 0 and x + bw < w and y + bh < h:
                letters |= cl == i
        keep = letters.any() and with_outline(reg, letters, step)
        if keep is not False:
            return keep
    return np.zeros((h, w), bool)


def with_outline(reg, letters, step):
    """letters grown by the flat outline round them, or False if there is
    none. Walked out in bands from the letters (the first is their
    anti-aliased seam) while most of each band is the outline's colour."""
    dt = cv2.distanceTransform((~letters).astype(np.uint8), cv2.DIST_L2, 5)

    def band(k):
        return (dt > (k - 1) * step) & (dt <= k * step)

    if not band(2).any():
        return False
    edge = np.median(reg[band(2)], axis=0)
    if not printed(reg, letters, edge, step):
        return False
    width = 0
    for k in range(2, EDGE_MAX + 1):
        b = band(k)
        if not b.any() or (np.sqrt(((reg[b] - edge) ** 2).sum(axis=1)) < INK_TOL).mean() < EDGE_HOLD:
            break
        width = k
    return dt <= width * step if width else False


def printed(reg, keep, ground, step):
    """Do the letters look like lettering? The detector also fires on
    ARTWORK (face markings, a hilt, a patch of rocks, a signature), which
    passes the box test. Lettering is FLAT colour and strongly set off from
    the art so it reads; painted marks are textured and blend into the
    ground. Measured on the letters' core (edges are anti-aliased blends)."""
    core = cv2.erode(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=step) > 0
    if core.sum() < 20:
        core = keep
    px = reg[core]
    ink = np.median(px, axis=0)
    spread = np.median(np.sqrt(((px - ink) ** 2).sum(axis=1)))
    contrast = np.sqrt(((ink - ground) ** 2).sum())
    return contrast >= MIN_CONTRAST and spread <= INK_FLAT


def halo_width(lum, letters, step=1):
    """How far the letters' halo (drop shadow, glow) reaches, in px: walk out
    from the letters until the luminance profile stays flat — or TURNS BACK.
    A halo fades one way only; artwork round the letters (a face, folds, a
    hilt under a false detection) rises and falls and never goes flat for
    long, so without the turn the walk runs out tens of px into the art."""
    dt = cv2.distanceTransform((~letters).astype(np.uint8), cv2.DIST_L2, 5)
    prof = []
    for k in range(1, HALO_STEPS + 1):
        band = (dt > (k - 1) * step) & (dt <= k * step)
        prof.append(lum[band].mean() if band.any() else np.nan)
    way = 0  # +1 brightening outward (a shadow), -1 darkening (a glow), 0 not known yet
    for k in range(2, HALO_STEPS - FLAT_RUN):
        run = [abs(prof[j] - prof[j - 1]) for j in range(k, k + FLAT_RUN)]
        if all(r < FLAT for r in run):  # NaN (no ground there) never passes
            return k * step
        diff = prof[k] - prof[k - 1]
        if way and diff * way <= -FLAT:
            return k * step
        if not way and abs(diff) >= FLAT:
            way = 1 if diff > 0 else -1
    return HALO_STEPS * step


def text_mask(img, boxes):
    """uint8 mask (255 = remove) for the text in `boxes` [(x0, y0, x1, y1)]."""
    H, W = img.shape[:2]
    step = scale_of(img.shape)
    pad = PAD * step
    reach = (HALO_STEPS + 5) * step
    lum = img.astype(float).mean(axis=2)
    mask = np.zeros((H, W), np.uint8)
    for x0, y0, x1, y1 in boxes:
        if y1 - y0 > x1 - x0:
            # horizontal text is never taller than wide; the detector's
            # tall boxes are artwork (face markings, a hilt)
            continue
        x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
        x1, y1 = min(W, int(x1) + pad), min(H, int(y1) + pad)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        keep = block_letters(img[y0:y1, x0:x1], step)
        if not keep.any():
            continue
        # the halo is measured and grown in a window reaching past the box
        wx0, wy0 = max(0, x0 - reach), max(0, y0 - reach)
        wx1, wy1 = min(W, x1 + reach), min(H, y1 + reach)
        sub = np.zeros((wy1 - wy0, wx1 - wx0), bool)
        sub[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0] = keep
        g = halo_width(lum[wy0:wy1, wx0:wx1], sub, step) + HALO_MARGIN * step
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        mask[wy0:wy1, wx0:wx1] |= cv2.dilate(sub.astype(np.uint8) * 255, k)
    return mask


def windows(mask):
    """Rectangles (x0, y0, x1, y1) to inpaint, together covering the mask.
    A small image goes whole (most context); a large one per group of
    nearby blocks, each with a margin of context round it."""
    H, W = mask.shape
    if max(H, W) <= WHOLE_MAX:
        return [(0, 0, W, H)]
    f = 8  # group on a coarse copy: a 257px dilation at full size is slow
    small = cv2.resize((mask > 0).astype(np.uint8), (max(1, W // f), max(1, H // f)),
                       interpolation=cv2.INTER_AREA) > 0
    small = cv2.dilate(small.astype(np.uint8), np.ones((2 * WINDOW_MARGIN // f + 1,) * 2, np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(small, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, _ = st[i]
        x0, y0, x1, y1 = x * f, y * f, min(W, (x + w) * f + f), min(H, (y + h) * f + f)
        sub = mask[y0:y1, x0:x1] > 0
        if not sub.any():
            continue
        ys, xs = np.nonzero(sub)
        bx0, bx1, by0, by1 = x0 + xs.min(), x0 + xs.max() + 1, y0 + ys.min(), y0 + ys.max() + 1
        m = max(WINDOW_MARGIN, (max(bx1 - bx0, by1 - by0)) // 4)
        out.append((max(0, bx0 - m), max(0, by0 - m), min(W, bx1 + m), min(H, by1 + m)))
    return out


def paste_back(img, fill, mask):
    res = img.copy()
    sel = mask > 0
    res[sel] = fill[sel]
    return res


_READER = None
_LAMA = None


def detect_boxes(img):
    """Text block boxes from CRAFT (detection only, any script)."""
    global _READER
    import easyocr
    if _READER is None:
        CRAFT_DIR.mkdir(parents=True, exist_ok=True)
        _READER = easyocr.Reader(["en"], gpu=False, recognizer=False, verbose=False,
                                 model_storage_directory=str(CRAFT_DIR),
                                 user_network_directory=str(CRAFT_DIR / "user_network"))
    H, W = img.shape[:2]
    horiz, free = _READER.detect(img, text_threshold=0.6, low_text=0.3,
                                 link_threshold=0.3, mag_ratio=1.5)
    boxes = [(x0, y0, x1, y1) for x0, x1, y0, y1 in horiz[0]]
    for poly in free[0]:  # rotated text: its bounding box
        xs, ys = [p[0] for p in poly], [p[1] for p in poly]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return [(max(0, int(a)), max(0, int(b)), min(W, int(c)), min(H, int(d)))
            for a, b, c, d in boxes]


def lama_fill(img, mask):
    """img with every masked pixel inpainted by LaMa (CPU)."""
    global _LAMA
    import torch
    if _LAMA is None:
        if not LAMA.exists():
            sys.exit(f"missing {LAMA.relative_to(REPO)}: run `just setup-untext`")
        _LAMA = torch.jit.load(str(LAMA), map_location="cpu").eval()
    work = img.copy()
    for x0, y0, x1, y1 in windows(mask):
        im, mk = work[y0:y1, x0:x1], mask[y0:y1, x0:x1] > 0
        if not mk.any():
            continue
        h, w = mk.shape
        ph, pw = -h % 8, -w % 8  # LaMa wants sides divisible by 8
        im_p = np.pad(im, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        mk_p = np.pad(mk, ((0, ph), (0, pw)), mode="symmetric")
        with torch.no_grad():
            x = torch.from_numpy(im_p).permute(2, 0, 1)[None].float() / 255
            m = torch.from_numpy(mk_p)[None, None].float()
            y = _LAMA(x, m)[0].permute(1, 2, 0).numpy()
        out = np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)[:h, :w]
        im[mk] = out[mk]  # later windows see this one's fill as context
    return work


def save_like(arr, src_im, alpha, path):
    out = Image.fromarray(arr)
    if alpha is not None:
        out.putalpha(alpha)
    elif src_im.mode == "L":
        out = out.convert("L")
    kw = {k: src_im.info[k] for k in ("icc_profile", "dpi", "exif") if k in src_im.info}
    fmt = (src_im.format or "").upper()
    if fmt == "JPEG":
        kw.update(quality=100, subsampling=0)
    elif fmt == "WEBP":
        kw.update(lossless=True)
    out.save(path, **kw)


def run(src, detect=detect_boxes, fill=lama_fill):
    """Untext one image. Returns the output path, or None when the mask is
    empty (the mask is still written, to paint by hand)."""
    src = Path(src)
    out, mask_p = output_paths(src)
    with Image.open(src) as im:
        im.load()  # format, mode and info stay readable after the close
    alpha = im.getchannel("A") if "A" in im.getbands() else None
    img = np.array(im.convert("RGB"))
    if mask_p.exists():
        with Image.open(mask_p) as m:
            mask = np.array(m.convert("L"))
        if mask.shape != img.shape[:2]:
            sys.exit(f"{mask_p.name} is {mask.shape[1]}x{mask.shape[0]}, the image is "
                     f"{img.shape[1]}x{img.shape[0]}: fix it, or delete it to detect again")
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
        print(f"using existing mask {mask_p.name} (delete it to detect again)")
    else:
        boxes = detect(img)
        mask = text_mask(img, boxes)
        Image.fromarray(mask).save(mask_p)
        print(f"{len(boxes)} text block(s) found -> {mask_p.name}")
    if not mask.any():
        print(f"nothing to remove: paint white in {mask_p.name} where text is, then re-run")
        return None
    res = paste_back(img, fill(img, mask), mask)
    save_like(res, im, alpha, out)
    print(f"wrote {out}  ({mask.mean() / 2.55:.1f}% of the image replaced)")
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def setup():
    """Fetch both models into tools/ (idempotent)."""
    if not LAMA.exists() or md5(LAMA) != LAMA_MD5:
        LAMA.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading LaMa -> {LAMA.relative_to(REPO)} (~200MB)")
        tmp = LAMA.with_suffix(".part")
        urllib.request.urlretrieve(LAMA_URL, tmp)
        if md5(tmp) != LAMA_MD5:
            tmp.unlink()
            sys.exit("LaMa download failed its checksum")
        tmp.rename(LAMA)
    detect_boxes(np.zeros((64, 64, 3), np.uint8))  # fetches CRAFT on first use
    print("untext models ready")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Remove the text from one image.")
    ap.add_argument("image", nargs="?", help="the image file (one file, not a folder)")
    ap.add_argument("--setup", action="store_true", help="download the models")
    a = ap.parse_args(argv)
    if a.setup:
        return setup()
    if not a.image:
        ap.error("give one image file")
    src = Path(a.image)
    if not src.is_file():
        sys.exit(f"not a file: {src}")
    try:
        run(src)
    except ValueError as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
