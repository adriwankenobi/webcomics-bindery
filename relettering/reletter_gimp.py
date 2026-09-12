# -*- coding: utf-8 -*-
# GIMP 2.10 python-fu (Python 2) batch: add the re-lettered text as editable
# text layers on top of each composed XCF, save the XCF in place, re-export
# the flattened PDF and a PNG preview for QA.
# Run via: gimp -i -d --batch-interpreter python-fu-eval -b "execfile(...)"

import json
import os
import sys

from gimpfu import *  # noqa

# run from the repo root (execfile has no __file__); comic via env var:
#   RELETTER_COMIC="<comic>" gimp -i -d --batch-interpreter python-fu-eval \
#     -b "execfile('relettering/reletter_gimp.py')" -b "pdb.gimp_quit(0)"
REPO = os.environ.get("RELETTER_REPO", os.getcwd())
NAME = os.environ.get("RELETTER_COMIC")
if not NAME:
    raise SystemExit("RELETTER_COMIC env var must name the comic folder")
XCF_DIR = os.path.join(REPO, "xcf", NAME)
PDF_DIR = os.path.join(REPO, "pdf", NAME)
UP_DIR = os.path.join(REPO, "upscaled", NAME)
WORK = os.path.join(REPO, "relettering", NAME)
QA_DIR = os.path.join(WORK, "qa")

# GIMP font family names for the user-supplied comic font (the font must
# also be installed system-wide so GIMP can see it). Defaults below;
# override them in relettering/fonts/fonts.json:
#   {"gimp_regular": "<family>", "gimp_bolditalic": "<family>"}
FONT = {"regular": "CC WildWords Lower",
        "bolditalic": "CC WildWords Lower Bold Italic"}
_fonts_json = os.path.join(REPO, "relettering", "fonts", "fonts.json")
if os.path.isfile(_fonts_json):
    _fj = json.load(open(_fonts_json))
    FONT["regular"] = _fj.get("gimp_regular", FONT["regular"])
    FONT["bolditalic"] = _fj.get("gimp_bolditalic", FONT["bolditalic"])

layout = json.load(open(os.path.join(WORK, "layout.json")))


def log(msg):
    sys.stdout.write("RELETTER " + msg + "\n")
    sys.stdout.flush()


def make_text_layer(img, text, style, size, color=(0, 0, 0)):
    lyr = pdb.gimp_text_fontname(img, None, 0, 0, text.encode("utf-8"),
                                 0, True, size, 0, FONT[style])
    pdb.gimp_text_layer_set_color(lyr, color)
    pdb.gimp_image_set_active_layer(img, lyr)
    return lyr


def space_width(img, style, size):
    a = make_text_layer(img, u"N N", style, size)
    wa = a.width
    pdb.gimp_image_remove_layer(img, a)
    b = make_text_layer(img, u"NN", style, size)
    wb = b.width
    pdb.gimp_image_remove_layer(img, b)
    return wa - wb


for stem in sorted(layout.keys()):
    xcf = os.path.join(XCF_DIR, stem + ".xcf")
    if not os.path.isfile(xcf):
        log("MISSING " + xcf)
        continue
    img = pdb.gimp_file_load(xcf, xcf)
    # the page layer: dimensions match the upscaled image
    up = pdb.gimp_file_load(os.path.join(UP_DIR, stem + ".jpg"), "u")
    uw, uh = up.width, up.height
    pdb.gimp_image_delete(up)
    page = None
    for lyr in img.layers:
        if lyr.width == uw and lyr.height == uh:
            page = lyr
            break
    if page is None:
        log("NO PAGE LAYER " + stem)
        pdb.gimp_image_delete(img)
        continue
    ox, oy = pdb.gimp_drawable_offsets(page)

    for lyr in list(img.layers):
        if lyr.name == "relettering":
            pdb.gimp_image_remove_layer(img, lyr)

    group = pdb.gimp_layer_group_new(img)
    pdb.gimp_item_set_name(group, "relettering")
    pdb.gimp_image_insert_layer(img, group, None, 0)

    sw_cache = {}
    for b in layout[stem]:
        bgroup = pdb.gimp_layer_group_new(img)
        pdb.gimp_item_set_name(bgroup, "bubble-%02d" % b["index"])
        pdb.gimp_image_insert_layer(img, bgroup, group, 0)
        size = b["font_px"]
        lh = b["line_height"]
        color = (255, 255, 255) if b.get("color") == "white" else (0, 0, 0)
        for li, line in enumerate(b["lines"]):
            runs = line["runs"]
            cx = ox + line["cx"]
            y0 = oy + line["y_top"]
            if len(runs) == 1:
                st, txt = runs[0]
                lyr = make_text_layer(img, txt, st, size, color)
                pdb.gimp_image_reorder_item(img, lyr, bgroup, 0)
                pdb.gimp_layer_set_offsets(
                    lyr, int(round(cx - lyr.width / 2.0)),
                    int(round(y0 + (lh - lyr.height) / 2.0)))
            else:
                key = size
                if key not in sw_cache:
                    sw_cache[key] = space_width(img, "regular", size)
                sp = sw_cache[key]
                lyrs = []
                for st, txt in runs:
                    lyr = make_text_layer(img, txt, st, size, color)
                    pdb.gimp_image_reorder_item(img, lyr, bgroup, 0)
                    lyrs.append(lyr)
                total = sum(l.width for l in lyrs) + sp * (len(lyrs) - 1)
                x = cx - total / 2.0
                for l in lyrs:
                    pdb.gimp_layer_set_offsets(
                        l, int(round(x)),
                        int(round(y0 + (lh - l.height) / 2.0)))
                    x += l.width + sp

    pdb.gimp_xcf_save(0, img, img.layers[0], xcf, xcf)

    dup = pdb.gimp_image_duplicate(img)
    flat = pdb.gimp_image_flatten(dup)
    pdf = os.path.join(PDF_DIR, stem + ".pdf")
    pdb.file_pdf_save(dup, flat, pdf, pdf, False, True, True)
    scaled = pdb.gimp_image_duplicate(dup)
    pdb.gimp_image_scale(scaled, scaled.width // 2, scaled.height // 2)
    png = os.path.join(QA_DIR, stem + ".png")
    pdb.file_png_save(scaled, pdb.gimp_image_flatten(scaled),
                      png, png, 0, 6, 1, 1, 1, 1, 1)
    pdb.gimp_image_delete(scaled)
    pdb.gimp_image_delete(dup)
    pdb.gimp_image_delete(img)
    log("DONE " + stem)

log("ALL DONE")
