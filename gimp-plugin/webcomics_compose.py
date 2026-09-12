#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Compose upscaled webcomic pages onto a template, save XCF + export PDF.
#
# Called in batch mode by the repo's process.py:
#   /Applications/GIMP-2.10.app/Contents/MacOS/gimp -i \
#     -b '(python-fu-webcomics-compose RUN-NONINTERACTIVE "<in_dir>" "<template>" "<xcf_dir>" "<pdf_dir>")' \
#     -b '(gimp-quit 0)'

from gimpfu import *
import os
import sys

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _log(msg):
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def _is_image_file(fname):
    return os.path.splitext(fname)[1].lower() in IMAGE_EXTS


def _add_page_number(img, number, art_bottom, strip_bottom, bg_rgb=None):
    """Page number inside a circle, centered in the reserved strip between
    the artwork bottom and the trim line. The circle is filled with the
    page's background color; number and ring are black on light backgrounds,
    white on dark ones."""
    if bg_rgb is None:
        bg_rgb = (255, 255, 255)
    luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
    ink = (0, 0, 0) if luma >= 128 else (255, 255, 255)
    canvas_w = pdb.gimp_image_width(img)
    text = str(number)
    font = "Sans Bold"
    strip_h = max(24, strip_bottom - art_bottom)
    d = min(64, strip_h - 8)
    size = max(12, int(d * 0.45))
    tw, th, asc, desc = pdb.gimp_text_get_extents_fontname(text, size, 0, font)
    while tw > d - 12 and size > 10:  # shrink until the number fits the ring
        size -= 2
        tw, th, asc, desc = pdb.gimp_text_get_extents_fontname(text, size, 0, font)
    cx = canvas_w // 2
    cy = (art_bottom + strip_bottom) // 2

    badge = pdb.gimp_layer_new(img, img.width, img.height, RGBA_IMAGE,
                               "Page number", 100, LAYER_MODE_NORMAL)
    pdb.gimp_image_insert_layer(img, badge, None, 0)
    pdb.gimp_image_set_active_layer(img, badge)
    old_fg = pdb.gimp_context_get_foreground()
    pdb.gimp_image_select_ellipse(img, CHANNEL_OP_REPLACE,
                                  cx - d // 2, cy - d // 2, d, d)
    pdb.gimp_context_set_foreground(ink)
    pdb.gimp_edit_fill(badge, FILL_FOREGROUND)
    pdb.gimp_selection_shrink(img, 3)
    pdb.gimp_context_set_foreground(tuple(bg_rgb))
    pdb.gimp_edit_fill(badge, FILL_FOREGROUND)
    pdb.gimp_selection_none(img)

    text_layer = pdb.gimp_text_layer_new(img, text, font, size, UNIT_PIXEL)
    pdb.gimp_image_insert_layer(img, text_layer, None, 0)
    pdb.gimp_text_layer_set_color(text_layer, ink)
    pdb.gimp_layer_set_offsets(text_layer,
                               cx - text_layer.width // 2,
                               cy - text_layer.height // 2)
    merged = pdb.gimp_image_merge_down(img, text_layer, EXPAND_AS_NECESSARY)
    pdb.gimp_item_set_name(merged, "Page number")
    pdb.gimp_context_set_foreground(old_fg)
    return merged


def _compose_one(in_path, template_xcf, xcf_path, pdf_path,
                 number=None, art_bottom=0, strip_bottom=0, bg_rgb=None):
    # 1) Load a fresh copy of the template
    img = pdb.gimp_file_load(template_xcf, template_xcf)

    canvas_w = pdb.gimp_image_width(img)
    canvas_h = pdb.gimp_image_height(img)

    # 2) Hide every template layer
    for layer in img.layers:
        pdb.gimp_item_set_visible(layer, False)

    # 3) Load the page image as a new top layer
    layer = pdb.gimp_file_load_layer(img, in_path)
    pdb.gimp_image_insert_layer(img, layer, None, 0)
    pdb.gimp_item_set_visible(layer, True)

    # 4) Scale to fit the full canvas, keeping proportions, centered. Never
    #    upscale: pages arrive pre-sized to their placement size, and the
    #    narrower half of a split spread must keep the same art scale as its
    #    sibling page rather than being blown up to fill.
    scale = min(float(canvas_w) / layer.width, float(canvas_h) / layer.height)
    if scale > 1.0:
        scale = 1.0
    new_w = int(round(layer.width * scale))
    new_h = int(round(layer.height * scale))
    pdb.gimp_layer_scale(layer, new_w, new_h, False)
    # always exactly centered; numbered pages arrive pre-cropped and pre-sized
    # (story box) so the number strip below stays free
    off_x = (canvas_w - new_w) // 2
    off_y = (canvas_h - new_h) // 2
    pdb.gimp_layer_set_offsets(layer, off_x, off_y)

    # 5) Background layer in the page's paper color, so margins blend with
    #    the page: white for white pages, beige for beige ones, dark for
    #    full-bleed covers. The color comes from the metadata file (measured
    #    on the source image before cropping); fall back to measuring the
    #    placed image's border ring when absent.
    if bg_rgb is None:
        ring = max(2, int(round(min(new_w, new_h) * 0.02)))
        pdb.gimp_image_select_rectangle(img, CHANNEL_OP_REPLACE,
                                        off_x, off_y, new_w, new_h)
        pdb.gimp_image_select_rectangle(img, CHANNEL_OP_SUBTRACT,
                                        off_x + ring, off_y + ring,
                                        new_w - 2 * ring, new_h - 2 * ring)
        bg_rgb = []
        for channel in (1, 2, 3):  # HISTOGRAM_RED, _GREEN, _BLUE
            # this GIMP build returns histogram values on the 0..255 scale
            median = pdb.gimp_drawable_histogram(layer, channel, 0.0, 1.0)[2]
            bg_rgb.append(int(median + 0.5))
        pdb.gimp_selection_none(img)

    bg = pdb.gimp_layer_new(img, canvas_w, canvas_h, RGB_IMAGE,
                            "Background", 100, LAYER_MODE_NORMAL)
    pdb.gimp_image_insert_layer(img, bg, None, len(img.layers))
    old_fg = pdb.gimp_context_get_foreground()
    pdb.gimp_context_set_foreground(tuple(bg_rgb))
    pdb.gimp_drawable_fill(bg, FILL_FOREGROUND)
    pdb.gimp_context_set_foreground(old_fg)
    sys.stdout.write("  background RGB %d,%d,%d\n" % tuple(bg_rgb))
    sys.stdout.flush()

    # Page number badge (story pages only; covers/indexes/art pages get none)
    if number is not None:
        _add_page_number(img, number, art_bottom, strip_bottom, bg_rgb)

    # 5) Save XCF first (keeps all layers, hidden template ones included)
    pdb.gimp_xcf_save(0, img, layer, xcf_path, xcf_path)

    # 6) Flatten AFTER the XCF is saved, then export the PDF — the PDF gets a
    #    single composed image while the XCF keeps its layers
    flat = pdb.gimp_image_flatten(img)
    pdb.file_pdf_save(img, flat, pdf_path, pdf_path, False, True, True)

    pdb.gimp_image_delete(img)


def webcomics_compose(in_dir, template_xcf, xcf_dir, pdf_dir,
                         numbers_file=""):
    if not os.path.isdir(in_dir):
        _log("ERROR: input folder not found: %s" % in_dir)
        return
    if not os.path.isfile(template_xcf):
        _log("ERROR: template not found: %s" % template_xcf)
        return

    # numbers file: header lines "ART_BOTTOM <y>" and "STRIP_BOTTOM <y>",
    # then "<filename>\t<number or ->\t<r,g,b or ->" per page (number: only
    # pages that get one; r,g,b: paper color measured from the source image)
    numbers = {}
    bg_colors = {}
    art_bottom = strip_bottom = 0
    if numbers_file and os.path.isfile(numbers_file):
        with open(numbers_file) as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("ART_BOTTOM"):
                    art_bottom = int(line.split()[1])
                elif line.startswith("STRIP_BOTTOM"):
                    strip_bottom = int(line.split()[1])
                elif "\t" in line:
                    fname, num, rgb = line.split("\t")
                    if num != "-":
                        numbers[fname] = int(num)
                    if rgb != "-":
                        bg_colors[fname] = tuple(int(v) for v in rgb.split(","))
    for d in (xcf_dir, pdf_dir):
        if not os.path.isdir(d):
            os.makedirs(d)

    files = sorted([f for f in os.listdir(in_dir) if _is_image_file(f)])
    if not files:
        _log("No image files found in: %s" % in_dir)
        return

    done = 0
    skipped = 0
    failed = 0
    for i, fname in enumerate(files):
        base = os.path.splitext(fname)[0]
        in_path = os.path.join(in_dir, fname)
        xcf_path = os.path.join(xcf_dir, base + ".xcf")
        pdf_path = os.path.join(pdf_dir, base + ".pdf")

        if (os.path.isfile(xcf_path) and os.path.isfile(pdf_path)
                and os.path.getmtime(xcf_path) >= os.path.getmtime(in_path)
                and os.path.getmtime(xcf_path) >= os.path.getmtime(template_xcf)):
            skipped += 1
            continue

        _log("COMPOSE %d/%d: %s" % (i + 1, len(files), fname))
        try:
            _compose_one(in_path, template_xcf, xcf_path, pdf_path,
                         numbers.get(fname), art_bottom, strip_bottom,
                         bg_colors.get(fname))
            done += 1
        except Exception as e:
            failed += 1
            _log("ERROR composing %s: %s" % (fname, str(e)))

    _log("COMPOSE DONE: %d composed, %d skipped (already done), %d failed" % (done, skipped, failed))


register(
    "python-fu-webcomics-compose",
    "Place webcomic pages on template, save XCF and export PDF",
    "For each image in in_dir: copy template_xcf, fit image to canvas keeping proportions, hide template layers, save XCF to xcf_dir and PDF to pdf_dir.",
    "Adrian Acerete",
    "Adrian Acerete",
    "2026",
    "<Toolbox>/Xtns/Batch/Webcomics - Compose Template",
    "",
    [
        (PF_STRING, "in_dir", "Folder with (upscaled) page images", ""),
        (PF_STRING, "template_xcf", "Template .xcf file", ""),
        (PF_STRING, "xcf_dir", "Output folder for .xcf files", ""),
        (PF_STRING, "pdf_dir", "Output folder for .pdf files", ""),
        (PF_STRING, "numbers_file", "Page-number mapping file (optional)", ""),
    ],
    [],
    webcomics_compose
)

main()
