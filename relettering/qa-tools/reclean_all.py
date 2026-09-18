"""Re-clean every page from pristine/ with the CURRENT cleaning into a temp
dir. Does not touch upscaled/ — pure verification.

This reproduces the pipeline by CALLING it: page_caption_boxes,
boxes_to_wipe and prepare_bubble are the very functions reletter_fit.main()
runs, so this tool can no longer drift from what the fit actually does. It
used to hand-mirror that preparation — four "mirror main()" blocks by the
end — and every new cleaning behaviour (the `lobes` override, sole_box,
unfound_balloon) had to be copied in here by hand or the verification
quietly stopped reproducing the real pipeline.

    OUTDIR=<dir> .venv-reletter/bin/python \\
        relettering/qa-tools/reclean_all.py "<comic>"
"""
import json
import os
import sys
import time
import unicodedata

if len(sys.argv) < 2:
    sys.exit('usage: OUTDIR=<dir> reclean_all.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.argv = ["reletter_fit.py", COMIC]      # reletter_fit reads the comic here
sys.path.insert(0, os.path.abspath("relettering"))
import cv2                                                   # noqa: E402
import reletter_fit as F                                     # noqa: E402

R = f"relettering/{COMIC}"
OUT = os.environ["OUTDIR"]
os.makedirs(OUT, exist_ok=True)


def nkey(d):
    return {unicodedata.normalize("NFC", k): v for k, v in d.items()}


TR = nkey(json.load(open(f"{R}/transcripts.json")))
LAY = nkey(json.load(open(os.environ.get("LAYOUT", f"{R}/layout.json"))))
stems = sorted(TR)
t0 = time.time()
for si, stem in enumerate(stems, 1):
    src = f"{R}/pristine/{stem}.jpg"
    if not os.path.exists(src):
        continue
    img = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)
    bubs = json.load(open(f"{R}/bubbles/{stem}.json"))
    tr = TR[stem]
    # the pages the fit actually typeset: a layout entry AND text to set
    fitted = {e["index"] for e in LAY.get(stem, [])
              if (tr[e["index"] - 1] if e["index"] - 1 < len(tr) else "")}
    cap_box, cap_mem, _ = F.page_caption_boxes(bubs, stem)
    boxes = F.boxes_to_wipe(bubs, stem, fitted)
    in_box = {i for _, mem in boxes for i in mem}
    for bi, b in enumerate(bubs, 1):
        if bi not in fitted:
            continue
        text = F.apply_fixes(tr[bi - 1] if bi - 1 < len(tr) else "")
        if not text.strip():
            continue
        prep = F.prepare_bubble(stem, bi, b, img, tr, bubs, cap_box, cap_mem)
        if prep is None:
            continue            # no mask on disk
        for jbi, jb, jmask, jallowed in prep["jobs"]:
            if jbi in fitted and jbi not in in_box:
                F.clean_bubble(img, jb, jmask, jallowed)
    for box, _ in boxes:
        F.clean_caption_box(img, box)
    cv2.imwrite(f"{OUT}/{stem}.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95,
                 int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR),
                 cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])
    if si % 25 == 0 or si == len(stems):
        el = time.time() - t0
        print(f"  {si}/{len(stems)} ({100 * si // len(stems)}%) "
              f"{el / 60:.1f} min, ~{el / si * (len(stems) - si) / 60:.1f} "
              f"min left", flush=True)
print("DONE")
