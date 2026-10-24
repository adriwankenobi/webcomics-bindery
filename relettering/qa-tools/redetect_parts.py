"""Re-detect pages of a comic that is TRANSCRIBED BUT NOT FITTED (parts/ +
sheets/manifest.json, no layout.json) and carry the transcripts across a
bubble-COUNT change. redetect_dry/redetect_apply refuse a count change
(post-fit: positional transcripts.json); before the fit the keys can be
remapped, so a detection fix that ADDS entries (a missed box) or replaces
fragments (a box found as strips) can still land.

usage (repo root):
  SP=<scratch> .venv-reletter/bin/python relettering/qa-tools/redetect_parts.py \
      "<comic>" <short key or stem prefix>... [--apply]

Dry run (default): detection into $SP/redetect/<stem>/, a per-page report.
Each new entry takes the text of the old entries that lie mostly inside it
(>= half of the old bbox); several non-empty texts = MERGED (joined in old
order, listed for review); an old entry with text no new one takes = GONE
(listed — that text is lost unless handled); a new entry nothing maps to =
NEW (no key: the driver reports it as a transcription gap).
--apply: backs up bubbles/<stem>* and the manifest to $SP/redetect-backup/,
installs the new detection, updates the manifest count, and writes the
remapped keys of every re-detected page to parts/zz-redetect.json (sorted
last, so it wins over the sheet parts). Refused on a fitted comic."""
import json, os, shutil, sys
from pathlib import Path
if len(sys.argv) < 3:
    sys.exit(__doc__)
COMIC = sys.argv[1]
APPLY = "--apply" in sys.argv
WANT = [a for a in sys.argv[2:] if a != "--apply"]
sys.argv = ["reletter_detect.py", COMIC]
sys.path.insert(0, "relettering")
import numpy as np, cv2, reletter_detect as D
from make_sheets import short_key

R = Path("relettering") / COMIC
SP = Path(os.environ.get("SP", "/tmp/relettering-qa"))
if (R / "layout.json").is_file():
    sys.exit("refused: layout.json exists — the comic is fitted; use "
             "redetect_dry.py / redetect_apply.py")
man_path = R / "sheets" / "manifest.json"
manifest = json.loads(man_path.read_text())
flat = {}
for part in sorted((R / "parts").glob("*.json")):
    flat.update(json.loads(part.read_text()))

pages = [(k, v["stem"]) for k, v in manifest.items()
         if any(k.startswith(w) or v["stem"].startswith(w) for w in WANT)]
if not pages:
    sys.exit("no manifest page matches " + " ".join(WANT))


def inside(o, n):
    """Fraction of old bbox o inside new bbox n."""
    ox, oy, ow, oh = o; nx, ny, nw, nh = n
    ix = max(0, min(ox + ow, nx + nw) - max(ox, nx))
    iy = max(0, min(oy + oh, ny + nh) - max(oy, ny))
    return ix * iy / max(1, ow * oh)


remap, review, installs = {}, [], []
for key, stem in pages:
    old = json.loads((R / "bubbles" / f"{stem}.json").read_text())
    img, new = D.detect_page(R / "pristine" / f"{stem}.jpg")
    out = SP / "redetect" / stem
    out.mkdir(parents=True, exist_ok=True)
    for bi, b in enumerate(new, 1):
        x, y, bw, bh = b["bbox"]
        mask = b.pop("_mask")
        cv2.imwrite(str(out / f"{stem}-b{bi:02d}-mask.png"),
                    (mask * 255).astype(np.uint8))
        crop = img[max(0, y - 8):y + bh + 8, max(0, x - 8):x + bw + 8]
        crop2 = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2),
                           interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(out / f"{stem}-b{bi:02d}.png"),
                    cv2.cvtColor(crop2, cv2.COLOR_RGB2BGR))
    (out / f"{stem}.json").write_text(json.dumps(new, indent=1))
    sk = short_key(stem)
    texts = [flat.get(f"{sk} b{i:02d}") for i in range(1, len(old) + 1)]
    def iou(a, b):
        i = inside(a, b) * a[2] * a[3]
        return i / max(1, a[2] * a[3] + b[2] * b[3] - i)

    owner = {}                       # old index -> new index
    for i, o in enumerate(old):
        # most of the old box inside the new one; among nested candidates
        # (a small balloon inside a big entry's bbox) the closest in shape
        best = max(range(len(new)),
                   key=lambda j: (inside(o["bbox"], new[j]["bbox"]) >= 0.5,
                                  iou(o["bbox"], new[j]["bbox"])),
                   default=None)
        if best is not None and inside(o["bbox"], new[best]["bbox"]) >= 0.5:
            owner[i] = best
    # same keys is not same detection: an entry can keep its index and turn
    # into a caption box (a strip rebuilt whole, another kind, a new bbox)
    same = (len(old) == len(new)
            and all(owner.get(i) == i for i in range(len(old)))
            and all((o["kind"], o["bbox"], o["block"])
                    == (n["kind"], n["bbox"], n["block"])
                    for o, n in zip(old, new)))
    lines = []
    for j, n in enumerate(new):
        src = [i for i in range(len(old)) if owner.get(i) == j]
        vals = [texts[i] for i in src if texts[i]]
        if not src:
            lines.append(f"  b{j+1:02d} NEW {n['kind']} {n['bbox']}")
            continue
        # a box rebuilt from several entries, or from an entry of another
        # kind (a strip seeded on a few of its letters), is a NEW reading of
        # the page: empty sources mean nobody transcribed the box (the strip
        # was "" as a fragment), so it gets no key and is transcribed fresh
        rebuilt = len(src) > 1 or all(old[i]["kind"] != n["kind"] for i in src)
        if rebuilt and not vals:
            lines.append(f"  b{j+1:02d} NEW {n['kind']} {n['bbox']} (rebuilt from "
                         + ",".join(f"b{i+1:02d}" for i in src) + ", all empty)")
            continue
        # strips of a caption box are ONE block of type (a space); merged
        # balloons are paragraphs, one per lobe
        sep = " " if n["kind"] in ("dark", "tint") else "\n\n"
        remap[f"{sk} b{j+1:02d}"] = sep.join(vals) if len(vals) > 1 else (
            vals[0] if vals else "")
        if rebuilt:
            review.append(f"{sk} b{j+1:02d}: REBUILT {n['kind']} from "
                          + ",".join(f"b{i+1:02d}" for i in src)
                          + f" — check {remap[f'{sk} b{j+1:02d}']!r}")
        tag = "MERGED" if len(vals) > 1 else ("" if src == [j] else "moved")
        if tag:
            lines.append(f"  b{j+1:02d} <- " + ",".join(f"b{i+1:02d}" for i in src)
                         + f" {tag} {n['kind']} {remap[f'{sk} b{j+1:02d}']!r}")
        if len(vals) > 1:
            review.append(f"{sk} b{j+1:02d}: MERGED {vals}")
    for i in range(len(old)):
        if i not in owner:
            lines.append(f"  old b{i+1:02d} GONE {old[i]['kind']} {texts[i]!r}")
            if texts[i]:
                review.append(f"{sk} old b{i+1:02d}: GONE {texts[i]!r}")
    new_keys = sum(1 for l in lines if " NEW " in l)
    print(f"--- {sk} ({stem}): {len(old)} -> {len(new)}"
          + (" unchanged" if same else f", {new_keys} new"))
    for l in lines:
        print(l)
    if not same:
        installs.append((key, stem, len(new)))

print(f"\n{len(installs)} of {len(pages)} pages change; review:")
for r in review:
    print("  " + r)
if not APPLY:
    print(f"\ndry run — nothing written (detection in {SP / 'redetect'})")
    sys.exit(0)

bk = SP / "redetect-backup"
bk.mkdir(parents=True, exist_ok=True)
if not (bk / "manifest.json").exists():
    shutil.copy2(man_path, bk / "manifest.json")
zz = R / "parts" / "zz-redetect.json"
done = json.loads(zz.read_text()) if zz.is_file() else {}
for key, stem, count in installs:
    for f in (R / "bubbles").glob(f"{stem}*"):
        if f.name == f"{stem}.json" or f.name.startswith(f"{stem}-b"):
            if not (bk / f.name).exists():
                shutil.copy2(f, bk / f.name)
            f.unlink()
    for f in (SP / "redetect" / stem).iterdir():
        shutil.copy2(f, R / "bubbles" / f.name)
    manifest[key]["count"] = count
    sk = short_key(stem)
    for k in [k for k in done if k.startswith(sk + " b")]:
        del done[k]
    done.update({k: v for k, v in remap.items() if k.startswith(sk + " b")})
# the sheet parts still hold the OLD numbering of these pages: a NEW index
# would find an old text under its key and count as transcribed (presence is
# what the driver checks), so every key of a re-detected page leaves them
gone = tuple(short_key(stem) + " b" for _, stem, _ in installs)
for part in sorted((R / "parts").glob("*.json")):
    if part == zz:
        continue
    d = json.loads(part.read_text())
    keep = {k: v for k, v in d.items() if not k.startswith(gone)}
    if len(keep) != len(d):
        if not (bk / f"parts-{part.name}").exists():
            shutil.copy2(part, bk / f"parts-{part.name}")
        part.write_text(json.dumps(keep, ensure_ascii=False, indent=1))
man_path.write_text(json.dumps(manifest, indent=1))
zz.write_text(json.dumps(done, ensure_ascii=False, indent=1))
print(f"applied {len(installs)} pages; manifest updated; {zz} ({len(done)} keys);"
      f" backup in {bk}")
