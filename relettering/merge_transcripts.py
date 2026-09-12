#!/usr/bin/env python3
"""Assemble transcripts.json from per-sheet parts + manifest; report gaps."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if len(sys.argv) < 2:
    sys.exit('usage: merge_transcripts.py "<comic folder name>"')
COMIC = sys.argv[1]
WORK = REPO / "relettering" / COMIC
manifest = json.loads((WORK / "sheets" / "manifest.json").read_text())

flat = {}
for part in sorted((WORK / "parts").glob("*.json")):
    flat.update(json.loads(part.read_text()))

out, missing = {}, []
for key, info in manifest.items():
    texts = []
    partial = False
    for bi in range(1, info["count"] + 1):
        t = flat.get(f"{key} b{bi:02d}")
        if t is None:
            missing.append(f"{key} b{bi:02d}")
            partial = True
        texts.append(t if t is not None else "")
    if not partial:
        out[info["stem"]] = texts
(WORK / "transcripts.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=0))
done = sum(1 for v in flat.values() if v is not None)
print(f"{done} transcribed, {len(missing)} missing")
for m in missing[:20]:
    print("  missing:", m)
