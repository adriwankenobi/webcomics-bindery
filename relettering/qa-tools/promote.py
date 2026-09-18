"""Promote ONLY the pages the re-fit actually changed.

Scope discipline (ARCHITECTURE.md): pages already signed off must keep
their entries verbatim, so a page moves only when its layout entries or its
cleaned pixels genuinely differ. Everything else keeps its old mtime, which
is what stops compose from rebuilding (and blanking) the whole book.
DRY RUN unless APPLY=1."""
import os, json, os, shutil, sys, unicodedata
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: promote.py "<comic folder name>"')
COMIC = sys.argv[1]
C=COMIC; R=f'relettering/{C}'
SP=os.environ.get('SP', '/tmp/relettering-qa')
APPLY = os.environ.get('APPLY')=='1'
# BASE  = the snapshot every already-approved page must keep verbatim
# NEWLAY= the layout the re-fit produced (refit_subset writes work3/layout.json)
# STAGE = the cleaned pages that re-fit produced alongside it
BASE=os.environ.get('BASE', f'{R}/layout.json.bak-round3')
NEWLAY=os.environ.get('NEWLAY', f'{R}/layout.json')
STAGE=os.environ.get('STAGE', f'{SP}/stage')
old=json.load(open(BASE))
new=json.load(open(NEWLAY))
# a subset re-fit only holds the pages it was given; everything else must
# fall through to the snapshot untouched
new={k:v for k,v in new.items()}
nums={os.path.splitext(k)[0]:v for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
stems=sorted(set(old)|set(new))
changed, merged = [], {}
for stem in stems:
    o, n = old.get(stem), new.get(stem)
    if n is None:                 # not in this subset re-fit — keep verbatim
        merged[stem] = o
        continue
    lay_diff = (json.dumps(o, sort_keys=True) != json.dumps(n, sort_keys=True))
    st=f'{STAGE}/{stem}.jpg'; up=f'upscaled/{C}/{stem}.jpg'
    pix_diff=False
    if os.path.exists(st) and os.path.exists(up):
        a=cv2.imread(st); b=cv2.imread(up)
        pix_diff = a is None or b is None or a.shape!=b.shape or \
                   int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max())>12
    if lay_diff or pix_diff:
        changed.append((nums.get(stem,'-'),stem,lay_diff,pix_diff))
        merged[stem]=n if n is not None else o
    else:
        merged[stem]=o          # verbatim: untouched page keeps its entries
changed.sort(key=lambda r: (r[0] if isinstance(r[0],int) else 9999))
print(f'{len(changed)} of {len(stems)} pages changed\n')
print(f'{"page":>5}  layout  pixels  stem')
for pg,stem,l,p in changed:
    print(f'{str(pg):>5}  {"YES" if l else " - ":>6}  {"YES" if p else " - ":>6}  {stem[-30:]}')
if APPLY:
    json.dump(merged, open(f'{R}/layout.json','w'), indent=1)
    for pg,stem,l,p in changed:
        src=f'{STAGE}/{stem}.jpg'
        if os.path.exists(src):
            shutil.copyfile(src, f'upscaled/{C}/{stem}.jpg')   # new mtime -> recompose
    print(f'\nAPPLIED: layout.json merged, {len(changed)} working pages replaced')
else:
    print('\n(dry run — set APPLY=1 to promote)')
json.dump([[c[0],c[1]] for c in changed], open(f'{SP}/changed.json','w'))
