"""Re-fit a subset of DRAWN page numbers into a scratch WORK dir (book size
pinned, so the fit is separable per page). Fast loop for validating fixes."""
import json, os, shutil, sys, unicodedata
from pathlib import Path
SP=os.environ.get('SP', '/tmp/relettering-qa')
if len(sys.argv) < 2:
    sys.exit('usage: refit_subset.py "<comic folder name>"')
COMIC = sys.argv[1]
BOOK = os.environ.get('BOOK')
if not BOOK:
    sys.exit('set BOOK=<book size> — an UNPINNED re-fit re-derives it and moves approved pages')
C=COMIC
pages=[int(a) for a in os.environ['PGS'].split(',')]
nums=json.load(open(f'xcf/{C}/numbers.json'))
inv={v:unicodedata.normalize('NFC',os.path.splitext(k)[0]) for k,v in nums.items() if v!='-'}
want={inv[p] for p in pages if p in inv}
stage=Path(SP)/'stage3'; work=Path(SP)/'work3'
shutil.rmtree(stage,ignore_errors=True); shutil.rmtree(work,ignore_errors=True)
stage.mkdir(parents=True); work.mkdir(parents=True)
src=Path(f'relettering/{C}/pristine')
picked=[q for q in sorted(src.glob('*.jpg'))
        if unicodedata.normalize('NFC',q.stem) in want]
for q in picked: shutil.copy2(q, stage/q.name)   # REAL filesystem names
shutil.copy2(f'relettering/{C}/transcripts.json', work/'transcripts.json')
print(f'staged {len(picked)}/{len(want)} pages', flush=True)
sys.argv=["reletter_fit.py",C,"--book",BOOK]
sys.path.insert(0,"relettering")
import reletter_fit as F
F.PAGES_DIR=stage; F.WORK=work
try:
    F.main()
except ValueError as e:
    print('(final report crashed, harmless):', e)
print('layout stems:', len(json.load(open(work/'layout.json'))))
