"""Apply re-detection for given drawn pages: writes bubbles JSON + masks +
crops, after backing up the originals. REFUSES a page whose bubble count
changed, or whose entries moved place at the SAME count (bNN's box no longer
mostly bNN's: balloons regrouped, so the order shifted) — either would hand
the positional transcripts to the wrong balloons."""
import os, json, os, shutil, sys, time, unicodedata
if len(sys.argv) < 2:
    sys.exit('usage: redetect_apply.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.argv=["reletter_detect.py",COMIC]; sys.path.insert(0,"relettering")
from pathlib import Path
import numpy as np, cv2, reletter_detect as D
C=COMIC; R=f'relettering/{C}'
SP=os.environ.get('SP', '/tmp/relettering-qa')
BK=Path(SP)/'bubbles_backup'; BK.mkdir(exist_ok=True)
nums=json.load(open(f'xcf/{C}/numbers.json'))
inv={v:unicodedata.normalize('NFC',os.path.splitext(k)[0]) for k,v in nums.items() if v!='-'}
_p=os.environ['PGS']
pages=(sorted(inv) if _p=='ALL' else [int(a) for a in _p.split(',')])
src=Path(f'{R}/pristine'); OUT=Path(f'{R}/bubbles')
applied=refused=0; t0=time.time(); done=0


def iou(a, b):
    ix=max(0,min(a[0]+a[2],b[0]+b[2])-max(a[0],b[0]))
    iy=max(0,min(a[1]+a[3],b[1]+b[3])-max(a[1],b[1]))
    return ix*iy/max(1,a[2]*a[3]+b[2]*b[3]-ix*iy)


for pg in pages:
    stem=inv.get(pg)
    if not stem: continue
    real=[q for q in src.glob('*.jpg') if unicodedata.normalize('NFC',q.stem)==stem]
    if not real: continue
    path=real[0]
    old=json.load(open(f'{R}/bubbles/{path.stem}.json'))
    img,new=D.detect_page(path)
    if len(new)!=len(old):
        print(f'p{pg}: REFUSED — count {len(old)} -> {len(new)}'); refused+=1; continue
    moved=[i+1 for i,(o,e) in enumerate(zip(old,new)) if iou(o['bbox'],e['bbox'])<0.5]
    if moved:
        print(f'p{pg}: REFUSED — same count, entries moved: '
              +', '.join(f'b{i:02d}' for i in moved)); refused+=1; continue
    # back up once
    for f in list(OUT.glob(f'{path.stem}.json'))+list(OUT.glob(f'{path.stem}-b*.png')):
        t=BK/f.name
        if not t.exists(): shutil.copy2(f,t)
    for bi,b in enumerate(new,1):
        x,y,bw,bh=b['bbox']; mask=b.pop('_mask')
        cv2.imwrite(str(OUT/f'{path.stem}-b{bi:02d}-mask.png'),(mask*255).astype(np.uint8))
        crop=img[max(0,y-8):y+bh+8, max(0,x-8):x+bw+8]
        crop2=cv2.resize(crop,(crop.shape[1]*2,crop.shape[0]*2),interpolation=cv2.INTER_LANCZOS4)
        cv2.imwrite(str(OUT/f'{path.stem}-b{bi:02d}.png'),cv2.cvtColor(crop2,cv2.COLOR_RGB2BGR))
    (OUT/f'{path.stem}.json').write_text(json.dumps(new,indent=1))
    applied+=1
    done+=1
    if done%10==0:
        el=time.time()-t0
        print(f'  [{done}/{len(pages)}] {el/60:.0f} min, ~{el/done*(len(pages)-done)/60:.0f} min left',flush=True)
print(f'\napplied={applied} refused={refused}  (originals backed up in {BK})')
