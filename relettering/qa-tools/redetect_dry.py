"""DRY-RUN re-detection: run detect_page on given drawn pages and compare
bubble count/order/geometry against the current bubbles JSON. Writes
nothing. Flags any change that would invalidate the transcripts."""
import json, os, sys, unicodedata
if len(sys.argv) < 2:
    sys.exit('usage: redetect_dry.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.argv=["reletter_detect.py",COMIC]; sys.path.insert(0,"relettering")
from pathlib import Path
import numpy as np, cv2, reletter_detect as D
C=COMIC; R=f'relettering/{C}'
nums=json.load(open(f'xcf/{C}/numbers.json'))
inv={v:unicodedata.normalize('NFC',os.path.splitext(k)[0]) for k,v in nums.items() if v!='-'}
TR={unicodedata.normalize('NFC',k):v for k,v in json.load(open(f'{R}/transcripts.json')).items()}
pages=[int(a) for a in os.environ['PGS'].split(',')]
src=Path(f'{R}/pristine')
for pg in pages:
    stem=inv.get(pg)
    if not stem: continue
    real=[q for q in src.glob('*.jpg') if unicodedata.normalize('NFC',q.stem)==stem]
    if not real: continue
    old=json.load(open(f'{R}/bubbles/{stem}.json'))
    img,new=D.detect_page(real[0])
    same_n = len(old)==len(new)
    print(f'--- p{pg} {stem[-3:]}: old={len(old)} new={len(new)} '
          f'{"COUNT OK" if same_n else "!! COUNT CHANGED (transcripts need remap)"}')
    n=max(len(old),len(new))
    for i in range(n):
        o=old[i] if i<len(old) else None
        e=new[i] if i<len(new) else None
        if o is None or e is None:
            print(f'    b{i+1:02d}: {"removed" if e is None else "added"}')
            continue
        ob,eb=o['bbox'],e['bbox']
        dk = o['kind']!=e['kind']
        dblk = o['block']!=e['block']
        # does the mask cover the text block rows?
        def cov(ent, mask):
            bx,by,bw,bh=ent['block']; x,y=ent['bbox'][:2]
            return sum(1 for yy in range(by,by+bh)
                       if 0<=yy-y<mask.shape[0]
                       and mask[yy-y, max(0,bx-x):bx-x+bw].sum()>=0.5*bw)
        mo=(cv2.imread(f'{R}/bubbles/{stem}-b{i+1:02d}-mask.png',cv2.IMREAD_GRAYSCALE)>127).astype(np.uint8)
        mn=e['_mask']
        co=cov(o,mo); cn=cov(e,mn); bh=o['block'][3]
        flag=''
        if dk: flag+=' KIND-CHANGED'
        if dblk: flag+=' BLOCK-MOVED'
        if cn>co: flag+='  <-- coverage improved'
        if cn<co: flag+='  <-- coverage WORSE'
        t=TR.get(stem,[])
        has=bool(t[i]) if i<len(t) else False
        print(f'    b{i+1:02d} {o["kind"]:6s} bbox {ob} -> {eb}  blockrows {co}/{bh} -> {cn}/{e["block"][3]}'
              f'{" [has text]" if has else ""}{flag}')
