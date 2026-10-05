"""BOOK-WIDE faint-residue scan — the gap in leftover.py.
Counts letter-shaped specks left in cleaned bubbles that detection's
letter_mask cannot see (they sit above the ink cut-off)."""
import os, json, os, sys, unicodedata
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: faint_scan.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.path.insert(0,'relettering'); sys.argv=['reletter_fit.py',COMIC]
import reletter_fit as F
C=COMIC; R=f'relettering/{C}'

# A `lobes` override REPLACES detection's mask for the fit and the cleaner
# (relettering/<comic>/layout_overrides.json). A gate that reads the mask
# PNG instead is measuring a region nothing ever cleaned: p276 b04's
# detection mask covers 79% of the panel, and letter_mask found 16
# letter-shaped pieces of ARTWORK in it.
sys.argv = ["reletter_fit.py", C]
import reletter_fit as _RF


def gate_mask(img, stem, bi, b, default, pristine=None, texts=None,
              bubbles=None):
    # the one home for the rule is reletter_fit.gate_mask (the `lobes`
    # override AND the balloons letter_lobes re-finds from the lettering)
    return _RF.gate_mask(img, stem, bi, b, default, pristine, texts, bubbles)

SP=os.environ.get('SP', '/tmp/relettering-qa')
def nk(d): return {unicodedata.normalize('NFC',k):v for k,v in d.items()}
nums={unicodedata.normalize('NFC',os.path.splitext(k)[0]):v for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
LAY=nk(json.load(open(os.environ.get('LAYOUT', f'{R}/layout.json'))))
TR=nk(json.load(open(f'{R}/transcripts.json')))
SRC=os.environ.get('PAGES',f'upscaled/{C}')
rows=[]
for stem in sorted(LAY):
    p=f'{SRC}/{stem}.jpg'
    if not os.path.exists(p): continue
    up=cv2.cvtColor(cv2.imread(p),cv2.COLOR_BGR2RGB)
    bubs=json.load(open(f'{R}/bubbles/{stem}.json'))
    laid={e['index'] for e in LAY[stem]}
    for bi,b in enumerate(bubs,1):
        if bi not in laid or b['kind'] not in ('bubble','tint','dark'): continue
        mp=f'{R}/bubbles/{stem}-b{bi:02d}-mask.png'
        if not os.path.exists(mp): continue
        x,y,w,h=b['bbox']
        _pp=f'{R}/pristine/{stem}.jpg'
        _pr=(cv2.cvtColor(cv2.imread(_pp),cv2.COLOR_BGR2RGB)
             if os.path.exists(_pp) else None)
        mask=gate_mask(up, stem, bi, b,
                       cv2.imread(mp,cv2.IMREAD_GRAYSCALE)>127,
                       _pr, TR.get(stem), bubs)
        if mask.shape!=(h,w): continue
        cu=up[y:y+h,x:x+w].min(axis=2)
        inside=mask&(cu>180)
        if inside.sum()<200: continue
        fill=float(np.median(cu[inside]))
        speck=mask&(cu<=fill-22)&(cu>fill-105)   # faint only: darker is real ink/outline
        n,l,s,_=cv2.connectedComponentsWithStats(speck.astype(np.uint8),8)
        span=max(w,h); cnt=0; area=0
        for i in range(1,n):
            cw,ch,ar=s[i,2],s[i,3],s[i,4]
            if ar<8: continue
            if max(cw,ch)>=0.45*span and ar<=0.28*max(1,cw*ch): continue  # outline
            if cw>0.5*w and ch<3: continue
            cnt+=1; area+=int(ar)
        if cnt>=3: rows.append((nums.get(stem,0),stem,bi,cnt,area,b['kind']))
rows.sort(key=lambda r:-r[3])
print(f'FAINT RESIDUE: {sum(r[3] for r in rows)} specks in {len(rows)} bubbles on {len(set(r[1] for r in rows))} pages')
for r in rows[:35]: print(f'  p{r[0]:>4} b{r[2]:02d} {r[5]:6s} specks={r[3]:3d} area={r[4]:5d}')
json.dump([[r[0],r[1],r[2],r[3],r[4],r[5]] for r in rows],open(f'{SP}/faint_scan.json','w'))
