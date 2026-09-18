"""Re-fit ONE bubble with the current code and report size + line layout."""
import json, os, sys, unicodedata
if len(sys.argv) < 2:
    sys.exit('usage: refit1.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.argv=["reletter_fit.py",COMIC]; sys.path.insert(0,"relettering")
import cv2, numpy as np, reletter_fit as F
C=COMIC; R=f'relettering/{C}'
stem=unicodedata.normalize('NFC',os.environ['STEM']); bi=int(os.environ['BI'])
def nkey(d): return {unicodedata.normalize('NFC',k):v for k,v in d.items()}
TR=nkey(json.load(open(f'{R}/transcripts.json')))
img=cv2.cvtColor(cv2.imread(f'{R}/pristine/{stem}.jpg'),cv2.COLOR_BGR2RGB)
b=json.load(open(f'{R}/bubbles/{stem}.json'))[bi-1]
x,y,w,h=b['bbox']
mask=(cv2.imread(f'{R}/bubbles/{stem}-b{bi:02d}-mask.png',cv2.IMREAD_GRAYSCALE)>127).astype(np.uint8)
rows={y+r[0]:(x+r[1],x+r[2]) for r in b['rows']}
if not b.get('strict'):
    bx_,by_,bw_,bh_=b['block']; rows=F.clip_rows(rows,by_+bh_//2)
m2=cv2.erode(mask,np.ones((3,3),np.uint8)); rr={}
for ry in range(m2.shape[0]):
    xs=np.flatnonzero(m2[ry])
    if len(xs): rr[y+ry]=(x+int(xs[0]),x+int(xs[-1])+1)
for yy,sp in rows.items():
    o=rr.get(yy); rr[yy]=sp if o is None else (min(o[0],sp[0]),max(o[1],sp[1]))
text=TR[stem][bi-1]
words,hard,paras=[],set(),set()
for para in text.split("\n\n"):
    for ln in para.split("\n"):
        words+=F.parse_runs(ln)
        if words: hard.add(len(words)-1)
    if words: paras.add(len(words)-1)
if words: hard.discard(len(words)-1); paras.discard(len(words)-1)
p={"bi":bi,"b":b,"words":words,"hard":hard,"paras":paras,"rows":(F.inset_rows(rows) if b["kind"]=="bubble" else rows),
   "oldh":F.old_cap_height(img,b),"ov":F.override_for(stem,bi),
   "porig":(F.orig_para_axes(img,b,len(paras)+1) if paras else None),
   "oaxis":(F.orig_axis_map(img,b) if b['kind']=='bubble' else None),
   "rows_raw":(F.inset_rows(rr) if (b["kind"]=="bubble" and rr) else rr),
   "lobes":([F.inset_rows(L) for L in F.mask_lobes(mask,x,y,len(paras)+1)] if (b["kind"]=="bubble" and paras and F.mask_lobes(mask,x,y,len(paras)+1)) else None)}
fb=F.find_bands(rows,len(paras)+1)
lo=F.mask_lobes(mask,x,y,len(paras)+1) if paras else None
print('paragraphs:',len(paras)+1,' find_bands:',None if fb is None else len(fb),
      ' mask_lobes:',None if lo is None else len(lo))
fit=F.Fitter()
for size in range(23,9,-1):
    L=F.fit_lines(fit,p,size)
    if L:
        print(f'SIZE {size}, {len(L)} lines')
        for (y0,x0,x1,runs,ax) in L:
            print(f'   y={y0:7.1f} ax={ax if ax is None else round(ax)} '
                  + ' '.join(wd for _,wd in runs)[:62])
        break
else:
    print('NO FIT')
