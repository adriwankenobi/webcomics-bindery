"""LEFTOVER SCAN (the check CLAUDE.md requires): run detection's own
letter_mask on the CLEANED working pages. A letter blob still sitting where
the new text lands, in a bubble whose transcript has text = failed cleaning.

The search region is the MASK **union the text block union every laid-out
line's box**. Intersecting with the mask alone — which this did until round 4
— hides the worst failure there is: lettering the mask never covered is
exactly what produces "text on top of each other", and the scan called the
book healthy (14 blobs / 5 pages) while p170 and p147 carried whole lines of
old text under the new. Widening the region found 42 blobs on 19 pages.

Still blind to: text on a COLOURED fill (letter_mask is a "dark text on light
paper" ring test, which is why those boxes were mis-detected in the first
place), and to faint residue above the ink cut-off. Look at the pages."""
import os, json, os, sys, unicodedata
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: leftover.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.argv=["reletter_detect.py",COMIC]; sys.path.insert(0,"relettering")
import reletter_detect as D
C=COMIC; R=f'relettering/{C}'

# A `lobes` override REPLACES detection's mask for the fit and the cleaner
# (relettering/<comic>/layout_overrides.json). A gate that reads the mask
# PNG instead is measuring a region nothing ever cleaned: p276 b04's
# detection mask covers 79% of the panel, and letter_mask found 16
# letter-shaped pieces of ARTWORK in it.
sys.argv = ["reletter_fit.py", C]
import reletter_fit as _RF


def gate_mask(img, stem, bi, b, default):
    ov = _RF.override_for(stem, bi)
    if ov and "lobes" in ov:
        got = _RF.lobes_from_rects(img, ov["lobes"], b["bbox"])
        if got is not None:
            return got[0].astype(bool)
    return default

SP=os.environ.get('SP', '/tmp/relettering-qa')
def nk(d): return {unicodedata.normalize('NFC',k):v for k,v in d.items()}
nums={unicodedata.normalize('NFC',os.path.splitext(k)[0]):v
      for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
TR=nk(json.load(open(f'{R}/transcripts.json')))
LAY=nk(json.load(open(os.environ.get('LAYOUT', f'{R}/layout.json'))))
SRC=os.environ.get('PAGES', f'upscaled/{C}')
rows=[]
for stem in sorted(LAY):
    p=f'{SRC}/{stem}.jpg'
    if not os.path.exists(p): continue
    img=cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    lm=D.letter_mask(img)
    bubs=json.load(open(f'{R}/bubbles/{stem}.json'))
    tr=TR[stem]; laid={e['index'] for e in LAY[stem]}
    ents={e['index']:e for e in LAY[stem]}
    H,W=lm.shape
    for bi,b in enumerate(bubs,1):
        t=tr[bi-1] if bi-1<len(tr) else ''
        if not t or bi not in laid: continue
        x,y,w,h=b['bbox']
        reg=np.zeros((H,W),np.uint8)
        mp=f'{R}/bubbles/{stem}-b{bi:02d}-mask.png'
        if os.path.exists(mp):
            m=cv2.imread(mp,cv2.IMREAD_GRAYSCALE)
            if m is not None and m.shape==(h,w):
                mm=gate_mask(img, stem, bi, b, m>127)
                reg[y:y+h,x:x+w]|=mm.astype(np.uint8)
        # the block CLIPPED to the bubble's own bbox: a block swallows the
        # balloon next door, and that neighbour's lettering is deliberately
        # KEPT (it has no transcript of its own), so counting it here would
        # flag the very fix that stopped erasing it
        bx,by,bw,bh=b['block']
        cx0,cy0=max(0,bx,x),max(0,by,y)
        cx1,cy1=min(W,bx+bw,x+w),min(H,by+bh,y+h)
        if cx1>cx0 and cy1>cy0: reg[cy0:cy1, cx0:cx1]=1
        e=ents[bi]; lh=e['line_height']
        for ln in e['lines']:
            lx0=int(ln['cx']-ln['width']/2)-4; lx1=int(ln['cx']+ln['width']/2)+4
            ly0=int(ln['y_top'])-2; ly1=int(ln['y_top']+lh)+2
            reg[max(0,ly0):min(H,ly1), max(0,lx0):min(W,lx1)]=1
        # a `lobes` override's rectangles are the whole writing area — this
        # entry's BLOCK spans every balloon AND the artwork between them
        # (p276 b04: 836x407 over a panel of sky), so the block term looks
        # where nothing was ever cleaned and letter_mask finds artwork
        _ov=_RF.override_for(stem,bi)
        if _ov and 'lobes' in _ov:
            # the balloons themselves, with a rim's worth of slack — the
            # hand-measured rectangle is deliberately generous and holds
            # artwork above each balloon's arc
            _got=_RF.lobes_from_rects(img,_ov['lobes'],b['bbox'])
            if _got is not None:
                _lim=np.zeros((H,W),np.uint8)
                _lim[y:y+h,x:x+w]=_got[0]
                _lim=cv2.dilate(_lim,np.ones((41,41),np.uint8))
                reg&=_lim
        hit=(lm>0)&(reg>0)
        if hit.sum()<25: continue
        n,lab,st,_=cv2.connectedComponentsWithStats(hit.astype(np.uint8),8)
        blobs=[(int(st[k,0]),int(st[k,1]),int(st[k,2]),int(st[k,3]),int(st[k,4]))
               for k in range(1,n) if st[k,4]>=25]
        if blobs:
            rows.append((nums.get(stem,0),stem,bi,len(blobs),b['kind'],blobs[:3]))
rows.sort(key=lambda r:-r[3])
tot=sum(r[3] for r in rows)
print(f'LEFTOVER: {tot} surviving letter blobs in {len(rows)} bubbles '
      f'on {len(set(r[1] for r in rows))} pages   [pages dir: {SRC}]')
for r in rows[:30]:
    print(f'  p{r[0]:>4} b{r[2]:02d} {r[4]:6s} blobs={r[3]:3d}  first={r[5][0] if r[5] else None}')
json.dump([[r[0],r[1],r[2],r[3],r[4]] for r in rows], open(os.environ.get('OUT',f'{SP}/leftover.json'),'w'))
