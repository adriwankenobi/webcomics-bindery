"""Artifact gate split by caption-box membership. A box we wipe WHOLE is a
large contiguous change by design, which is_artifact cannot tell from damage,
so count those separately — blobs OUTSIDE a box are what must not grow."""
import os, sys, glob, json, unicodedata
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: gate_split.py "<comic folder name>"')
COMIC = sys.argv[1]
sys.path.insert(0,'relettering'); sys.argv=['reletter_fit.py',COMIC]
import reletter_fit as F
SP=os.environ['SP']; C=COMIC; R=f'relettering/{C}'
txt=open("relettering/qa_scan.py").read()
ns={}; exec(compile(txt[txt.index("def is_artifact"):txt.index("bad = 0")],"qa","exec"),ns)
is_artifact=ns["is_artifact"]
N=lambda s: unicodedata.normalize('NFC',s)
def nk(d): return {N(k):v for k,v in d.items()}
LAY=nk(json.load(open(os.environ.get('LAYOUT', f'{R}/layout.json')))); TR=nk(json.load(open(f'{R}/transcripts.json')))
nums={N(k.rsplit('.',1)[0]):v for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
for name,d in (('SHIPPED',f'upscaled/{C}'),('CANDIDATE',os.environ['CAND'])):
    inbox=outbox=0; outpages={}
    for f in sorted(glob.glob(f'{R}/pristine/*.jpg')):
        raw=os.path.splitext(os.path.basename(f))[0]; stem=N(raw)
        g=os.path.join(d,os.path.basename(f))
        if not os.path.exists(g): continue
        a=cv2.imread(f).astype(np.int16); b=cv2.imread(g).astype(np.int16)
        if a.shape!=b.shape: continue
        bp=f'{R}/bubbles/{raw}.json'
        if not os.path.exists(bp): continue
        bubs=json.load(open(bp)); tr=TR.get(stem,[])
        fitted={e['index'] for e in LAY.get(stem,[]) if (tr[e['index']-1] if e['index']-1<len(tr) else '')}
        boxes=[bx for bx,_ in F.caption_boxes(bubs,fitted)]
        for bi in fitted:
            ov=F.override_for(stem,bi)
            if ov and 'box' in ov: boxes.append(tuple(ov['box']))
        dd=(np.abs(a-b).max(axis=2)>25).astype(np.uint8)
        if not dd.any(): continue
        n,lab,st,_=cv2.connectedComponentsWithStats(dd,8)
        for k in range(1,n):
            x,y,w,h,area=st[k]
            if not is_artifact(w,h,area): continue
            cx,cy=x+w/2.0,y+h/2.0
            if any(bx0-8<=cx<=bx1+8 and by0-8<=cy<=by1+8 for bx0,by0,bx1,by1 in boxes):
                inbox+=1
            else:
                outbox+=1; outpages.setdefault(nums.get(stem,0),[]).append((int(x),int(y),int(w),int(h),int(area)))
    print(f'{name}: {outbox} blobs OUTSIDE a caption box on {len(outpages)} pages'
          f'   ({inbox} inside boxes = deliberate whole-box wipes)')
    for pg,v in sorted(outpages.items()): print(f'    p{pg}: {v[:3]}')
