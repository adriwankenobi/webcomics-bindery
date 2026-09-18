"""qa_scan's validated gate, run against an arbitrary cleaned-pages dir."""
import os, sys, glob
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: qagate.py "<comic folder name>"')
COMIC = sys.argv[1]
C=COMIC; R=f'relettering/{C}'
txt=open("relettering/qa_scan.py").read()
ns={}; exec(compile(txt[txt.index("def is_artifact"):txt.index("bad = 0")],"qa","exec"),ns)
is_artifact=ns["is_artifact"]
for name,d in (('SHIPPED',f'upscaled/{C}'),('CANDIDATE',os.environ['CAND'])):
    bad=0; pages={}
    for f in sorted(glob.glob(f'{R}/pristine/*.jpg')):
        stem=os.path.basename(f); g=os.path.join(d,stem)
        if not os.path.exists(g): continue
        a=cv2.imread(f).astype(np.int16); b=cv2.imread(g).astype(np.int16)
        if a.shape!=b.shape: continue
        dd=(np.abs(a-b).max(axis=2)>25).astype(np.uint8)
        if not dd.any(): continue
        n,lab,st,_=cv2.connectedComponentsWithStats(dd,8)
        for k in range(1,n):
            x,y,w,h,area=st[k]
            if is_artifact(w,h,area):
                bad+=1; pages.setdefault(stem,[]).append((int(x),int(y),int(w),int(h),int(area)))
    print(f'{name}: {bad} artifact blobs on {len(pages)} pages')
    for s,v in list(pages.items())[:8]:
        print(f'    {s[-26:]}: {v[:2]}')
