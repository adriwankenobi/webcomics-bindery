"""How far does typeset text stick out of its balloon?
For every laid-out line, test its ink box against the bubble mask.
LAYOUT env selects the layout.json to measure."""
import json, os, sys, unicodedata
import numpy as np, cv2
if len(sys.argv) < 2:
    sys.exit('usage: protrude.py "<comic folder name>"')
COMIC = sys.argv[1]
C=COMIC; R=f'relettering/{C}'
LAY_PATH=os.environ.get('LAYOUT', f'{R}/layout.json')
def nk(d): return {unicodedata.normalize('NFC',k):v for k,v in d.items()}
nums={unicodedata.normalize('NFC',os.path.splitext(k)[0]):v
      for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
LAY=nk(json.load(open(LAY_PATH)))
rows=[]
for stem,ents in LAY.items():
    bj=f'{R}/bubbles/{stem}.json'
    if not os.path.exists(bj): continue
    bubs=json.load(open(bj))
    for e in ents:
        b=bubs[e['index']-1]
        if b['kind']!='bubble': continue
        x,y,w,h=b['bbox']
        mp=f"{R}/bubbles/{stem}-b{e['index']:02d}-mask.png"
        if not os.path.exists(mp): continue
        m=(cv2.imread(mp,cv2.IMREAD_GRAYSCALE)>127)
        lh=e['line_height']; worst=0
        for ln in e['lines']:
            x0=int(round(ln['cx']-ln['width']/2.0)); x1=int(round(ln['cx']+ln['width']/2.0))
            # the ink band of a line sits inside the line box
            y0=int(round(ln['y_top']+0.18*lh)); y1=int(round(ln['y_top']+0.92*lh))
            out=0
            for yy in range(y0,y1):
                ry=yy-y
                if ry<0 or ry>=m.shape[0]: out+=max(0,x1-x0); continue
                rowm=m[ry]
                for xx in (range(max(x0,x),min(x1,x+w))):
                    if not rowm[xx-x]: out+=1
                out+=max(0,x-x0)+max(0,x1-(x+w))
            worst=max(worst,out)
        if worst: rows.append((nums.get(stem,0),stem,e['index'],worst,e['font_px']))
rows.sort(key=lambda r:-r[3])
print(f'{len(rows)} bubbles have at least one line protruding outside the mask')
for t in (200,800,2000):
    print(f'  worst-line protrusion > {t}px: {sum(1 for r in rows if r[3]>t)}')
print('\nWORST 20:')
for r in rows[:20]:
    print(f'  p{r[0]:>4} b{r[2]:02d} font={r[4]:2d} protruding_px={r[3]}')
