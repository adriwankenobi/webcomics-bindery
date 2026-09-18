"""A coloured caption box detected as TWO stacked entries (a `tint` strip and a
`bubble` strip) split where the gradient fill crosses LIGHT_MIN.
Each strip is cleaned and typeset independently -> different sizes, a band of
old text left uncleaned at the junction, new text printed over it."""
import os, json, os, unicodedata
import sys
if len(sys.argv) < 2:
    sys.exit('usage: split_caps.py "<comic folder name>"')
COMIC = sys.argv[1]
C=COMIC; R=f'relettering/{C}'
SP=os.environ.get('SP', '/tmp/relettering-qa')
def nk(d): return {unicodedata.normalize('NFC',k):v for k,v in d.items()}
nums={unicodedata.normalize('NFC',k.rsplit('.',1)[0]):v for k,v in json.load(open(f'xcf/{C}/numbers.json')).items()}
LAY=nk(json.load(open(f'{R}/layout.json'))); TR=nk(json.load(open(f'{R}/transcripts.json')))
pairs=[]
for stem in sorted(LAY):
    bubs=json.load(open(f'{R}/bubbles/{stem}.json'))
    fonts={e['index']:e['font_px'] for e in LAY[stem]}
    laid=set(fonts)
    for i,a in enumerate(bubs,1):
        if a['kind']!='tint' or i not in laid: continue
        ax,ay,aw,ah=a['bbox']
        for j,b in enumerate(bubs,1):
            if j==i or b['kind']!='bubble' or j not in laid: continue
            bx,by,bw,bh=b['bbox']
            # b starts where a ends (within a line height) and they share x
            if not (-40 <= by-(ay+ah) <= 30): continue
            ovl=min(ax+aw,bx+bw)-max(ax,bx)
            if ovl < 0.80*max(aw,bw): continue
            pairs.append((nums.get(stem,0),stem,i,j,fonts[i],fonts[j],aw,ah,bw,bh))
pairs.sort()
pages=sorted({p[0] for p in pairs})
print(f'SPLIT CAPTION BOXES: {len(pairs)} boxes on {len(pages)} pages')
print(f'{"page":>5} {"tint":>5} {"bub":>5} {"px_t":>5} {"px_b":>5}  size mismatch')
for p in pairs:
    print(f'{p[0]:>5} b{p[2]:02d}   b{p[3]:02d}  {p[4]:>5} {p[5]:>5}  {"YES" if p[4]!=p[5] else "-"}')
print('\npages:',pages)
json.dump([[p[0],p[1],p[2],p[3]] for p in pairs],open(f'{SP}/split_caps.json','w'))
