"""Run the REAL reletter_fit.main() against staged pristine pages with the
book size pinned, so nothing but the genuinely-changed pages moves."""
import os, sys, os
from pathlib import Path
SP=os.environ.get('SP', '/tmp/relettering-qa')
if len(sys.argv) < 2:
    sys.exit('usage: rebuild_fit.py "<comic folder name>"')
COMIC = sys.argv[1]
BOOK = os.environ.get('BOOK')
if not BOOK:
    sys.exit('set BOOK=<book size> — an UNPINNED re-fit re-derives it and moves approved pages')
sys.argv=["reletter_fit.py",COMIC,"--book",BOOK]
sys.path.insert(0,"relettering")
import reletter_fit as F
F.PAGES_DIR = Path(SP)/"stage"          # clean the staged pristine copies
print("PAGES_DIR ->", F.PAGES_DIR, flush=True)
F.main()
