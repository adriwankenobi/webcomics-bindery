"""Check the keep-or-re-letter measurements against a comic that was
already fitted: the original's printed cap height per issue measured from
the blobs against the one the fit measured (layout.json old_cap_h), the
verdict each gives; the PREDICTED size ratio (placeholder text through
the real fit) against the ACTUAL one; the font gate's verdict and
statistics, and how well the placeholder's word counts match the
transcripts.

READ-ONLY on the comic: results go to $SP/decide/<comic>.json only.

    SP=/tmp/relettering-qa .venv-reletter/bin/python \
        relettering/qa-tools/decide_calibrate.py "<comic folder name>"

The actual new cap height is recomputed from each entry's font sizes with
TODAY's metric (Fitter.cap_height): layouts fitted before the cap-height
fix recorded a new_cap_h ~7% high.
"""
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

if len(sys.argv) < 2:
    sys.exit('usage: decide_calibrate.py "<comic folder name>"')
COMIC = sys.argv[1]
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "relettering"))
import reletter_fit as F  # noqa: E402
F.configure(COMIC)
import numpy as np  # noqa: E402
import reletter_decide as D  # noqa: E402
from issue_plan import decide_issue  # noqa: E402
from make_sheets import issue_of  # noqa: E402

OUT = Path(os.environ.get("SP", "/tmp/relettering-qa")) / "decide"
OUT.mkdir(parents=True, exist_ok=True)


def actual_ratios(fit):
    """-> ({issue: [new/old cap ratios]}, {issue: [old cap heights px]})."""
    layout = json.loads((F.WORK / "layout.json").read_text())
    out, olds = defaultdict(list), defaultdict(list)
    for stem, entries in layout.items():
        for e in entries:
            if e.get("kind") != "bubble" or not e.get("old_cap_h"):
                continue
            sizes = [ln.get("font_px", e["font_px"]) for ln in e["lines"]]
            out[issue_of(stem)].append(
                fit.cap_height(min(sizes or [e["font_px"]])) / e["old_cap_h"])
            olds[issue_of(stem)].append(e["old_cap_h"])
    return out, olds


def word_agreement():
    """Per placeholder vs transcript: |words - words| / words, bubbles."""
    tr = json.loads((F.WORK / "transcripts.json").read_text())
    errs = []
    for stem, texts in tr.items():
        path = next(iter((F.WORK / "pristine").glob(f"{stem}.*")), None)
        page = D.load_page(F, path) if path else None
        if page is None:
            continue
        img, bubbles = page
        for b, ph, t in zip(bubbles, D.page_texts(img, bubbles), texts):
            real = F.apply_fixes(t).replace("*", "").split()
            if b["kind"] == "bubble" and real and ph:
                errs.append(abs(len(ph.split()) - len(real)) / len(real))
    return float(np.median(errs)), float(np.mean(errs)), len(errs)


def main():
    t0 = time.time()
    fit = F.Fitter()
    detail = {}
    pred = D.assess(F.WORK / "pristine", detail)
    act, olds = actual_ratios(fit)
    res_px = D.print_res()
    # the book the real fit used (most dialogue sits at it), to tell an
    # error in the ESTIMATED book from an error per balloon
    from collections import Counter
    layout = json.loads((F.WORK / "layout.json").read_text())
    real_book = Counter(e["font_px"] for v in layout.values() for e in v
                        if e.get("kind") == "bubble").most_common(1)[0][0]
    with_real = D.apply_book(detail["rows"], fit, real_book)
    rows = []
    for issue, rec in pred.items():
        a = act.get(issue)
        a_med = round(float(np.median(a)), 3) if a else None
        n = rec["balloons"] or len(a or [])
        o = olds.get(issue)
        cap_act = (round(float(np.median(o)) * D.MM_PER_IN / res_px, 3)
                   if o else None)
        size_pred = decide_issue("similar", rec["orig_cap_mm"], n)
        size_act = decide_issue("similar", cap_act, n)
        rows.append({
            "issue": issue, "cap_mm": rec["orig_cap_mm"],
            "cap_mm_actual": cap_act,
            "pred": rec["size_ratio"], "actual": a_med,
            "pred_real_book": (round(float(np.median(with_real[issue])), 3)
                               if with_real.get(issue) else None),
            "err": (round(rec["size_ratio"] - a_med, 3)
                    if rec["size_ratio"] is not None and a_med is not None
                    else None),
            "size_pred": size_pred["decision"],
            "size_pred_unsure": size_pred["unsure"],
            "size_actual": size_act["decision"],
            "font": rec["font"], "font_balloons": rec["font_balloons"],
            "font_delta": rec["font_delta"],
            "decision": rec["decision"], "unsure": rec["unsure"]})
    wmed, wmean, wn = word_agreement()
    ref = D.font_reference(fit, 20.0)
    res = {"comic": COMIC, "minutes": round((time.time() - t0) / 60, 1),
           "word_err_median": round(wmed, 3), "word_err_mean": round(wmean, 3),
           "word_n": wn, "our_font_stats_cap20": ref,
           "book_estimated": detail["book"], "book_real": real_book,
           "issues": rows, "balloons": detail["rows"]}
    (OUT / f"{COMIC}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1))
    print(f"\n== {COMIC}: {res['minutes']} min; placeholder word-count "
          f"error median {wmed:.2f} mean {wmean:.2f} over {wn} balloons")
    print(f"our font @cap20: {ref}")
    print(f"book size: estimated {detail['book']}px, the real fit's "
          f"{real_book}px")
    print(f"{'issue':<10}{'mm':>6}{'mm act':>7}{'pred':>7}{'@book':>7}"
          f"{'actual':>8}{'err':>7}  size(pred/act)"
          f"   font     n   median delta to our font")
    for r in rows:
        flag = "" if (r["size_pred"] == r["size_actual"]
                      or r["size_pred_unsure"]) else "  << MISMATCH"
        print(f"{r['issue']:<10}{r['cap_mm'] or 0:>6.2f}"
              f"{r['cap_mm_actual'] or 0:>7.2f}{r['pred'] or 0:>7.3f}"
              f"{r['pred_real_book'] or 0:>7.3f}{r['actual'] or 0:>8.3f}"
              f"{r['err'] if r['err'] is not None else 0:>7.3f}  "
              f"{r['size_pred']:>8}/{r['size_actual']:<8} {r['font']:<9}"
              f"{r['font_balloons']:>4}   {r['font_delta']}{flag}")


if __name__ == "__main__":
    main()
