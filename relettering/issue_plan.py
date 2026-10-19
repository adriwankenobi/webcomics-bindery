"""Keep the original lettering or re-letter it — the per-ISSUE decision.

The book is printed, so re-lettering is only worth it where it makes the
letters bigger, in a font like ours. Every issue may have had its own
letterer, so the call is all of an issue's pages or none. It is made after
detection and BEFORE transcription, so a kept issue is never transcribed.

`reletter_decide.py` measures each issue and writes relettering/<comic>/
issue_decisions.json ONCE (never rewritten: a re-run must not flip an
issue that has been transcribed or approved). The hand-written
issue_overrides.json ({"6.01": "reletter"}) always wins. No decisions file
= a comic fitted before this step existed: everything is re-lettered.

Pure (json/pathlib only): process.py imports it under the base .venv.
"""
import json
from pathlib import Path

from make_sheets import issue_of

SIZE_BAR = 1.10       # re-letter when ours is at least 10% taller
UNSURE_MARGIN = 0.03  # either side of the bar: re-letter, but list it
DECISIONS = "issue_decisions.json"
OVERRIDES = "issue_overrides.json"
CHOICES = ("keep", "reletter")


def decide_issue(font: str, ratio, n_entries: int) -> dict:
    """The rule. `font`: "similar" | "different" | "unsure"; `ratio`: the
    issue's median predicted new/old cap height over its dialogue balloons
    (None when there is none to measure); `n_entries`: every detected entry
    on the issue's pages. Either gate saying re-letter wins, and an unsure
    gate re-letters (the user decides from the list at the end)."""
    # rounded: 1.10 + 0.03 is 1.1300000000000001 in floating point
    lo = round(SIZE_BAR - UNSURE_MARGIN, 6)
    hi = round(SIZE_BAR + UNSURE_MARGIN, 6)
    r = f"x{ratio:.2f}" if ratio is not None else "no ratio"

    def out(decision, unsure, reason):
        return {"decision": decision, "unsure": unsure, "reason": reason}
    if n_entries == 0:
        return out("keep", False, "no lettering detected")
    if font == "different":
        return out("reletter", False, f"font different ({r})")
    if ratio is None:
        return out("reletter", True,
                   f"font {font}, no measurable dialogue balloon")
    if font == "unsure":
        return out("reletter", True, f"font unsure, ours {r}")
    if ratio >= hi:
        return out("reletter", False, f"font similar, ours {r} (>= x{hi:.2f})")
    if ratio >= lo:
        return out("reletter", True,
                   f"font similar, ours {r} (near the x{SIZE_BAR:.2f} bar)")
    return out("keep", False, f"font similar, ours {r} (< x{lo:.2f})")


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{path}: not valid JSON ({e})") from None


def _read(work: Path):
    path = work / DECISIONS
    if not path.is_file():
        return None, {}
    decisions = _load(path)
    opath = work / OVERRIDES
    overrides = _load(opath) if opath.is_file() else {}
    for issue, value in overrides.items():
        if issue not in decisions:
            raise ValueError(
                f"{opath}: issue {issue!r} is not in {DECISIONS} "
                f"(issues: {', '.join(sorted(decisions))})")
        if value not in CHOICES:
            raise ValueError(f"{opath}: {issue!r}: {value!r} must be one of "
                             f"{' / '.join(CHOICES)}")
    return decisions, overrides


def load_plan(work: Path):
    """{issue: "keep" | "reletter"} with the overrides applied, or None for
    a comic with no decisions file (everything re-lettered)."""
    decisions, overrides = _read(work)
    if decisions is None:
        return None
    return {issue: overrides.get(issue, d["decision"])
            for issue, d in decisions.items()}


def kept_stems(work: Path, stems) -> set:
    """The page stems whose issue keeps its original lettering."""
    plan = load_plan(work)
    if plan is None:
        return set()
    return {s for s in stems if plan.get(issue_of(s)) == "keep"}


def report_lines(work: Path) -> list:
    """What was decided, for the run's output: kept, re-lettered, then the
    unsure ones (decided re-letter) the user should look at."""
    decisions, overrides = _read(work)
    if decisions is None:
        return ["every issue re-lettered (no keep-or-re-letter decisions)"]
    plan = load_plan(work)

    def item(issue):
        d = decisions[issue]
        r = d.get("size_ratio")
        txt = (f"{issue} (x{r:.2f}, font {d.get('font')})" if r is not None
               else f"{issue} (font {d.get('font')})")
        return txt + (" (override)" if issue in overrides else "")
    keep = [i for i in sorted(decisions) if plan[i] == "keep"]
    relet = [i for i in sorted(decisions) if plan[i] == "reletter"]
    unsure = [i for i in relet
              if decisions[i].get("unsure") and i not in overrides]
    lines = [f"keep original lettering ({len(keep)}): "
             + (", ".join(item(i) for i in keep) or "-"),
             f"re-letter ({len(relet)}): "
             + (", ".join(item(i) for i in relet) or "-")]
    if unsure:
        lines.append(f"unsure (decided re-letter, {len(unsure)}): "
                     + ", ".join(f"{item(i)}: {decisions[i]['reason']}"
                                 for i in unsure))
    lines.append(f"to change one, edit {work / OVERRIDES} "
                 '({"<issue>": "keep" | "reletter"}); a keep->reletter flip '
                 "needs that issue transcribed; re-run the same command")
    return lines
