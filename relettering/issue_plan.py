"""Keep the original lettering or re-letter it — the per-ISSUE decision.

The book is printed, so re-lettering is only worth it where the original
prints too small, in a font like ours. Every issue may have had its own
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

# Re-letter when the original's dialogue capitals PRINT shorter than this.
# An absolute size, not our gain over the original: the fit is held down by
# the balloons, so above it our type comes out only a fraction of a mm
# taller (invisible on paper) at the price of a font swap; below it,
# lifting the type to the book's size is a gain a reader sees.
MIN_CAP_MM = 1.25
DECISIONS = "issue_decisions.json"
OVERRIDES = "issue_overrides.json"
CHOICES = ("keep", "reletter")


def decide_issue(font: str, cap_mm, n_entries: int) -> dict:
    """The rule. `font`: "similar" | "different" | "unsure"; `cap_mm`: the
    issue's median printed cap height of the ORIGINAL's dialogue balloons,
    in mm (None when there is none to measure); `n_entries`: every detected
    entry on the issue's pages. Either gate saying re-letter wins, and an
    unsure gate re-letters (the user decides from the list at the end)."""
    c = f"original {cap_mm:.2f}mm" if cap_mm is not None else "no cap height"

    def out(decision, unsure, reason):
        return {"decision": decision, "unsure": unsure, "reason": reason}
    if n_entries == 0:
        return out("keep", False, "no lettering detected")
    if font == "different":
        return out("reletter", False, f"font different ({c})")
    if cap_mm is None:
        return out("reletter", True,
                   f"font {font}, no measurable dialogue balloon")
    if font == "unsure":
        return out("reletter", True, f"font unsure, {c}")
    if cap_mm < MIN_CAP_MM:
        return out("reletter", False,
                   f"font similar, {c} (< {MIN_CAP_MM:.2f}mm)")
    return out("keep", False, f"font similar, {c} (>= {MIN_CAP_MM:.2f}mm)")


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
        cap, r = d.get("orig_cap_mm"), d.get("size_ratio")
        facts = ([f"{cap:.2f}mm"] if cap is not None else []) \
            + ([f"ours x{r:.2f}"] if r is not None else []) \
            + [f"font {d.get('font')}"]
        return (f"{issue} ({', '.join(facts)})"
                + (" (override)" if issue in overrides else ""))
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
