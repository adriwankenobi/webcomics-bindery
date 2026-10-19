"""The keep-or-re-letter rule and the per-comic plan files
(relettering/issue_plan.py). Pure: runs under the base .venv.

    .venv/bin/python -m unittest discover tests
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "relettering"))

import issue_plan  # noqa: E402


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


class DecideIssue(unittest.TestCase):
    CASES = [
        # font, ratio, entries -> decision, unsure
        ("similar", 1.00, 50, "keep", False),
        ("similar", 1.06, 50, "keep", False),
        ("similar", 1.07, 50, "reletter", True),
        ("similar", 1.12, 50, "reletter", True),
        ("similar", 1.13, 50, "reletter", False),
        ("different", 0.99, 50, "reletter", False),
        ("unsure", 1.00, 50, "reletter", True),
        ("similar", None, 5, "reletter", True),
        ("similar", None, 0, "keep", False),
    ]

    def test_rule(self):
        for font, ratio, n, decision, unsure in self.CASES:
            got = issue_plan.decide_issue(font, ratio, n)
            self.assertEqual((got["decision"], got["unsure"]),
                             (decision, unsure), (font, ratio, n))
            self.assertTrue(got["reason"])


class LoadPlan(unittest.TestCase):
    KEPT = "Series 6.01 002"
    RELET = "Series 6.28 004"

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def decisions(self):
        write_json(self.work / issue_plan.DECISIONS, {
            "6.01": {"decision": "keep", "unsure": False, "reason": "r",
                     "font": "similar", "size_ratio": 1.0, "balloons": 9},
            "6.28": {"decision": "reletter", "unsure": False, "reason": "r",
                     "font": "similar", "size_ratio": 1.2, "balloons": 9},
        })

    def test_no_decisions_file_is_legacy(self):
        self.assertIsNone(issue_plan.load_plan(self.work))
        self.assertEqual(
            issue_plan.kept_stems(self.work, [self.KEPT, self.RELET]), set())

    def test_decisions(self):
        self.decisions()
        self.assertEqual(issue_plan.load_plan(self.work),
                         {"6.01": "keep", "6.28": "reletter"})

    def test_override_wins(self):
        self.decisions()
        write_json(self.work / issue_plan.OVERRIDES, {"6.01": "reletter"})
        self.assertEqual(issue_plan.load_plan(self.work)["6.01"], "reletter")

    def test_override_unknown_issue_raises(self):
        self.decisions()
        write_json(self.work / issue_plan.OVERRIDES, {"6.99": "keep"})
        with self.assertRaises(ValueError) as cm:
            issue_plan.load_plan(self.work)
        self.assertIn("6.99", str(cm.exception))
        self.assertIn("issue_overrides.json", str(cm.exception))

    def test_override_bad_value_raises(self):
        self.decisions()
        write_json(self.work / issue_plan.OVERRIDES, {"6.01": "Keep"})
        with self.assertRaises(ValueError) as cm:
            issue_plan.load_plan(self.work)
        self.assertIn("Keep", str(cm.exception))

    def test_unreadable_overrides_name_the_file(self):
        self.decisions()
        (self.work / issue_plan.OVERRIDES).write_text('{"6.01": keep}')
        with self.assertRaises(ValueError) as cm:
            issue_plan.load_plan(self.work)
        self.assertIn("issue_overrides.json", str(cm.exception))

    def test_kept_stems(self):
        self.decisions()
        self.assertEqual(
            issue_plan.kept_stems(self.work, [self.KEPT, self.RELET]),
            {self.KEPT})


class ReportLines(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        write_json(self.work / issue_plan.DECISIONS, {
            "6.01": {"decision": "keep", "unsure": False, "reason": "r",
                     "font": "similar", "size_ratio": 1.0, "balloons": 9},
            "6.04": {"decision": "reletter", "unsure": True, "reason": "r",
                     "font": "similar", "size_ratio": 1.08, "balloons": 9},
            "6.28": {"decision": "reletter", "unsure": False, "reason": "r",
                     "font": "similar", "size_ratio": 1.2, "balloons": 9},
        })
        write_json(self.work / issue_plan.OVERRIDES, {"6.01": "reletter"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_report(self):
        lines = issue_plan.report_lines(self.work)
        self.assertTrue(any(ln.lstrip().startswith("unsure") and "6.04" in ln
                            for ln in lines), lines)
        self.assertTrue(any("issue_overrides.json" in ln for ln in lines))
        self.assertTrue(any("6.01" in ln and "(override)" in ln
                            for ln in lines), lines)


if __name__ == "__main__":
    unittest.main()
