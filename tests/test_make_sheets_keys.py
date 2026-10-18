"""Sheet keys ("13-012") are what the transcripts are keyed on, so two pages
must never share one. A series that numbers its issues "6.01".."6.29" must
keep the whole issue number: taking the "6" alone collapses every issue onto
one set of page keys, and the manifest keeps only the last page of each.

    .venv/bin/python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load():
    sys.path.insert(0, str(REPO / "relettering"))
    import make_sheets
    return make_sheets


class ShortKey(unittest.TestCase):
    def test_plain_issue_numbers_keep_their_keys(self):
        ms = load()
        cases = {
            "Series 13 - Title 012": "13-012",
            "Series 1 - Title 045": "1-045",
            "Series 1 - Title 999c art-1":
                "1-1",
        }
        for stem, key in cases.items():
            self.assertEqual(ms.short_key(stem), key, stem)

    def test_dotted_issue_numbers_stay_distinct(self):
        ms = load()
        self.assertEqual(ms.short_key("Series 6.29 002"),
                         "6.29-002")
        self.assertNotEqual(ms.short_key("Series 6.01 002"),
                            ms.short_key("Series 6.04 002"))
        # split issues carry a letter suffix: "7.02a" and "7.06b"
        self.assertEqual(ms.short_key("Series 7.02a 003"),
                         "7.02a-003")
        self.assertEqual(ms.short_key("Series 7.06b 003"),
                         "7.06b-003")

    def test_colliding_keys_are_refused(self):
        ms = load()
        with self.assertRaises(SystemExit):
            ms.check_unique({"a 1 x 002": 3, "a 1 y 002": 4})
        # a page with no bubbles takes no key, so it cannot collide
        ms.check_unique({"a 1 x 002": 3, "a 1 y 002": 0})


try:
    import cv2  # noqa: F401  (reletter_fit needs it; only in .venv-reletter)
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


@unittest.skipUnless(HAVE_CV2, "needs .venv-reletter")
class OverrideKey(unittest.TestCase):
    """layout_overrides.json and the frame-found caption boxes are keyed on
    the SAME key as the transcripts; a private copy of the old regex sent
    every override for "6.29-002 b03" to "6-002 b03", i.e. to page 002 of
    every issue in the series."""

    def test_override_key_is_the_sheet_key(self):
        sys.path.insert(0, str(REPO / "relettering"))
        import reletter_fit
        self.assertEqual(
            reletter_fit.override_key("Series 6.29 002", 3),
            "6.29-002 b03")
        self.assertEqual(
            reletter_fit.override_key(
                "Series 13 - Title 012", 1),
            "13-012 b01")


if __name__ == "__main__":
    unittest.main()
