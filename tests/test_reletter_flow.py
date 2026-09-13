"""Unit tests for the --relettering flow helpers in process.py.
Run: .venv/bin/python -m unittest discover tests"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False))


class TranscriptGaps(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        write_json(self.work / "sheets" / "manifest.json", {
            "1-002": {"stem": "Comic 002", "count": 2},
            "1-003": {"stem": "Comic 003", "count": 1},
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_reports_untranscribed_bubbles_per_page(self):
        write_json(self.work / "parts" / "sheet-001.json",
                   {"1-002 b01": "HOLA", "1-003 b01": ""})
        gaps = process.transcript_gaps(self.work)
        self.assertEqual(gaps, {"1-002": [2]})

    def test_complete_transcription_has_no_gaps(self):
        write_json(self.work / "parts" / "sheet-001.json",
                   {"1-002 b01": "A", "1-002 b02": "B", "1-003 b01": ""})
        self.assertEqual(process.transcript_gaps(self.work), {})

    def test_later_parts_files_fill_gaps_left_by_earlier_ones(self):
        write_json(self.work / "parts" / "sheet-001.json",
                   {"1-002 b01": "A", "1-002 b02": "B"})
        write_json(self.work / "parts" / "zz-fix.json", {"1-003 b01": "C"})
        self.assertEqual(process.transcript_gaps(self.work), {})


if __name__ == "__main__":
    unittest.main()


class ParseReletterDone(unittest.TestCase):
    def test_collects_stems_from_done_lines_only(self):
        out = ("GIMP-Warning: something\n"
               "RELETTER SKIP Comic 002\n"
               "RELETTER DONE Comic 003\n"
               "RELETTER MISSING /x/Comic 004.xcf\n"
               "RELETTER DONE Comic 005-1\n"
               "RELETTER ALL DONE\n")
        self.assertEqual(process.parse_reletter_done(out),
                         ["Comic 003", "Comic 005-1"])

    def test_no_done_lines_gives_empty_list(self):
        self.assertEqual(process.parse_reletter_done("RELETTER ALL DONE\n"), [])


class GimpPendingStems(unittest.TestCase):
    """The GIMP text pass is done for a page when its QA PNG is at least as
    new as the XCF it was rendered from (compose recomposing the page makes
    the XCF newer again)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.work, self.xcf = root / "work", root / "xcf"
        (self.work / "qa").mkdir(parents=True)
        self.xcf.mkdir()
        write_json(self.work / "layout.json",
                   {"Comic 002": [], "Comic 003": [], "Comic 004": []})

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, path: Path, mtime: float) -> None:
        path.write_bytes(b"")
        import os
        os.utime(path, (mtime, mtime))

    def test_pending_when_png_missing_or_older_than_xcf(self):
        self.touch(self.xcf / "Comic 002.xcf", 100)
        self.touch(self.work / "qa" / "Comic 002.png", 200)   # done
        self.touch(self.xcf / "Comic 003.xcf", 300)
        self.touch(self.work / "qa" / "Comic 003.png", 250)   # recomposed
        self.touch(self.xcf / "Comic 004.xcf", 100)           # never passed
        self.assertEqual(process.gimp_pending_stems(self.work, self.xcf),
                         ["Comic 003", "Comic 004"])

    def test_pages_without_an_xcf_are_not_pending(self):
        # nothing to letter yet — compose hasn't produced the page
        self.assertEqual(process.gimp_pending_stems(self.work, self.xcf), [])


class WaitForTranscripts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.work = Path(self.tmp.name)
        write_json(self.work / "sheets" / "manifest.json",
                   {"1-002": {"stem": "Comic 002", "count": 1}})
        (self.work / "parts").mkdir()
        self.prompts = 0

    def tearDown(self):
        self.tmp.cleanup()

    def complete(self):
        write_json(self.work / "parts" / "sheet-001.json", {"1-002 b01": "X"})

    def wait(self, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return process.wait_for_transcripts(self.work, **kw)

    def test_already_complete_returns_true_without_prompting(self):
        self.complete()

        def read_line(_prompt=""):
            self.prompts += 1
        self.assertTrue(self.wait(
            interactive=True, read_line=read_line))
        self.assertEqual(self.prompts, 0)

    def test_non_interactive_returns_false_without_blocking(self):
        def read_line(_prompt=""):
            raise AssertionError("must not block when not interactive")
        self.assertFalse(self.wait(
            interactive=False, read_line=read_line))

    def test_interactive_waits_until_the_parts_are_filled(self):
        def read_line(_prompt=""):
            self.prompts += 1
            if self.prompts == 2:      # user fills the files on 2nd Enter
                self.complete()
        self.assertTrue(self.wait(
            interactive=True, read_line=read_line))
        self.assertEqual(self.prompts, 2)

    def test_ctrl_c_while_waiting_returns_false(self):
        def read_line(_prompt=""):
            raise KeyboardInterrupt
        self.assertFalse(self.wait(
            interactive=True, read_line=read_line))


class ReletterPreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.fonts = root / "fonts"
        self.fonts.mkdir()
        self.py = root / "python"
        self.gimp = root / "gimp"

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_present_gives_no_problems(self):
        for f in ("regular.ttf", "bolditalic.ttf"):
            (self.fonts / f).write_bytes(b"")
        self.py.write_bytes(b"")
        self.gimp.write_bytes(b"")
        self.assertEqual(process.reletter_preflight(
            fonts_dir=self.fonts, reletter_py=self.py, gimp=self.gimp), [])

    def test_names_every_missing_piece(self):
        (self.fonts / "regular.ttf").write_bytes(b"")
        problems = process.reletter_preflight(
            fonts_dir=self.fonts, reletter_py=self.py, gimp=self.gimp)
        joined = "\n".join(problems)
        self.assertEqual(len(problems), 3)
        self.assertIn("bolditalic.ttf", joined)
        self.assertIn(str(self.py), joined)
        self.assertIn(str(self.gimp), joined)


class EnsurePristine(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.up, self.work = root / "upscaled", root / "work"
        self.up.mkdir()
        self.work.mkdir()
        import os
        (self.up / "Comic 002.jpg").write_bytes(b"page")
        os.utime(self.up / "Comic 002.jpg", (1000, 1000))

    def tearDown(self):
        self.tmp.cleanup()

    def test_copies_pages_preserving_mtimes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            process.ensure_pristine(self.up, self.work)
        copy = self.work / "pristine" / "Comic 002.jpg"
        self.assertEqual(copy.read_bytes(), b"page")
        self.assertEqual(copy.stat().st_mtime, 1000)

    def test_existing_pristine_copies_are_never_overwritten(self):
        (self.work / "pristine").mkdir()
        (self.work / "pristine" / "Comic 002.jpg").write_bytes(b"original")
        with contextlib.redirect_stdout(io.StringIO()):
            process.ensure_pristine(self.up, self.work)
        self.assertEqual((self.work / "pristine" / "Comic 002.jpg").read_bytes(),
                         b"original")


class RelettertingFlagCli(unittest.TestCase):
    def run_cli(self, *args):
        import subprocess
        return subprocess.run([sys.executable, str(REPO / "process.py"), *args],
                              capture_output=True, text=True)

    def test_flag_is_rejected_with_a_single_step(self):
        r = self.run_cli("no-such-comic", "compose", "--relettering")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--relettering", r.stdout + r.stderr)
        self.assertIn("full pipeline", r.stdout + r.stderr)

    def test_flag_is_accepted_with_the_full_pipeline(self):
        # gets past argument validation and fails on the (fake) comic folder
        r = self.run_cli("no-such-comic", "all", "--relettering")
        self.assertIn("Comic folder not found", r.stdout + r.stderr)


class ParseReletterOutput(unittest.TestCase):
    OUT = ("GIMP-Warning: noise\n"
           "RELETTER SKIP Comic 002\n"
           "RELETTER SKIP Comic 003\n"
           "RELETTER DONE Comic 004\n"
           "RELETTER ALL DONE\n")

    def test_reports_done_skipped_and_completion(self):
        r = process.parse_reletter_output(self.OUT)
        self.assertEqual(r["done"], ["Comic 004"])
        self.assertEqual(r["skipped"], 2)
        self.assertTrue(r["finished"])

    def test_script_dying_midway_is_not_finished_and_keeps_the_noise(self):
        out = ("RELETTER DONE Comic 004\n"
               "Traceback (most recent call last):\n"
               "  File \"reletter_gimp.py\", line 99\n"
               "RuntimeError: font not found\n")
        r = process.parse_reletter_output(out)
        self.assertFalse(r["finished"])
        self.assertEqual(r["done"], ["Comic 004"])
        self.assertIn("RuntimeError: font not found", r["other"])


class ReletterStatus(unittest.TestCase):
    """The upfront table: one row per relettering marker, so a resumed run
    shows where it stands before doing anything."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.work, self.xcf = root / "work", root / "xcf"
        self.work.mkdir()
        self.xcf.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def rows(self):
        return dict(process.reletter_status(self.work, self.xcf))

    def test_fresh_comic_has_everything_pending(self):
        rows = self.rows()
        self.assertEqual(rows["pristine copies"], "pending")
        self.assertEqual(rows["bubble detection"], "pending")
        self.assertEqual(rows["contact sheets"], "pending")
        self.assertEqual(rows["transcription"], "pending")
        self.assertEqual(rows["fit + clean"], "pending")
        self.assertEqual(rows["GIMP text layers"], "pending")

    def test_mid_transcription_reports_the_gap(self):
        (self.work / "pristine").mkdir()
        (self.work / "pristine" / "Comic 002.jpg").write_bytes(b"x")
        write_json(self.work / "bubbles" / "Comic 002.json", [])
        write_json(self.work / "sheets" / "manifest.json",
                   {"1-002": {"stem": "Comic 002", "count": 3}})
        write_json(self.work / "parts" / "sheet-001.json", {"1-002 b01": "A"})
        rows = self.rows()
        self.assertEqual(rows["pristine copies"], "done")
        self.assertEqual(rows["bubble detection"], "done")
        self.assertEqual(rows["contact sheets"], "done")
        self.assertEqual(rows["transcription"], "2 bubbles missing on 1 pages")
        self.assertEqual(rows["fit + clean"], "pending")

    def test_finished_book_is_all_done(self):
        write_json(self.work / "bubbles" / "Comic 002.json", [])
        write_json(self.work / "sheets" / "manifest.json",
                   {"1-002": {"stem": "Comic 002", "count": 1}})
        write_json(self.work / "parts" / "sheet-001.json", {"1-002 b01": "A"})
        write_json(self.work / "layout.json", {"Comic 002": []})
        import os
        (self.xcf / "Comic 002.xcf").write_bytes(b"")
        os.utime(self.xcf / "Comic 002.xcf", (100, 100))
        (self.work / "qa").mkdir()
        (self.work / "qa" / "Comic 002.png").write_bytes(b"")
        os.utime(self.work / "qa" / "Comic 002.png", (200, 200))
        rows = self.rows()
        self.assertEqual(rows["pristine copies"], "n/a (fit already ran)")
        self.assertEqual(rows["transcription"], "complete")
        self.assertEqual(rows["fit + clean"], "done")
        self.assertEqual(rows["GIMP text layers"], "done")
