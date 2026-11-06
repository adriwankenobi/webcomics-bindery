"""The template geometry cache is keyed on the template's CONTENTS, so a
clone or a copy (new mtimes) reuses it instead of re-probing with GIMP.
Run: .venv/bin/python -m unittest discover tests"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import process  # noqa: E402


class TemplateGeometryCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.template = d / "template.xcf"
        self.template.write_bytes(b"template bytes")
        self.cache = d / "template_geometry.json"
        self.patches = [mock.patch.object(process, "TEMPLATE", self.template),
                        mock.patch.object(process, "GEOMETRY_CACHE", self.cache)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def probe(self, stdout):
        return mock.patch.object(process.subprocess, "run", return_value=mock.Mock(
            stdout=stdout, stderr=""))

    def test_a_new_mtime_on_the_same_template_reuses_the_cache(self):
        with self.probe("CANVAS 1875 2775\nRES 300\nBOX 1 2 3 4\n") as run:
            first = process.template_geometry()
        self.assertEqual(run.call_count, 1)
        os.utime(self.template, (1_000_000_000.5, 1_000_000_000.5))
        with self.probe("") as run:
            again = process.template_geometry()
        run.assert_not_called()
        self.assertEqual(again, first)
        self.assertEqual(json.loads(self.cache.read_text()), first)

    def test_a_changed_template_is_measured_again(self):
        with self.probe("CANVAS 1875 2775\nRES 300\n"):
            process.template_geometry()
        self.template.write_bytes(b"edited template")
        with self.probe("CANVAS 1800 2700\nRES 300\n") as run:
            geo = process.template_geometry()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(geo["canvas"], [1800, 2700])


if __name__ == "__main__":
    unittest.main()
