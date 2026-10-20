"""Unit tests for Training/champion_slot.py.

Hermetic: small files in a temporary directory, no R2.
Run with:  python -m unittest discover -s Training/tests -t .
"""

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import champion_slot
from champion_slot import EXIT_ERROR, EXIT_OK, MANIFEST


class ChampionSlotTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.stdout = self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.stderr = self.enterContext(contextlib.redirect_stderr(io.StringIO()))
        self.slot = self.root / "slot"
        self.slot.mkdir()
        for name in champion_slot.FILES:
            (self.slot / name).write_bytes(f"{name} bytes".encode())
        self.pointer = self.root / "current.json"

    def cli(self, *args):
        return champion_slot.main([str(a) for a in args])

    def store(self, version="1234-1"):
        """What pipeline.sh and the workflow's store step do."""
        self.assertEqual(self.cli("manifest", self.slot), EXIT_OK)
        args = ["pointer", "--manifest", self.slot / MANIFEST, "--version", version]
        return self.cli(*args, "--out", self.pointer)

    def verify(self, slot=None):
        return self.cli("verify", "--pointer", self.pointer, "--dir", slot or self.slot)

    def test_manifest_lists_each_file_hash(self):
        self.store()
        files = json.loads((self.slot / MANIFEST).read_text())["files"]
        self.assertEqual(sorted(files), sorted(champion_slot.FILES))
        for name, digest in files.items():
            data = (self.slot / name).read_bytes()
            self.assertEqual(digest, hashlib.sha256(data).hexdigest())

    def test_round_trip_verifies(self):
        self.assertEqual(self.store(), EXIT_OK)
        self.assertEqual(self.cli("version", self.pointer), EXIT_OK)
        self.assertEqual(self.stdout.getvalue().strip(), "1234-1")
        self.assertEqual(self.verify(), EXIT_OK)
        self.assertIn("3 files match", self.stdout.getvalue())

    def test_model_alone_is_a_slot(self):
        for name in ("metrics.json", "fixture_scores.json"):
            (self.slot / name).unlink()
        self.assertEqual(self.store(), EXIT_OK)
        self.assertEqual(self.verify(), EXIT_OK)

    def test_tampered_or_missing_file_fails_closed(self):
        self.store()
        restored = self.root / "restored"
        for case in ("tampered", "missing"):
            with self.subTest(case=case):
                if restored.exists():
                    for p in restored.iterdir():
                        p.unlink()
                else:
                    restored.mkdir()
                for p in self.slot.iterdir():
                    (restored / p.name).write_bytes(p.read_bytes())
                if case == "tampered":
                    (restored / "fixture_scores.json").write_text("{}")
                else:
                    (restored / "metrics.json").unlink()
                self.assertEqual(self.verify(restored), EXIT_ERROR)

    def test_slot_without_model_is_refused(self):
        (self.slot / "model.npz").unlink()
        self.assertEqual(self.cli("manifest", self.slot), EXIT_ERROR)
        self.assertFalse((self.slot / MANIFEST).exists())

    def test_bad_versions_are_refused(self):
        for version in ("", "../champion", "a/b", "-1", "x" * 200, "v 1"):
            with self.subTest(version=version):
                self.assertEqual(self.store(version), EXIT_ERROR)

    def test_malformed_pointers_are_refused(self):
        self.store()
        good = json.loads(self.pointer.read_text())
        digest = "0" * 64
        cases = {
            "not json": "{",
            "extra key": {**good, "note": 1},
            "no files": {"version": "1"},
            "no model": {"version": "1", "files": {"metrics.json": digest}},
            "unknown file": {
                "version": "1",
                "files": {**good["files"], "../evil": digest},
            },
            "bad hash": {"version": "1", "files": {"model.npz": "abc"}},
            "bad version": {**good, "version": "../x"},
        }
        for case, data in cases.items():
            with self.subTest(case=case):
                text = data if isinstance(data, str) else json.dumps(data)
                self.pointer.write_text(text)
                self.assertEqual(self.cli("version", self.pointer), EXIT_ERROR)
                self.assertEqual(self.verify(), EXIT_ERROR)

    def test_outputs_are_deterministic(self):
        self.store()
        first = (self.slot / MANIFEST).read_bytes(), self.pointer.read_bytes()
        self.store()
        self.assertEqual(
            ((self.slot / MANIFEST).read_bytes(), self.pointer.read_bytes()), first
        )


if __name__ == "__main__":
    unittest.main()
