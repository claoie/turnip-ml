"""Unit tests for Training/canonicalize.py.

Run with:  python -m unittest discover -s Training/tests -t .
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from canonicalize import Taxonomy, TaxonomyError, normalize

# The farm's seed taxonomy (version 1).
SEED = [
    ("cork", "cork", 1),
    ("corkscrew", "cork", 1),
    ("gainer", "gainer", 1),
    ("cartfull", "cartfull", 1),
    ("hook", "hook", 1),
    ("scoot", "scoot", 1),
]


class CanonicalizeTest(unittest.TestCase):
    def setUp(self):
        self.taxonomy = Taxonomy(SEED, 1)

    def test_normalize(self):
        self.assertEqual(normalize("  Cork \t 720\n"), "cork 720")

    def test_master_plan_combo_example(self):
        names, unmapped = self.taxonomy.canonicalize(
            "hook - scoot - gainer - cartfull (combo)"
        )
        self.assertEqual(names, ["cartfull", "gainer", "hook", "scoot"])
        self.assertEqual(unmapped, [])

    def test_alias_case_and_whitespace(self):
        self.assertEqual(self.taxonomy.canonicalize("  CorkScrew "), (["cork"], []))

    def test_all_delimiters_split(self):
        names, _ = self.taxonomy.canonicalize("cork/gainer+hook,scoot-cartfull")
        self.assertEqual(names, ["cartfull", "cork", "gainer", "hook", "scoot"])

    def test_duplicate_names_collapse(self):
        self.assertEqual(self.taxonomy.canonicalize("cork / corkscrew"), (["cork"], []))

    def test_unmapped_tokens_keep_their_normalized_form(self):
        names, unmapped = self.taxonomy.canonicalize("Hook - Mystery  Flip")
        self.assertEqual(names, ["hook", "mystery flip"])
        self.assertEqual(unmapped, ["mystery flip"])

    def test_whole_label_is_looked_up_before_splitting(self):
        taxonomy = Taxonomy([*SEED, ("b-twist", "b-twist", 1)], 1)
        self.assertEqual(taxonomy.canonicalize("B-Twist"), (["b-twist"], []))
        # Without the mapping the same label splits into two unmapped tokens.
        self.assertEqual(
            self.taxonomy.canonicalize("B-Twist"), (["b", "twist"], ["b", "twist"])
        )

    def test_mapped_raw_with_a_parenthetical(self):
        taxonomy = Taxonomy([*SEED, ("cork (720)", "cork", 1)], 1)
        self.assertEqual(taxonomy.canonicalize("Cork (720)"), (["cork"], []))
        self.assertEqual(self.taxonomy.canonicalize("Cork (720)"), (["cork"], []))

    def test_label_without_tokens_yields_nothing(self):
        for label in ("(combo)", " - ", "/,+"):
            with self.subTest(label=label):
                self.assertEqual(self.taxonomy.canonicalize(label), ([], []))


class VersionTest(unittest.TestCase):
    MAPPINGS = (
        ("cork720", "cork", 1),
        ("cork720", "cork 720", 2),
        ("dub", "double cork", 5),
    )

    def test_greatest_version_at_or_below_the_pin_applies(self):
        self.assertEqual(
            Taxonomy(self.MAPPINGS, 1).canonicalize("cork720")[0], ["cork"]
        )
        self.assertEqual(
            Taxonomy(self.MAPPINGS, 2).canonicalize("cork720")[0], ["cork 720"]
        )
        self.assertEqual(
            Taxonomy(self.MAPPINGS, 3).canonicalize("cork720")[0], ["cork 720"]
        )

    def test_mappings_above_the_pin_are_ignored(self):
        self.assertEqual(
            Taxonomy(self.MAPPINGS, 4).canonicalize("dub"), (["dub"], ["dub"])
        )
        self.assertEqual(
            Taxonomy(self.MAPPINGS, 5).canonicalize("dub"), (["double cork"], [])
        )

    def test_conflicting_mappings_at_one_version(self):
        with self.assertRaises(TaxonomyError):
            Taxonomy([("Cork", "cork", 1), ("cork", "gainer", 1)], 1)
        # The same mapping listed twice is not a conflict.
        Taxonomy([("cork", "cork", 1), ("Cork", "cork", 1)], 1)

    def test_invalid_canonical_names(self):
        for canonical in ("Cork", " cork", "a+b", ""):
            with self.subTest(canonical=canonical), self.assertRaises(TaxonomyError):
                Taxonomy([("x", canonical, 1)], 1)
        # A bad mapping above the pin is never applied, so never checked.
        Taxonomy([("x", "a+b", 9)], 1)


if __name__ == "__main__":
    unittest.main()
