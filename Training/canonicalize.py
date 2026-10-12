"""Raw trick labels -> canonical names at one pinned taxonomy version.

The farm keeps raw free-text labels forever; training consumes canonical
names through `label_taxonomy` mappings (raw, canonical, taxonomy_version),
pinned to a single version for the whole run.

Per raw label:
  1. Normalize: lowercase, trim, collapse internal whitespace.
  2. Strip parentheticals such as "(combo)".
  3. Look the whole string up (as typed, then without parentheticals). On
     a miss, split it on - / + , and look up each token, so
     "hook - scoot - gainer - cartfull (combo)" becomes four names on one
     window. The whole-string lookup comes first so a mapped name that
     itself contains a delimiter (e.g. "b-twist") is not split.
  4. A token with no mapping keeps its normalized form as its canonical
     name and is reported as unmapped, for taxonomy curation.

Version resolution: for each raw string, the mapping with the greatest
taxonomy_version <= the pinned version applies (mappings are immutable per
version, and a later version may remap a raw string).

Canonical names are normalized strings without "+": training joins combo
names with "+", so a "+" inside a name would be ambiguous.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

SPLIT_RE = re.compile(r"[-/+,]")
PARENTHETICAL_RE = re.compile(r"\([^()]*\)")


class TaxonomyError(ValueError):
    """Mappings that cannot be applied unambiguously."""


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(label: str) -> tuple[str, list[str]]:
    """(whole normalized label, its delimiter-split tokens)."""
    whole = normalize(PARENTHETICAL_RE.sub(" ", normalize(label)))
    tokens = [normalize(t) for t in SPLIT_RE.split(whole)]
    return whole, [t for t in tokens if t]


class Taxonomy:
    def __init__(self, mappings: Iterable[tuple[str, str, int]], version: int):
        self.version = version
        best: dict[str, tuple[int, str]] = {}
        for raw, canonical, raw_version in mappings:
            if raw_version > version:
                continue
            if canonical != normalize(canonical) or not canonical or "+" in canonical:
                raise TaxonomyError(
                    f"canonical name {canonical!r} must be a non-empty normalized "
                    'string without "+"'
                )
            key = normalize(raw)
            current = best.get(key)
            if current is None or raw_version > current[0]:
                best[key] = (raw_version, canonical)
            elif raw_version == current[0] and canonical != current[1]:
                raise TaxonomyError(
                    f"raw {key!r} maps to both {current[1]!r} and {canonical!r} "
                    f"at taxonomy_version {raw_version}"
                )
        self._lookup = {key: canonical for key, (_, canonical) in best.items()}

    def canonicalize(self, label: str) -> tuple[list[str], list[str]]:
        """(sorted canonical names, sorted unmapped tokens) for one raw label.

        A label that is nothing but parentheticals or delimiters yields no
        names at all.
        """
        whole, tokens = _tokens(label)
        # The label as typed first (a mapped raw may itself carry a
        # parenthetical), then without parentheticals, then token by token.
        for candidate in (normalize(label), whole):
            if candidate in self._lookup:
                return [self._lookup[candidate]], []
        names: set[str] = set()
        unmapped: set[str] = set()
        for token in tokens:
            canonical = self._lookup.get(token)
            if canonical is None:
                unmapped.add(token)
                names.add(token)
            else:
                names.add(canonical)
        return sorted(names), sorted(unmapped)
