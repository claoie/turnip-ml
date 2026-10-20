#!/usr/bin/env python3
"""Champion slot: the files that make up the champion, and their hashes.

The champion is model.npz (required) with metrics.json (the promotion
gate's baseline) and fixture_scores.json (the fixture gate's baseline,
docs/TRAINING_DESIGN.md §6.3). Replacing three R2 objects is not atomic,
so the training pipeline workflow stores each champion as a version,
training/champion/versions/<version>/, and only then writes
training/champion/current.json, the pointer: the version plus the SHA-256
of each of its files. An upload that fails partway therefore never
changes the champion, and every restore is verified against the pointer
(CONTRIBUTING rule 3). Old versions stay, so a rollback (§8) has them.

Subcommands
-----------
manifest DIR
    Write DIR/champion.json, {"files": {name: sha256}}, over the champion
    files present in DIR. Training/pipeline.sh runs it on the slot it
    stages in data/run/champion_update/.
pointer --manifest FILE --version V --out FILE
    Write the pointer: the manifest's files plus the version they are
    stored under.
version POINTER
    Print the pointer's version, validated so it is safe inside an R2 key.
verify --pointer FILE --dir DIR
    Check that every file the pointer lists is in DIR with its SHA-256.

Exit codes: 0 = ok, 2 = error (fail closed: a missing model.npz, an
unknown file name, a malformed manifest or pointer, a missing file or a
hash mismatch).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, NoReturn

from fetch_dataset import HEX64_RE
from fixture_gate import GateError, write_text_atomic

EXIT_OK = 0
EXIT_ERROR = 2

FILES = ("model.npz", "metrics.json", "fixture_scores.json")
REQUIRED = "model.npz"
MANIFEST = "champion.json"
# A run id and attempt (e.g. 1234-1), or any short name without "/".
VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")


class SlotError(Exception):
    """A champion slot that cannot be trusted."""


def _fail(msg: str) -> NoReturn:
    raise SlotError(msg)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _dump(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _check_files(files: Any, where: str) -> dict[str, str]:
    if not isinstance(files, dict) or REQUIRED not in files:
        _fail(f"{where}: needs a files object listing {REQUIRED}")
    for name, digest in files.items():
        if name not in FILES:
            _fail(f"{where}: unknown champion file {name!r}")
        if not isinstance(digest, str) or not HEX64_RE.fullmatch(digest):
            _fail(f"{where}: {name} needs a 64-hex sha256")
    return files


def _load(path: Path, keys: set[str]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        _fail(f"cannot read {path}: {e}")
    if not isinstance(data, dict) or set(data) != keys:
        _fail(f"{path}: needs exactly the keys {sorted(keys)}")
    _check_files(data["files"], str(path))
    return data


def manifest(slot: Path) -> dict[str, Any]:
    files = {
        name: sha256_file(slot / name) for name in FILES if (slot / name).is_file()
    }
    return {"files": _check_files(files, str(slot))}


def check_version(version: Any, where: str) -> str:
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        _fail(f"{where}: version must match {VERSION_RE.pattern}")
    return version


def pointer(manifest_path: Path, version: str) -> dict[str, Any]:
    files = _load(manifest_path, {"files"})["files"]
    return {"version": check_version(version, "--version"), "files": files}


def read_version(pointer_path: Path) -> str:
    data = _load(pointer_path, {"version", "files"})
    return check_version(data["version"], str(pointer_path))


def verify(pointer_path: Path, slot: Path) -> int:
    """The number of files checked."""
    data = _load(pointer_path, {"version", "files"})
    check_version(data["version"], str(pointer_path))
    files = data["files"]
    for name, digest in sorted(files.items()):
        path = slot / name
        if not path.is_file():
            _fail(f"{path} is missing from the champion")
        if sha256_file(path) != digest:
            _fail(f"{path} does not match the champion pointer's sha256")
    return len(files)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest", help=f"write DIR/{MANIFEST}")
    m.add_argument("dir", type=Path)
    pt = sub.add_parser("pointer", help="write the pointer for a version")
    pt.add_argument("--manifest", required=True, type=Path)
    pt.add_argument("--version", required=True)
    pt.add_argument("--out", required=True, type=Path)
    v = sub.add_parser("version", help="print a pointer's version")
    v.add_argument("pointer", type=Path)
    vf = sub.add_parser("verify", help="check a slot against its pointer")
    vf.add_argument("--pointer", required=True, type=Path)
    vf.add_argument("--dir", required=True, type=Path)
    args = p.parse_args(argv)
    try:
        if args.command == "manifest":
            write_text_atomic(args.dir / MANIFEST, _dump(manifest(args.dir)))
        elif args.command == "pointer":
            write_text_atomic(args.out, _dump(pointer(args.manifest, args.version)))
        elif args.command == "version":
            print(read_version(args.pointer))
        else:
            n = verify(args.pointer, args.dir)
            print(f"champion verified: {n} files match the pointer")
    except (SlotError, GateError, OSError) as e:
        print(f"champion_slot: error: {e}", file=sys.stderr)
        return EXIT_ERROR
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
