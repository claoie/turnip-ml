# Contributing to turnip-ml

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
```

CI is Python 3.12 on Ubuntu.

## Where things live

| Path | What |
|---|---|
| `Training/` | Trick-detection training code (more arrives with the feat/12–15 rebases) |
| `.github/workflows/` | CI |

This file is the pointer doc; the READMEs under each directory are the source of truth.

## Rules that are not negotiable

1. **Trick labels come from the maintainer — never inferred from footage.** Do not
   guess what trick a clip shows; labels are the maintainer's words.

2. **Determinism.** Training and eval code must be deterministic: fixed seeds,
   sorted JSON keys, fixed float rounding, no timestamps, no dict-order
   dependence. If you touch scoring, prove it by running the scorer twice on
   the same input and diffing the outputs.

3. **Large artifacts are never committed.** No model binaries, datasets,
   or videos in git. Any helper that downloads external bytes must verify
   hashes and fail loudly on mismatch — never silently pass.

4. **No secrets, ever.** CI runs on fork PRs, where secrets are invisible.
   Do not add anything that needs a secret to CI.

## Pull requests

Work happens on the `Museoie/turnip-ml` fork (no push access to
`hoiekim/*`); open cross-fork PRs against `hoiekim/turnip-ml` `main`.

Use this as the PR body:

```markdown
## Summary
<what changed and why>

## Related issues
Closes hoiekim/turnip-ml#<N>

## Test plan
<what you ran>

## Checklist
- [ ] No model binaries, datasets, or videos committed
- [ ] Deterministic: re-ran on identical input, outputs byte-identical (or n/a)
```

## CI on your PR

- `python-checks` runs ruff + mypy on PRs touching Python files.
