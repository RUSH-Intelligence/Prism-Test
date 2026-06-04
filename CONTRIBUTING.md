# Contributing to Prism-Test

This guide is for developers working on the repo. For how to *use* the eval harness, see [EVAL_HARNESS.md](EVAL_HARNESS.md).

## One-time setup

Install `pre-commit` and `detect-secrets` globally via `pipx` (recommended) or `pip`:

```bash
pipx install pre-commit
pipx install detect-secrets
```

Then in your clone of the repo:

```bash
pre-commit install
```

This wires the hooks into `.git/hooks/pre-commit` so they run automatically before every commit.

To sanity-check the setup, run the hooks against the whole repo once:

```bash
pre-commit run --all-files
```

## Branch & PR workflow

Direct pushes to `main` are not allowed. All changes go through a pull request.

```bash
git checkout -b your-branch-name
# ...make changes, commit...
git push -u origin your-branch-name
gh pr create --fill
```

Get at least one review before merging.

## What pre-commit checks

| Check | What it catches |
|---|---|
| `trailing-whitespace` | Whitespace at end of lines (auto-fixed) |
| `end-of-file-fixer` | Missing final newline (auto-fixed) |
| `check-yaml` / `check-json` / `check-toml` | Broken config syntax |
| `check-merge-conflict` | Unresolved `<<<<<<<` markers |
| `check-case-conflict` | `Foo.py` + `foo.py` (breaks on Windows/Mac) |
| `detect-private-key` | SSH/SSL private keys |
| `check-added-large-files` | Files larger than 5MB |
| `detect-secrets` | API keys, tokens, credentials |

## When a hook blocks your commit

### "Files were modified by this hook"

`trailing-whitespace` or `end-of-file-fixer` auto-fixed your files. The fix is in your working tree but **not staged**. Re-stage and commit again:

```bash
git add -A
git commit -m "your message"
```

### "Detect secrets" fired

First decide: is it a **real secret** or a **false positive**?

**Real secret (e.g., an API key you pasted into a file):**

1. Treat the key as compromised — rotate it immediately.
2. Remove it from the file. Use an environment variable or a gitignored `.env` instead.
3. Commit again.

**False positive (e.g., a long hash, a model ID, an example value):**

Two options, in order of preference:

- Mark the specific line as known-safe:
  ```python
  api_key = "EXAMPLE_VALUE_NOT_REAL"  # pragma: allowlist secret
  ```
- Or update the baseline so future scans skip it:
  ```bash
  detect-secrets scan --baseline .secrets.baseline
  git add .secrets.baseline
  ```
  (On Windows PowerShell, if the baseline gets corrupted by encoding issues, regenerate with `cmd /c "detect-secrets scan > .secrets.baseline"`.)

### "Check added large files" fired (>5MB)

Don't commit it. Large files belong somewhere else:

- **Model weights** → HuggingFace Hub
- **Datasets** → HuggingFace Hub, S3, or cluster storage
- **Run outputs / logs** → already gitignored (`results/`, `*.log`); check why this one wasn't
- **Genuinely needed in repo** → Git LFS, or bump the limit in `.pre-commit-config.yaml` after discussion

### "Detect private key" fired

A PEM-formatted key (`-----BEGIN ... PRIVATE KEY-----`) was found in a staged file. Almost always a real key. Remove it, rotate it, use a secrets manager.

### Last resort: bypass

```bash
git commit --no-verify
```

Skips all hooks for one commit. Use only when you're certain everything is safe — it disables secret scanning too. Avoid making it a habit.

## Tests

Run the unit test suite from the repo root:

```bash
python -m unittest discover eval_harness/tests -v
```

Tests should not load real models. See `CLAUDE.md` for the testing convention.
