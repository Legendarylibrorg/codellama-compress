# Contributing

## Development
1. Create a feature branch from `main`.
2. Make focused changes with clear commit messages.
3. Run local checks before opening a PR.

## Required Checks
- Secret scanning (`detect-secrets`)
- Dependency audit (`pip-audit`)
- Project tests/linting where applicable

## Dependency management

- `pyproject.toml` is the **source of truth** for dependencies.
- `requirements.txt` is **auto-generated** for tooling compatibility (e.g. Dependabot).
  Regenerate it with:

```bash
python scripts/export_requirements.py
```

## Pull Requests
- Keep PRs small and reviewable.
- Explain purpose, scope, and test results.
- Ensure no credentials or secrets are committed.
