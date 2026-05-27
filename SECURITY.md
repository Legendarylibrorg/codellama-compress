# Security Policy

## Supported Versions
Security updates are applied to the `main` branch.

## Reporting a Vulnerability
Please report security issues privately via GitHub Security Advisories for this repository.

If advisories are unavailable, open a private report with:
- Impact summary
- Reproduction steps
- Affected files and versions
- Suggested remediation (if known)

Do not post exploit details in public issues.

## Threat model (CLI / pipeline)

This project loads Hugging Face models, runs optional **code execution** for HumanEval/MBPP
(`evaluate code`), and writes generated shell/Docker helper scripts (`export bundle`).

Assumptions and mitigations:

- Treat `--model-dir` and checkpoint paths as **trusted** local inputs. Paths reject `..` and
  **symlinks** during resolution.
- `trust_remote_code` defaults to **false** in config and code. Enabling it requires **both**
  config `trust_remote_code: true` and environment variable
  `CODELLAMA_COMPRESS_TRUST_REMOTE_CODE=1` (break-glass only for repos you fully trust).
- Training/calibration datasets must be on an **allowlist** (`bigcode/starcoderdata` by default).
  Extend per process with `CODELLAMA_COMPRESS_DATASET_ALLOWLIST_EXTRA` (comma-separated Hub ids).
  Pin revisions via `dataset.revision` in JSON config when possible.
- `evaluate code` is **blocked on bare-metal hosts** unless you run in Docker/CI **or** pass
  `--allow-insecure-code-exec` with `CODELLAMA_COMPRESS_ALLOW_CODE_EXEC=1`.
- Code-eval still uses a **best-effort** POSIX subprocess sandbox (blocked builtins/imports,
  resource limits, isolated mode). Prefer containers/VMs for hostile or unknown models.
- Generated export scripts are quoted/validated; `convert_gguf.sh` validates path arguments.
  Review all generated scripts before running on production hosts.
- Env reports (`pip freeze`, `nvidia-smi`) are **off by default**; pass `--env-report` when needed,
  or use `codellama-compress util env-report`.
