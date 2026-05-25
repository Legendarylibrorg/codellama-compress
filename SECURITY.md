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

- Treat `--model-dir` and checkpoint paths as **trusted** local inputs; paths are resolved and
  traversal patterns are rejected.
- `trust_remote_code` defaults to **false**; enabling it via config can execute arbitrary code
  from a model repository.
- Code-eval uses a **best-effort** POSIX subprocess sandbox (resource limits, import guard,
  isolated mode). It is not a substitute for OS-level isolation (containers/VMs) on hostile code.
- Generated export scripts are quoted/validated; review them before running on production hosts.
- Run artifacts may include `pip freeze` and `nvidia-smi`; disable with `--no-env-report` when sharing outputs.
