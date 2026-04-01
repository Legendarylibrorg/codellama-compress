from __future__ import annotations

import os

from codellama_compress.code_exec import run_python_sandboxed


def test_run_python_sandboxed_ok() -> None:
    r = run_python_sandboxed(code="print('hi')\n")
    if os.name != "posix":
        assert r.reason == "unsupported_platform"
    else:
        assert r.ok is True
        assert "hi" in r.stdout


def test_run_python_sandboxed_timeout() -> None:
    r = run_python_sandboxed(code="while True:\n    pass\n", timeout_s=0.2)
    if os.name != "posix":
        assert r.reason == "unsupported_platform"
    else:
        assert r.ok is False
        assert r.reason == "timeout"
