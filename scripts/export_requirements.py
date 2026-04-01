from __future__ import annotations

from pathlib import Path


def _extract_toml_string_array(text: str, *, key: str) -> list[str]:
    """
    Extract a TOML array of *double-quoted strings* for a simple key, e.g.:

      dependencies = [
        "torch>=2.1",
        "transformers>=4.40",
      ]

    This intentionally avoids adding a TOML dependency.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith(f"{key}"):
            if "=" in line and "[" in line.split("=", 1)[1]:
                start = i
                break
    if start is None:
        return []

    # Collect from the first '[' to the matching ']'.
    buf: list[str] = []
    depth = 0
    started = False
    for line in lines[start:]:
        # Strip comments.
        line = line.split("#", 1)[0]
        if "[" in line:
            depth += line.count("[")
            started = True
        if started:
            buf.append(line)
        if "]" in line and started:
            depth -= line.count("]")
            if depth <= 0:
                break

    body = "\n".join(buf)
    # Extract double-quoted strings; we don't attempt full TOML escapes.
    out: list[str] = []
    i = 0
    while True:
        i = body.find('"', i)
        if i == -1:
            break
        j = body.find('"', i + 1)
        if j == -1:
            break
        s = body[i + 1 : j].strip()
        if s:
            out.append(s)
        i = j + 1
    return out


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    req = root / "requirements.txt"

    text = pyproject.read_text()
    deps = _extract_toml_string_array(text, key="dependencies")

    out = []
    out.append("# AUTO-GENERATED from pyproject.toml")
    out.append("# Do not edit by hand. Regenerate with:")
    out.append("#   python scripts/export_requirements.py")
    out.append("")
    out.extend(deps)
    out.append("")

    req.write_text("\n".join(out))


if __name__ == "__main__":
    main()
