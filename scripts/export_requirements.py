from __future__ import annotations

import re
from pathlib import Path


def _extract_array(pyproject_text: str, key: str) -> list[str]:
    # Minimal TOML-ish extraction for our simple arrays:
    # dependencies = [ "a", "b>=1" ]
    m = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\[(.*?)\]\s*$", pyproject_text, re.DOTALL)
    if not m:
        return []
    body = m.group(1)
    items = re.findall(r'"([^"]+)"', body)
    return [i.strip() for i in items if i.strip()]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    req = root / "requirements.txt"

    text = pyproject.read_text()
    deps = _extract_array(text, "dependencies")

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
