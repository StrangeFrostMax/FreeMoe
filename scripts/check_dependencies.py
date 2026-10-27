#!/usr/bin/env python3
"""Verify the documented build-dependency inventory against build files."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    arch = (ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8") if (ROOT / "CMakeLists.txt").exists() else ""
    build_text = makefile + "\n" + cmake
    section = arch.split("## Build dependency inventory", 1)[-1].split("\n## ", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or line.startswith("|---") or "Name" in line:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        refs = re.findall(r"`([^`]+)`", cells[3])
        implicit = "implicit" in cells[3].lower()
        found = implicit or any(ref in build_text for ref in refs)
        rows.append({"name": cells[0], "version": cells[1], "licence": cells[2],
                     "build_reference": cells[3], "found_in_build": found})
    licence_ok = (ROOT / "LICENSES" / "VNPU-MIT.txt").exists()
    status = "pass" if rows and all(r["found_in_build"] for r in rows) and licence_ok else "fail"
    out = {"status": status, "dependencies": rows, "vnpu_license_present": licence_ok}
    print(json.dumps(out, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
