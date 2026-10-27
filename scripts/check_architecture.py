#!/usr/bin/env python3
"""Verify docs/ARCHITECTURE.md module inventory against the source tree."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    doc_path = ROOT / "docs" / "ARCHITECTURE.md"
    if not doc_path.exists():
        print(json.dumps({"status": "fail", "reason": "missing docs/ARCHITECTURE.md"}))
        return 1
    text = doc_path.read_text(encoding="utf-8")
    exists_paths = re.findall(r"\|\s*EXISTS\s*\|\s*`([^`]+)`", text)
    missing_paths = [p for p in exists_paths if not (ROOT / p).exists()]
    src_files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "src").rglob("*") if p.is_file())
    not_listed = [p for p in src_files if p not in text]
    status = "pass" if not missing_paths and not not_listed else "fail"
    out = {
        "status": status,
        "exists_rows": len(exists_paths),
        "src_files": src_files,
        "missing_paths": missing_paths,
        "src_files_not_listed": not_listed,
    }
    print(json.dumps(out, indent=2))
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
