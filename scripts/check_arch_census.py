#!/usr/bin/env python3
"""Offline consistency gate for the architecture census (C11).

`tools/triage_upstream_archs.py --check` re-derives the census from a pinned
llama.cpp checkout, which needs the network. This script is the half that runs
in `make test`: it takes the committed artefact as given and checks the things
that would be *wrong statements* rather than stale measurements.

* the partition adds up — one verdict per family, counts match rows;
* every `blocked` family names a capability, every named capability is declared,
  and `blocked_by_capability` is that same claim restated rather than a second
  copy that can drift;
* **no family with an adapter row is classified blocked, and every adapter row
  that has an upstream file appears in the census as mapped.** This is the check
  that would have caught the previous census: its capability list treated
  `build_moe_ffn(` as an unimplemented block, so `llama`, `granite`, `phi3`,
  `qwen3moe` and thirteen other mapped families were counted as needing one;
* the table in `docs/GAP_ANALYSIS.md` is the artefact rendered, not hand-typed.

Exit 0 with the open-work counts; exit 1 on any inconsistency.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "docs" / "evidence" / "arch_census.json"
DOC = ROOT / "docs" / "GAP_ANALYSIS.md"
VERDICTS = {"mapped", "mapped-caveat", "blocked", "unclassified"}


def load_tool():
    path = ROOT / "tools" / "triage_upstream_archs.py"
    spec = importlib.util.spec_from_file_location("triage_upstream_archs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def main() -> int:
    problems: list[str] = []
    if not CENSUS.exists():
        print(f"{CENSUS.relative_to(ROOT)} is missing", file=sys.stderr)
        return 1
    census = json.loads(CENSUS.read_text(encoding="utf-8"))
    rows = census.get("rows", [])
    tool = load_tool()

    # 1. Shape and partition.
    archs = [row["arch"] for row in rows]
    if len(archs) != len(set(archs)):
        problems.append("duplicate architecture in the census rows")
    if census.get("files") != len(rows):
        problems.append(f"`files` is {census.get('files')} but there are {len(rows)} rows")
    counts = census.get("counts", {})
    if sum(counts.values()) != len(rows):
        problems.append(f"counts sum to {sum(counts.values())}, rows are {len(rows)}")
    for row in rows:
        if row["verdict"] not in VERDICTS:
            problems.append(f"{row['arch']}: unknown verdict {row['verdict']!r}")
    recounted: dict[str, int] = {}
    for row in rows:
        recounted[row["verdict"]] = recounted.get(row["verdict"], 0) + 1
    if recounted != counts:
        problems.append(f"`counts` says {counts}, rows say {recounted}")

    # 2. Capability attribution is one claim, stated once.
    capabilities = census.get("capabilities", {})
    claimed: dict[str, list[str]] = {}
    for row in rows:
        if row["verdict"] == "blocked":
            cid = row.get("capability", "")
            if cid not in capabilities:
                problems.append(f"{row['arch']}: blocked by unknown capability {cid!r}")
                continue
            claimed.setdefault(cid, []).append(row["arch"])
        elif row.get("capability"):
            problems.append(f"{row['arch']}: {row['verdict']} row still names a capability")
    claimed = {cid: sorted(names) for cid, names in claimed.items()}
    if claimed != census.get("blocked_by_capability", {}):
        problems.append("`blocked_by_capability` disagrees with the per-row capability ids")
    # A capability no family needs at this pin is dead vocabulary, not a false
    # claim — it is reported so the list gets pruned, but it does not fail the
    # gate: upstream can drop a family without anything here being wrong.
    unused = sorted(set(capabilities) - set(claimed))

    # 3. The adapter table is the source of truth for what is mapped.
    adapters = set(tool.adapter_arches())
    aliases = tool.ALIASES
    by_arch = {row["arch"]: row for row in rows}
    for row in rows:
        mapped_row = row["arch"] in adapters or aliases.get(row["arch"]) in adapters
        if mapped_row and row["verdict"] == "blocked":
            problems.append(f"{row['arch']}: has an adapter row but is classified blocked")
        if not mapped_row and row["verdict"].startswith("mapped"):
            problems.append(f"{row['arch']}: classified {row['verdict']} with no adapter row")
    for arch in sorted(adapters):
        upstream = aliases.get(arch, arch)
        if upstream in by_arch and not by_arch[upstream]["verdict"].startswith("mapped"):
            problems.append(f"{arch}: adapter row exists but the census calls it "
                            f"{by_arch[upstream]['verdict']}")

    # 4. A caveat has to say what the unmodelled block is.
    for row in rows:
        if row["verdict"] == "mapped-caveat" and not row.get("blocking"):
            problems.append(f"{row['arch']}: mapped-caveat with no block named")
        if row["verdict"] == "unclassified" and row.get("nearest") is None:
            problems.append(f"{row['arch']}: unclassified with no nearest mapped family; "
                            f"the read queue is what makes the remaining work countable")

    # 5. The document is the artefact rendered.
    doc = DOC.read_text(encoding="utf-8")
    if tool.splice(doc, tool.render_table(census)) != doc:
        problems.append("docs/GAP_ANALYSIS.md: the arch-census table is not what the "
                        "artefact renders")

    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        print("\nregenerate with tools/triage_upstream_archs.py --llama-dir DIR --write",
              file=sys.stderr)
        return 1

    if unused:
        print(f"note: capabilities with no family at this pin: {', '.join(unused)}")
    blocked = counts.get("blocked", 0)
    unclassified = counts.get("unclassified", 0)
    print(f"arch census: {census['files']} families, {counts.get('mapped', 0)} mapped"
          f" (+{counts.get('mapped-caveat', 0)} with a caveat), {blocked} blocked by a named"
          f" capability, {unclassified} needing a graph read")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
