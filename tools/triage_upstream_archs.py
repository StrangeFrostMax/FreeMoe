#!/usr/bin/env python3
"""Census of every upstream llama.cpp architecture, with its blocking fact.

Reads `src/models/*.cpp` from a **pinned** llama.cpp checkout and classifies
each family against what this build actually implements — not against a list
that was true once. The output is committed as `docs/evidence/arch_census.json`
so a reader can see the partition without network access, and so a drift between
the document and the tree is detectable.

    tools/triage_upstream_archs.py --llama-dir DIR --write   # regenerate
    tools/triage_upstream_archs.py --llama-dir DIR --check   # re-derive, diff
    tools/triage_upstream_archs.py --summary                 # read the artefact

Verdicts, and why there are four rather than two:

  mapped              an adapter row exists for it (`freemoe adapters`)
  mapped-caveat       an adapter row exists *and* the graph contains a marker for
                      a block this engine does not model (a vision tower, an MTP
                      head). The row is still real; the caveat is listed so the
                      document says which sub-graph is ignored
  blocked             no row, and the graph calls something with no
                      implementation in `src/engine.cpp`; `capabilities[].id`
                      names what it would take
  unclassified        no row and no marker hit. This is **not** a claim that the
                      family is one adapter row away: upstream can differ
                      structurally without naming a missing primitive (an encoder
                      stack, a pooling head, post-norm placement). It needs the
                      graph read that `docs/PROCESS.md` requires before a row is
                      claimed. The count of these is the honest size of the
                      remaining census work

`--check` is the gate: it re-derives the census from the pinned tree and fails if
the committed artefact disagrees. `scripts/check_arch_census.py` is the offline
half, which verifies the artefact against `freemoe adapters` and the document.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "evidence" / "arch_census.json"

# Pinned upstream tree. Bump deliberately, in a commit that regenerates the
# artefact and the table in `docs/GAP_ANALYSIS.md`; an unpinned census is a
# snapshot of whatever happened to be on the machine that ran it.
PIN = "0ec4027a2a0183de87c007069adddc06e665f7de"
REPO = "ggml-org/llama.cpp"

# Blocks this build implements. These are *not* blocking facts: the census used
# to list `build_moe_ffn(` as one, which classified `llama`, `granite`, `phi3`
# and fifteen other mapped families as "needs a block". A capability list that
# is not derived from the engine drifts from it in both directions.
IMPLEMENTED = {
    "moe": r"build_moe_ffn\(",
    "ssm-mamba": r"ggml_ssm_|build_mamba",
    "attn-temperature": r"attn_temp",
    "weightless-norm": r"build_norm\(\s*[^,]+,\s*NULL\s*,\s*NULL",
    "shared-expert": r"ffn_gate_shexp|ffn_up_shexp",
    "router-bias": r"ffn_exp_probs_b|exp_probs_b",
    "qk-norm": r"attn_q_norm|attn_k_norm",
    "sliding-window": r"n_swa|sliding_window",
    "sinks": r"attn_sinks",
    "mrope": r"rope_sections|mrope",
    "fused-qkv": r"TENSOR_ATTN_QKV",
    "fused-gate-up": r"gate_up",
    "mla": r"attn_kv_a|wkv_a|kv_a_norm",
    "tied-output": r"TENSOR_DUPLICATED",
    "embedding-scale": r"f_embedding_scale|embedding_scale",
    "partial-rotary": r"n_rot(\s*<|\s*!=)",
}

# Blocks this build does not implement, with the capability id that would close
# them. One id per family, choosing the most fundamental missing piece, because
# "needs three things" is a batch, not a row.
CAPABILITIES = [
    ("C-SHORTCONV", "short convolution / conv1d in the block", r"shortconv|short_conv|conv1d"),
    ("C-LINEAR-ATTN", "linear/recurrent attention layer", r"build_recurrent|linear_attn|gated_linear"),
    ("C-DELTA-NET", "gated delta-net scan", r"delta_net|ggml_gated_delta|build_delta"),
    ("C-DSA-INDEXER", "sparse-attention (DSA) indexer", r"indexer"),
    ("C-RWKV", "RWKV scan", r"rwkv|RWKV"),
    ("C-ENC-DEC", "encoder/decoder cross-attention stack", r"ggml_cross|encoder"),
    ("C-HYPERCONN", "hyper-connections (per-layer residual streams)", r"hc_|hyper_connection|dsv4_hc"),
    ("C-ATTN-COMP", "attention compressor", r"attn_comp"),
    ("C-MULTIMODAL", "vision tower / multimodal projector", r"ggml_conv|vision|mm_projector|clip"),
    ("C-TID2EID", "token-id expert routing", r"experts_tid2eid|tid2eid"),
    ("C-ALTUP", "AltUp / per-layer token embeddings", r"build_inp_embd_per_layer|altup|AltUp"),
    ("C-DUAL-FFN", "dense and MoE FFN summed in one layer", r"ffn_norm_exps"),
]

# Blocks that are not blocking, with the reason. The NextN/MTP head is the case
# this list exists for: upstream writes its tensors as `blk.{n_layer}.nextn.*`
# and builds the head only when a loader asks for it (`ml.load_mtp`, false in an
# inference run), so a checkpoint carrying one runs its trunk unchanged. The
# engine ignores those tensors by rule and says so in `inspect`; the tolerance is
# pinned by the `qwen3moe-mtp` fixture and the `mtp_head_tolerated` test.
#
# Listing it as a capability was wrong in a way that mattered: it made 16
# families look unbuildable when the only thing standing between them and a row
# was the rest of their block, and it hid the fact that nothing in the tree
# proved the tolerance at all.
TOLERATED = [
    ("NextN / MTP head", r"mtp\.|nextn\.",
     "tensors the trunk never reads; the loader ignores them by rule"),
]

# Upstream spells some families differently from their adapter row.
ALIASES = {"openai-moe": "gpt-oss", "granite-moe": "granitemoe"}

# Marker hits that are known false positives, each with the reason. Every entry
# here is a claim about the engine, so it carries its own sentence and the table
# prints it: an unexplained override would be the same defect as a stale
# capability list. A new entry needs a paragraph in docs/GAP_ANALYSIS.md C11.
EXPLAINED = {
    "mamba2": {
        "C-SHORTCONV": "the `conv1d` in this file is `ssm_conv1d`, the SSM short "
                       "convolution the mamba2 block already implements — not a "
                       "generic shortconv layer",
    },
}


def graph_body(path: Path) -> str:
    text = path.read_text(errors="ignore")
    marker = "::graph::graph("
    idx = text.find(marker)
    if idx < 0:
        idx = text.find("::graph(")
    return text[idx:] if idx >= 0 else text


def adapter_arches() -> list[str]:
    """Adapter rows from the built binary; falls back to parsing the table."""
    exe = ROOT / "build" / "freemoe"
    if exe.exists():
        proc = subprocess.run([str(exe), "adapters", "--json"],
                              capture_output=True, text=True, check=False)
        if proc.returncode == 0 and proc.stdout.strip().startswith("["):
            return [row["arch"] for row in json.loads(proc.stdout)]
    source = (ROOT / "src" / "adapter.cpp").read_text(encoding="utf-8")
    return re.findall(r'\{"([a-z0-9.-]+)", "LLM_ARCH_', source)


def classify(arch: str, path: Path, mapped: set[str]) -> dict:
    graph = graph_body(path)
    implemented = sorted(name for name, pattern in IMPLEMENTED.items() if re.search(pattern, graph))
    hits = [(cid, what) for cid, what, pattern in CAPABILITIES if re.search(pattern, graph)]
    explained = {cid: why for cid, why in EXPLAINED.get(arch, {}).items()}
    hits = [hit for hit in hits if hit[0] not in explained]
    tolerated = [name for name, pattern, _ in TOLERATED if re.search(pattern, graph)]
    row = arch in mapped or ALIASES.get(arch) in mapped
    if row:
        # A caveat is not a blocker: the row exists and runs. The unmodelled
        # sub-graph is listed in `blocking` so the document can name it, but the
        # row claims no capability, because a capability is what an *unmapped*
        # family waits on.
        verdict = "mapped-caveat" if hits else "mapped"
        capability = ""
    elif hits:
        verdict = "blocked"
        capability = hits[0][0]
    else:
        verdict = "unclassified"
        capability = ""
    return {
        "arch": arch,
        "file": path.name,
        "verdict": verdict,
        "capability": capability,
        "blocking": [cid for cid, _ in hits],
        "tolerated": tolerated,
        "explained": explained,
        "implemented": implemented,
    }


def build(llama_dir: Path, pin: str) -> dict:
    models = llama_dir / "src" / "models"
    if not models.is_dir():
        raise SystemExit(f"{models} is not a directory")
    mapped = set(adapter_arches())
    rows = [classify(path.stem, path, mapped) for path in sorted(models.glob("*.cpp"))
            if path.name != "models.cpp"]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
    by_capability: dict[str, list[str]] = {}
    for row in rows:
        if row["verdict"] == "blocked":
            by_capability.setdefault(row["capability"], []).append(row["arch"])
    return {
        "schema": "freemoe.arch-census.v1",
        "upstream": {"repo": REPO, "pin": pin},
        "method": "graph-marker scan; a verdict is not evidence a family runs",
        "adapter_rows": len(mapped),
        "files": len(rows),
        "counts": counts,
        "capabilities": {cid: what for cid, what, _ in CAPABILITIES},
        "blocked_by_capability": {k: sorted(v) for k, v in sorted(by_capability.items())},
        "rows": rows,
    }


def render_table(census: dict) -> str:
    counts = census["counts"]
    total = census["files"]
    lines = [
        "<!-- arch-census:begin (generated by tools/triage_upstream_archs.py; do not edit by hand) -->",
        "",
        f"Upstream `{census['upstream']['repo']}` at "
        f"`{census['upstream']['pin'][:12]}…`, `src/models/*.cpp`: **{total} families**.",
        f"This build has **{census['adapter_rows']} adapter rows**. The partition:",
        "",
        "| Verdict | Families | Meaning |",
        "|---|---:|---|",
        f"| mapped | {counts.get('mapped', 0)} | adapter row, and the graph names nothing this build lacks |",
        f"| mapped, with a caveat | {counts.get('mapped-caveat', 0)} | adapter row, but the file also builds a block the row does not model |",
        f"| blocked | {counts.get('blocked', 0)} | no row; the graph calls a primitive this build has no implementation of |",
        f"| unclassified | {counts.get('unclassified', 0)} | no row and no marker hit — needs the graph read before a row can be claimed |",
        "",
        "`unclassified` is not a polite word for \"almost supported\". Upstream can differ",
        "structurally without naming a missing primitive (an encoder stack, a pooling",
        "head, post-norm placement), so a family only leaves that bucket once its block",
        "has been read and it has either a row with a fixture and an oracle golden, or a",
        "named capability.",
        "",
    ]
    if census["blocked_by_capability"]:
        lines += ["| Capability | Families | Of which |", "|---|---:|---|"]
        for cid, families in census["blocked_by_capability"].items():
            what = census["capabilities"].get(cid, cid)
            lines.append(f"| `{cid}` — {what} | {len(families)} | {', '.join(f'`{f}`' for f in families)} |")
        lines.append("")
    tolerated = [r["arch"] for r in census["rows"] if r.get("tolerated")]
    if tolerated:
        lines.append(f"**Not blocking:** {len(tolerated)} families carry a NextN / MTP head. "
                     "Its tensors are named `blk.{n_layer}.nextn.*`, upstream builds the head "
                     "only when a loader asks for it (`ml.load_mtp`), and this engine maps those "
                     "tensors to `IGNORED` with the reason recorded in `inspect`. Pinned by the "
                     "`qwen3moe-mtp` fixture and the `mtp_head_tolerated` test — bit-identical "
                     "logits against the same file without the head.")
        lines.append("")
        lines.append(f"Families carrying one: {', '.join(f'`{a}`' for a in sorted(tolerated))}.")
        lines.append("")

    caveats = [r for r in census["rows"] if r["verdict"] == "mapped-caveat"]
    if caveats:
        lines.append("Mapped rows whose file also builds a block the row does not model:")
        lines.append("")
        for row in caveats:
            lines.append(f"- `{row['arch']}` — {', '.join(f'`{c}`' for c in row['blocking'])}")
        lines.append("")
    explained = [(r["arch"], cid, why) for r in census["rows"]
                 for cid, why in r.get("explained", {}).items()]
    if explained:
        lines.append("Known false positives of the marker scan, each with its reason:")
        lines.append("")
        for arch, cid, why in explained:
            lines.append(f"- `{arch}` / `{cid}` — {why}.")
        lines.append("")
    lines.append("<!-- arch-census:end -->")
    return "\n".join(lines)


def splice(text: str, block: str) -> str:
    begin, end = "<!-- arch-census:begin", "<!-- arch-census:end -->"
    start, stop = text.find(begin), text.find(end)
    if start < 0 or stop < 0:
        raise SystemExit("docs/GAP_ANALYSIS.md: arch-census markers not found")
    return text[:start] + block + text[stop + len(end):]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-dir", help="path to a llama.cpp checkout at the pinned commit")
    parser.add_argument("--pin", default=None, help="upstream commit the checkout is at")
    parser.add_argument("--write", action="store_true", help="write the artefact and the doc table")
    parser.add_argument("--check", action="store_true", help="verify the committed artefact")
    parser.add_argument("--summary", action="store_true", help="print the committed partition")
    parser.add_argument("--json", action="store_true", help="print rows as JSON")
    args = parser.parse_args()

    if args.summary or not args.llama_dir:
        if not OUT.exists():
            print(f"{OUT} does not exist; run with --llama-dir and --write", file=sys.stderr)
            return 2
        census = json.loads(OUT.read_text(encoding="utf-8"))
        if args.json:
            print(json.dumps(census["rows"], indent=2))
            return 0
        print(f"{census['files']} families at {census['upstream']['pin'][:12]}: "
              + ", ".join(f"{k}={v}" for k, v in sorted(census['counts'].items())))
        for cid, families in census["blocked_by_capability"].items():
            print(f"  {cid:<16} {len(families):>3}  {' '.join(families)}")
        return 0

    pin = args.pin
    if pin is None:
        head = subprocess.run(["git", "-C", args.llama_dir, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=False)
        pin = head.stdout.strip() if head.returncode == 0 else PIN
    if pin != PIN:
        print(f"warning: checkout is at {pin[:12]}, the committed census is pinned at {PIN[:12]}; "
              f"--check will fail", file=sys.stderr)

    census = build(Path(args.llama_dir), pin)
    text = json.dumps(census, indent=1, sort_keys=True) + "\n"
    table = render_table(census)

    if args.json:
        print(json.dumps(census["rows"], indent=2))
        return 0

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
        doc = ROOT / "docs" / "GAP_ANALYSIS.md"
        doc.write_text(splice(doc.read_text(encoding="utf-8"), table), encoding="utf-8")
        print(f"{OUT.relative_to(ROOT)}: {census['files']} families "
              f"({', '.join(f'{k}={v}' for k, v in sorted(census['counts'].items()))}); "
              f"table regenerated in docs/GAP_ANALYSIS.md")
        return 0

    if args.check:
        committed = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        problems = []
        if committed != text:
            diff = "\n".join(difflib.unified_diff(
                committed.splitlines(), text.splitlines(),
                fromfile=str(OUT.relative_to(ROOT)), tofile="re-derived from the checkout",
                lineterm="", n=1))
            problems.append(diff)
        doc = (ROOT / "docs" / "GAP_ANALYSIS.md").read_text(encoding="utf-8")
        if splice(doc, table) != doc:
            problems.append("docs/GAP_ANALYSIS.md: the arch-census table is stale")
        if problems:
            print("\n".join(problems))
            print("\nrun tools/triage_upstream_archs.py --llama-dir DIR --write", file=sys.stderr)
            return 1
        print(f"arch census matches upstream {PIN[:12]} and docs/GAP_ANALYSIS.md")
        return 0

    print(json.dumps({k: v for k, v in census.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
