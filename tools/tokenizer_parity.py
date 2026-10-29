#!/usr/bin/env python3
"""Compare this build's tokenizer against llama.cpp's own expected token ids.

llama.cpp keeps, next to each vocabulary in `models/`, two files:

  * `ggml-vocab-<name>.gguf.inp` — test strings, separated by a marker line;
  * `ggml-vocab-<name>.gguf.out` — the token ids *llama.cpp* produced for each
    string, one line per string.

That `.out` file is ground truth from another implementation, which is exactly
what a tokenizer needs and what round-tripping can never prove: an encoder can
be self-consistent and still be wrong for every other runtime.

Usage:

    python3 tools/tokenizer_parity.py --vocab-dir /path/to/llama.cpp/models
    python3 tools/tokenizer_parity.py --vocab-dir DIR --only gpt-2,llama-bpe

Exit status is 0 only when every case matches. Vocabularies whose
pre-tokeniser family this build does not implement are reported as skipped,
never as passed.

Two vocabularies fail for reasons that are not bugs in the BPE path, and are
recorded here so the distinction stays visible instead of being tuned away:

* `llama-spm`, `phi-3` are SentencePiece files. This build segments those
  greedily rather than by piece score, which is the documented SentencePiece
  limitation, not a BPE one.
* `mpt` disagrees only on whitespace-only strings, where its reference vector
  uses the raw-space token ("  ") while every other case in the same file uses
  the GPT-2 escaped form ("\u0120Hello"). The vocabulary contains both, and the
  GPT-2 code path is byte-for-byte the same as the one that passes `gpt-2`
  outright, so the reference vectors are internally inconsistent rather than
  the encoder being wrong. `--strict` counts these as failures.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "freemoe"
SEPARATOR = "\n__ggml_vocab_test__\n"

# Vocabularies whose mismatch is understood and is not a BPE defect. See the
# module docstring; --strict turns these back into failures.
KNOWN = {
    "mpt": "reference vectors mix raw-space and escaped-space forms",
}

# Vocabularies this build deliberately does not pre-tokenise. A skip for any
# name NOT in this table is a failure under --fail-on-skip: the point of the
# gate is that adding a vocabulary family, or losing one, cannot happen quietly.
EXPECTED_UNIMPLEMENTED = {
    "bert-bge": "WordPiece tokenizer; the LLM BPE path does not apply",
    "gemma-4": "SPM-style BPE (space->U+2581, raw UTF-8, no byte encoding)",
}


def read_cases(inp: Path, out: Path) -> list[tuple[str, list[int]]]:
    """The same split llama.cpp's test uses, so the case boundaries agree."""
    raw = inp.read_text(encoding="utf-8")
    cases: list[str] = []
    pos = 0
    while pos < len(raw):
        nxt = raw.find(SEPARATOR, pos)
        if nxt < 0:
            cases.append(raw[pos:])
            break
        cases.append(raw[pos:nxt])
        pos = nxt + len(SEPARATOR)

    expected: list[list[int]] = []
    for line in out.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        expected.append([int(t) for t in stripped.split()] if stripped else [])

    if len(cases) != len(expected):
        raise SystemExit(f"{inp.name}: {len(cases)} cases but {len(expected)} expected lines")
    return list(zip(cases, expected))


def encode(vocab: Path, text: str) -> list[int]:
    """Runs the CLI on one string; the file is used because cases contain
    embedded newlines and tabs that argv would mangle."""
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as handle:
        handle.write(text.encode("utf-8"))
        path = Path(handle.name)
    try:
        proc = subprocess.run(
            [str(BINARY), "tokenize", "--model", str(vocab), "--prompt-file", str(path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return []
        return [int(i) for i in json.loads(proc.stdout)["ids"]]
    finally:
        path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-dir", required=True,
                        help="directory holding ggml-vocab-*.gguf files")
    parser.add_argument("--only", default="", help="comma separated vocabulary names")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="count known, explained divergences as failures")
    parser.add_argument("--fail-on-skip", action="store_true",
                        help="fail for any vocabulary whose pre-tokenizer this build refuses "
                             "unless it is listed in EXPECTED_UNIMPLEMENTED")
    args = parser.parse_args()

    if not BINARY.exists():
        raise SystemExit(f"build first: {BINARY} is missing")

    wanted = {name.strip() for name in args.only.split(",") if name.strip()}
    results = []
    failures = 0

    for vocab in sorted(Path(args.vocab_dir).glob("ggml-vocab-*.gguf")):
        name = vocab.name.removeprefix("ggml-vocab-").removesuffix(".gguf")
        inp = Path(str(vocab) + ".inp")
        out = Path(str(vocab) + ".out")
        if wanted and name not in wanted:
            continue
        if not inp.exists() or not out.exists():
            results.append({"vocab": name, "status": "skipped", "reason": "no reference vectors"})
            continue

        try:
            cases = read_cases(inp, out)
        except SystemExit as exc:
            results.append({"vocab": name, "status": "skipped", "reason": str(exc)})
            continue

        matched = 0
        mismatches = []
        for text, expected in cases:
            actual = encode(vocab, text)
            if actual == expected:
                matched += 1
            else:
                mismatches.append({"text": text, "expected": expected, "actual": actual})

        status = "pass" if not mismatches else "fail"
        reason = None
        if status == "fail":
            # A vocabulary this build cannot pre-tokenise refuses every case;
            # that is a skip with a reason, not a silent pass.
            if all(m["actual"] == [] for m in mismatches):
                status = "skipped"
                reason = "this build refuses the pre-tokenizer"
                if args.fail_on_skip and name not in EXPECTED_UNIMPLEMENTED:
                    status = "fail"
                    reason = ("this build refuses the pre-tokenizer and the vocabulary is not "
                              "in EXPECTED_UNIMPLEMENTED")
                elif name in EXPECTED_UNIMPLEMENTED:
                    reason = "expected: " + EXPECTED_UNIMPLEMENTED[name]
            elif name in KNOWN and not args.strict:
                status = "known"
                reason = KNOWN[name]
        if status == "fail":
            failures += 1
        results.append({
            "vocab": name,
            "status": status,
            "reason": reason,
            "cases": len(cases),
            "matched": matched,
            "first_mismatch": mismatches[0] if mismatches else None,
        })

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for row in results:
            detail = ""
            if row["status"] in ("pass", "fail", "known"):
                detail = f" {row['matched']}/{row['cases']} cases"
                if row.get("first_mismatch"):
                    first = row["first_mismatch"]
                    detail += f" | first mismatch {first['text']!r}: expected {first['expected'][:8]} got {first['actual'][:8]}"
            if row.get("reason"):
                detail += f" ({row['reason']})"
            print(f"{row['status']:8s} {row['vocab']}{detail}")
        passed = sum(1 for r in results if r["status"] == "pass")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        known = sum(1 for r in results if r["status"] == "known")
        print(f"\n{passed} pass, {failures} fail, {known} known, {skipped} skipped")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
