#!/usr/bin/env python3
"""Check the reference decoders against llama.cpp's own dequantizers.

The unit tests pin `src/quant.cpp` to `tools/quant_reference.py`. That proves the
two agree, not that either is right. This script closes the loop: it compiles a
tiny harness against an upstream llama.cpp checkout, feeds it the same synthetic
blocks the reference generator produces, and compares element by element.

    python3 tools/quant_upstream_check.py --upstream /path/to/llama.cpp

Nothing here is needed to build or test FreeMoe; it is the evidence that the
transcriptions are faithful, and it is the check to re-run after adding a type.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BRIDGE = r"""
#include "ggml-quants.h"
#include "ggml.h"

extern "C" void fm_dequantize_row(int type, const void *src, float *dst, long long count) {
    switch (type) {
    case 2:  dequantize_row_q4_0((const block_q4_0 *)src, dst, count); break;
    case 3:  dequantize_row_q4_1((const block_q4_1 *)src, dst, count); break;
    case 6:  dequantize_row_q5_0((const block_q5_0 *)src, dst, count); break;
    case 7:  dequantize_row_q5_1((const block_q5_1 *)src, dst, count); break;
    case 8:  dequantize_row_q8_0((const block_q8_0 *)src, dst, count); break;
    // Q8_1 has no reference dequantizer upstream (it only exists as an
    // activation format), so it is covered by the reference decoder alone.
    case 10: dequantize_row_q2_K((const block_q2_K *)src, dst, count); break;
    case 11: dequantize_row_q3_K((const block_q3_K *)src, dst, count); break;
    case 12: dequantize_row_q4_K((const block_q4_K *)src, dst, count); break;
    case 13: dequantize_row_q5_K((const block_q5_K *)src, dst, count); break;
    case 14: dequantize_row_q6_K((const block_q6_K *)src, dst, count); break;
    case 15: dequantize_row_q8_K((const block_q8_K *)src, dst, count); break;
    case 16: dequantize_row_iq2_xxs((const block_iq2_xxs *)src, dst, count); break;
    case 17: dequantize_row_iq2_xs((const block_iq2_xs *)src, dst, count); break;
    case 18: dequantize_row_iq3_xxs((const block_iq3_xxs *)src, dst, count); break;
    case 19: dequantize_row_iq1_s((const block_iq1_s *)src, dst, count); break;
    case 20: dequantize_row_iq4_nl((const block_iq4_nl *)src, dst, count); break;
    case 21: dequantize_row_iq3_s((const block_iq3_s *)src, dst, count); break;
    case 22: dequantize_row_iq2_s((const block_iq2_s *)src, dst, count); break;
    case 23: dequantize_row_iq4_xs((const block_iq4_xs *)src, dst, count); break;
    case 29: dequantize_row_iq1_m((const block_iq1_m *)src, dst, count); break;
    case 34: dequantize_row_tq1_0((const block_tq1_0 *)src, dst, count); break;
    case 35: dequantize_row_tq2_0((const block_tq2_0 *)src, dst, count); break;
    case 39: dequantize_row_mxfp4((const block_mxfp4 *)src, dst, count); break;
    case 40: dequantize_row_nvfp4((const block_nvfp4 *)src, dst, count); break;
    case 41: dequantize_row_q1_0((const block_q1_0 *)src, dst, count); break;
    case 42: dequantize_row_q2_0((const block_q2_0 *)src, dst, count); break;
    default: break;
    }
}
"""

MAIN = r"""
// Reads raw blocks on stdin and prints llama.cpp's decoded floats.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ggml_abort(const char *file, int line, const char *fmt, ...) {
    (void) file; (void) line; (void) fmt;
    abort();
}
// Only the quantizers reference these, and the harness never quantizes.
size_t ggml_row_size(int type, long long ne) { (void) type; (void) ne; return 0; }
size_t ggml_type_size(int type) { (void) type; return 0; }
const char * ggml_type_name(int type) { (void) type; return "oracle"; }

extern void fm_dequantize_row(int type, const void * src, float * dst, long long count);

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: type count block_size block_bytes\n"); return 2; }
    const int type = atoi(argv[1]);
    const long long count = atoll(argv[2]);
    const long long bs = atoll(argv[3]);
    const long long bb = atoll(argv[4]);
    const long long nblocks = count / bs;
    unsigned char * buf = malloc((size_t) (nblocks * bb));
    if (fread(buf, (size_t) bb, (size_t) nblocks, stdin) != (size_t) nblocks) {
        fprintf(stderr, "short read\n");
        return 3;
    }
    float * out = calloc((size_t) count, sizeof(float));
    fm_dequantize_row(type, buf, out, count);
    for (long long i = 0; i < count; ++i) printf("%.9g\n", out[i]);
    return 0;
}
"""

VERSION_HEADER = '#define GGML_VERSION "local"\n#define GGML_COMMIT "local"\n'


def load_reference():
    spec = importlib.util.spec_from_file_location("quant_reference", ROOT / "tools" / "quant_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_harness(upstream: Path, work: Path) -> Path:
    ggml = upstream / "ggml"
    sources = ggml / "src"
    if not (sources / "ggml-quants.c").exists():
        raise SystemExit(f"{sources} does not look like a llama.cpp checkout")

    (work / "ggml-version.h").write_text(VERSION_HEADER)
    (work / "bridge.cpp").write_text(BRIDGE)
    (work / "main.c").write_text(MAIN)
    includes = ["-I", str(ggml / "include"), "-I", str(sources), "-I", str(work)]

    subprocess.run(["gcc", "-O2", *includes, "-c", str(sources / "ggml-quants.c"),
                    "-o", str(work / "quants.o")], check=True)
    subprocess.run(["g++", "-O2", *includes, "-c", str(work / "bridge.cpp"),
                    "-o", str(work / "bridge.o")], check=True)
    harness = work / "harness"
    subprocess.run(["gcc", "-O2", *includes, str(work / "main.c"),
                    str(work / "bridge.o"), str(work / "quants.o"), "-o", str(harness), "-lm"],
                   check=True)
    return harness


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True, help="path to a llama.cpp checkout")
    parser.add_argument("--blocks", type=int, default=3, help="blocks per type")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    reference = load_reference()
    # Types that are element formats (one value per block) have no packed layout
    # to check, and 143 has no verified layout to compare against.
    # Q8_1 only exists upstream as an activation format with no reference
    # dequantizer; PQ2_0/PTQ1_0 are extensions with no upstream decoder.
    skip = {0, 1, 9, 24, 25, 26, 27, 28, 30, 142, 143}

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        harness = build_harness(Path(args.upstream), work)
        state = 0x5EED_1234
        for type_id in sorted(reference.GEOMETRY):
            if type_id in skip or type_id not in reference.DECODERS:
                continue
            block_size, block_bytes = reference.GEOMETRY[type_id]
            stream = bytearray()
            expected: list[float] = []
            for _ in range(args.blocks):
                raw, state = reference.make_block(type_id, state)
                stream += raw
                expected += reference.DECODERS[type_id](raw)
            count = block_size * args.blocks
            run = subprocess.run(
                [str(harness), str(type_id), str(count), str(block_size), str(block_bytes)],
                input=bytes(stream),
                capture_output=True,
            )
            if run.returncode != 0:
                results.append({"type_id": type_id, "status": "error",
                                "detail": run.stderr.decode()[:200]})
                continue
            got = [float(line) for line in run.stdout.split()]
            if len(got) != count:
                results.append({"type_id": type_id, "status": "length",
                                "detail": f"{len(got)} != {count}"})
                continue
            # Compare with a relative tolerance scaled to the block's magnitude.
            magnitude = max((abs(v) for v in expected), default=1.0) or 1.0
            worst = max((abs(a - b) for a, b in zip(expected, got)), default=0.0)
            ok = worst <= 1e-6 * magnitude
            results.append({
                "type_id": type_id,
                "status": "pass" if ok else "fail",
                "worst_abs": worst,
                "reference_magnitude": magnitude,
            })

    failures = [r for r in results if r["status"] != "pass"]
    if args.json:
        print(json.dumps({"checked": len(results), "failures": len(failures),
                          "results": results}, indent=2))
    else:
        for row in results:
            print(f"type {row['type_id']:3d}  {row['status']:5s}  "
                  f"{row.get('worst_abs', 0.0):.3e}")
        print(f"{len(results)} types checked, {len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
