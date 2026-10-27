#!/usr/bin/env python3
"""Reproduce the Zen 3 detection fallback evidence on any x86 CI host.

The machine does not have to be Zen 3: `--test-simulate-zen3` forces the same
decision branch that `znver3` detection would take. This script writes a
compact JSON verdict plus a detailed command log under `docs/evidence/`.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"
EVIDENCE = ROOT / "docs" / "evidence"
PHASE0 = BUILD / "phase0"
TRACES = BUILD / "test-tmp"
LOG_PATH = EVIDENCE / "zen3_fallback.log"
JSON_PATH = EVIDENCE / "zen3_fallback.json"

log_lines: list[str] = []


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    log_lines.append("$ " + " ".join(command))
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.stdout:
        log_lines.append(proc.stdout.rstrip("\n"))
    if proc.stderr:
        log_lines.append("[stderr]\n" + proc.stderr.rstrip("\n"))
    log_lines.append(f"[exit_code] {proc.returncode}\n")
    return proc


def last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.strip().startswith("{"):
            return json.loads(line)
    raise ValueError("no JSON object found in command output")


def backend_init_fields(path: Path) -> dict:
    fields: dict = {}
    if not path.exists():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("event") == "backend_init":
            fields = event
    return fields


def simulated_checks(info: dict, trace: dict, requested_profile: str) -> dict:
    notes = info.get("notes", "")
    backend = info.get("backend", "")
    return {
        "simd_bits_256": info.get("simd_bits") == 256,
        "backend_has_no_avx512": "AVX512" not in backend,
        "backend_names_dual_pipe": "dual-pipe" in backend,
        "notes_zen3_detected": "Zen 3 detected" in notes and "Zen 3 not detected" not in notes,
        "notes_avx512_disabled": "AVX-512 dispatch disabled" in notes,
        "notes_fallback_selected": "Selected VNPU dispatch: Zen 3 AVX2 dual-pipe fallback" in notes,
        "trace_zen3_detected": trace.get("zen3_detected") is True,
        "trace_detection_simulated": trace.get("zen3_detection_simulated") is True,
        "trace_fallback_selected": trace.get("zen3_fallback_selected") is True,
        "trace_requested_profile": trace.get("requested_profile") == requested_profile,
        "trace_selected_attempt": trace.get("selected_attempt") == "Zen 3 AVX2 dual-pipe fallback",
    }


def main() -> int:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    TRACES.mkdir(parents=True, exist_ok=True)

    binaries = [BUILD / "freemoe", BUILD / "tests", BUILD / "vnpu-selftest"]
    if not all(path.exists() for path in binaries):
        build = run(["make", "-j2"])
        if build.returncode != 0:
            log_lines.append("FATAL: build failed\n")

    unit = run(["./build/tests", "zen3_fallback"])
    vendor = run(["./build/vnpu-selftest", "zen3_detection_fallback"])

    non_sim_trace = TRACES / "zen3-auto-cli-trace.jsonl"
    sim_auto_trace = TRACES / "zen3-auto-simulated-cli-trace.jsonl"
    sim_avx512_trace = TRACES / "avx512-simulated-cli-trace.jsonl"

    non_sim = run([
        "./build/freemoe", "device-info", "--backend", "vnpu",
        "--vnpu-isa", "zen3-auto", "--json", "--trace", str(non_sim_trace.relative_to(ROOT)),
    ])
    sim_auto = run([
        "./build/freemoe", "device-info", "--backend", "vnpu",
        "--vnpu-isa", "zen3-auto", "--test-simulate-zen3", "--json",
        "--trace", str(sim_auto_trace.relative_to(ROOT)),
    ])
    sim_avx512 = run([
        "./build/freemoe", "device-info", "--backend", "vnpu",
        "--vnpu-isa", "avx512", "--test-simulate-zen3", "--json",
        "--trace", str(sim_avx512_trace.relative_to(ROOT)),
    ])

    unit_match = re.search(r"(\d+) checks, (\d+) failures", unit.stdout)
    vendor_match = re.search(r"(\d+) tests, (\d+) failures", vendor.stdout)
    unit_pass = (
        unit.returncode == 0 and "[PASS] zen3_fallback" in unit.stdout and
        unit_match is not None and unit_match.group(2) == "0" and
        int(unit_match.group(1)) > 0
    )
    vendor_pass = (
        vendor.returncode == 0 and "[PASS] zen3_detection_fallback" in vendor.stdout and
        vendor_match is not None and vendor_match.group(2) == "0"
    )

    command_rcs_ok = all(proc.returncode == 0 for proc in
                         (unit, vendor, non_sim, sim_auto, sim_avx512))
    non_sim_info = last_json(non_sim.stdout)
    non_sim_fields = backend_init_fields(non_sim_trace)
    non_sim_notes = non_sim_info.get("notes", "")
    host_zen3 = "Zen 3 detected" in non_sim_notes and "Zen 3 not detected" not in non_sim_notes
    non_sim_checks = {
        "diagnostics_present": (
            "Selected VNPU dispatch:" in non_sim_notes and
            ("Zen 3 detected" in non_sim_notes or "Zen 3 not detected" in non_sim_notes)
        ),
        "trace_records_detection_state": (
            non_sim_fields.get("zen3_detected") is host_zen3 and
            non_sim_fields.get("zen3_detection_simulated") is False
        ),
        "real_zen3_uses_256bit": (not host_zen3 or non_sim_info.get("simd_bits") == 256),
        "real_zen3_has_no_avx512": (
            not host_zen3 or "AVX512" not in non_sim_info.get("backend", "")
        ),
    }

    sim_auto_checks = simulated_checks(
        last_json(sim_auto.stdout), backend_init_fields(sim_auto_trace), "zen3-auto")
    sim_avx512_checks = simulated_checks(
        last_json(sim_avx512.stdout), backend_init_fields(sim_avx512_trace), "avx512")

    checks = {
        "all_commands_exit_zero": command_rcs_ok,
        "unit_zen3_fallback_pass": unit_pass,
        "vendor_zen3_detection_fallback_pass": vendor_pass,
        "non_simulated_diagnostics": all(non_sim_checks.values()),
        **{f"simulated_zen3_auto_{k}": v for k, v in sim_auto_checks.items()},
        **{f"simulated_avx512_{k}": v for k, v in sim_avx512_checks.items()},
    }
    status = "pass" if all(checks.values()) else "fail"
    result = {
        "status": status,
        "note": "Zen 3 detection is simulated with --test-simulate-zen3 on non-Zen-3 hosts.",
        "host_zen3_detected": host_zen3,
        "unit_checks": int(unit_match.group(1)) if unit_match else 0,
        "unit_failures": int(unit_match.group(2)) if unit_match else -1,
        "vendor_tests": int(vendor_match.group(1)) if vendor_match else 0,
        "vendor_failures": int(vendor_match.group(2)) if vendor_match else -1,
        "non_simulated_checks": non_sim_checks,
        "simulated_zen3_auto_checks": sim_auto_checks,
        "simulated_avx512_checks": sim_avx512_checks,
        "checks": checks,
        "evidence_log": "docs/evidence/zen3_fallback.log",
    }

    json_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    JSON_PATH.write_text(json_text, encoding="utf-8")
    PHASE0.mkdir(parents=True, exist_ok=True)
    (PHASE0 / "zen3_fallback.json").write_text(json_text, encoding="utf-8")
    log_lines.append("[verdict]\n" + json_text.rstrip("\n") + "\n")
    LOG_PATH.write_text("\n".join(log_lines).rstrip("\n") + "\n", encoding="utf-8")
    print(f"zen3 fallback check: {status.upper()}; evidence in docs/evidence/")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
