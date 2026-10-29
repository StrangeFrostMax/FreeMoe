#!/usr/bin/env python3
"""Reproduce the Zen 3 detection fallback evidence on any x86 CI host.

The machine does not have to be Zen 3: `--test-simulate-zen3` forces the same
decision branch that `znver3` detection would take. This script writes a
compact JSON verdict plus a detailed command log under `docs/evidence/`.
"""
from __future__ import annotations

import argparse
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


def simulated_checks(info: dict, trace: dict, requested_profile: str,
                     real_zen3: bool = False) -> dict:
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
        # `--test-simulate-zen3` cannot turn genuine detection into a
        # simulation: on a real Zen 3 host the runtime reports the detection as
        # real, and that is the correct answer, not a failure.
        "trace_detection_simulated":
            trace.get("zen3_detection_simulated") is (not real_zen3),
        "trace_fallback_selected": trace.get("zen3_fallback_selected") is True,
        "trace_requested_profile": trace.get("requested_profile") == requested_profile,
        "trace_selected_attempt": trace.get("selected_attempt") == "Zen 3 AVX2 dual-pipe fallback",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the committed evidence is current without rewriting it")
    args = parser.parse_args()
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
        last_json(sim_auto.stdout), backend_init_fields(sim_auto_trace), "zen3-auto",
        real_zen3=host_zen3)
    sim_avx512_checks = simulated_checks(
        last_json(sim_avx512.stdout), backend_init_fields(sim_avx512_trace), "avx512",
        real_zen3=host_zen3)

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
    log_lines.append("[verdict]\n" + json_text.rstrip("\n") + "\n")
    log_text = "\n".join(log_lines).rstrip("\n") + "\n"

    if args.check:
        # The evidence is committed, so it can rot: this check exists because it
        # did. `zen3_fallback.json` recorded 61 unit checks long after the test
        # had 64, and no gate noticed, so the repository claimed evidence from a
        # run that no longer existed. Compare the regenerated bytes instead of
        # trusting the file.
        # Only the verdict is comparable: the log also carries `device-info`
        # output, whose free-memory figures change on every run (which is what
        # made a byte comparison of the whole log fail immediately). The verdict
        # block is the part that records the check counts, and it is the part a
        # reader quotes.
        committed_text = JSON_PATH.read_text(encoding="utf-8") if JSON_PATH.exists() else ""
        current_log = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else ""
        marker = "[verdict]\n"
        index = current_log.rfind(marker)
        log_verdict_text = current_log[index + len(marker):] if index >= 0 else ""

        # Some of the verdict is a property of the *host*, not of the code: on a
        # machine that really is Zen 3 the unit test runs two extra checks, and
        # the non-simulated diagnostics are verified against genuine detection
        # instead of against "not detected". Byte-comparing those on a
        # different machine reports drift that is not there -- exactly what
        # happened on the first CI run of this check, on a Zen 3 runner, while
        # the file was correct.
        # The detection-state check is the other derived value: on a real Zen 3
        # host the simulator cannot make the detection synthetic, so the field
        # holds the opposite (correct) answer there.
        host_dependent = {"host_zen3_detected", "unit_checks", "non_simulated_checks"}

        def drop_host_dependent(payload: dict) -> dict:
            out: dict = {}
            for key, value in payload.items():
                if key in host_dependent:
                    continue
                if isinstance(value, dict):
                    value = {k: v for k, v in value.items()
                             if not k.endswith("trace_detection_simulated")}
                out[key] = value
            return out

        def load(text: str) -> dict | None:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return None

        def comparable(payload: dict) -> dict:
            return drop_host_dependent(payload)

        committed = load(committed_text)
        logged = load(log_verdict_text)
        same_host = (committed or {}).get("host_zen3_detected") == host_zen3
        if same_host:
            problems = []
            if committed_text != json_text:
                problems.append(str(JSON_PATH.relative_to(ROOT)))
            if index < 0 or log_verdict_text != json_text:
                problems.append(str(Path(str(LOG_PATH.relative_to(ROOT)) + " (verdict block)")))
        else:
            # Different host class: compare what the host cannot influence and
            # say what was skipped, so the line is not read as full verification.
            problems = []
            if committed is None or comparable(committed) != comparable(result):
                problems.append(str(JSON_PATH.relative_to(ROOT)))
            if logged is None or comparable(logged) != comparable(result):
                problems.append(str(Path(str(LOG_PATH.relative_to(ROOT)) + " (verdict block)")))
        if problems:
            for path in problems:
                print(f"{path} is stale; run make zen3-fallback", file=sys.stderr)
            return 1
        if same_host:
            print(f"zen3 fallback evidence is up to date ({result.get('status', '?')}, "
                  f"{result.get('unit_checks', -1)} unit checks)")
        else:
            print(f"zen3 fallback evidence matches on host-independent fields "
                  f"({result.get('status', '?')}); the recorded run was on a "
                  f"{'Zen 3' if (committed or {}).get('host_zen3_detected') else 'non-Zen-3'} "
                  f"host and this one is not, so unit_checks and the non-simulated "
                  f"diagnostics are not compared")
        return 0 if status == "pass" else 1

    JSON_PATH.write_text(json_text, encoding="utf-8")
    PHASE0.mkdir(parents=True, exist_ok=True)
    (PHASE0 / "zen3_fallback.json").write_text(json_text, encoding="utf-8")
    LOG_PATH.write_text(log_text, encoding="utf-8")
    print(f"zen3 fallback check: {status.upper()}; evidence in docs/evidence/")
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
