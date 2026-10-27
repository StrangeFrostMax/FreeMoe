# Patches

Small, self-contained fixes for the source tree shipped in `Maxnpuxl.zip.zip`.
Each lives in `patches/` as a `-p1` unified diff against the extracted archive,
and each states how to reproduce the defect and verify the fix.

---

## 0001 — Chrome-trace timestamps are written on a 10 ms grid

**File:** `src/core/runtime.cpp` (`write_trace`)
**Apply from:** the directory containing `src/`

### Symptom

Traces come out with every command inside a 10 ms window sharing a single start
timestamp. The timeline therefore cannot answer the questions a timeline exists
to answer — where the gaps are, what overlaps, what the duty cycle is — while
the durations look perfectly healthy, which is why this survives a casual look
at the file.

Observed from an LLM engine driving the library: a 928-command run produced
**29 distinct start timestamps, all exactly 10000 µs apart**, with the durations
inside each 10 ms slot summing to just under 10000 µs.

### Cause

One line:

```cpp
out << "\",\"ph\":\"X\",\"ts\":" << (e.start_ns / 1000.0)
```

`std::ostream` keeps its default precision of **6 significant digits**. A
monotonic timestamp in microseconds near 1.2e9 is therefore emitted as
`1.1953e+09`, whose least significant digit *is* 10 ms. Durations are small
numbers, so the same six digits are plenty for them — which is exactly why only
the start times were destroyed and the trace still looked plausible.

It is not the clock. `steady_clock` resolves changes 100–200 ns apart on the
same machine.

### Reproducer and evidence

Same tree, same command, only this patch differing:

```sh
make -j
./bin/vnpu-cli trace --shape 512x512x512 --output trace.json
grep -o '"ts":[0-9.e+-]*' trace.json | head -3
```

| | first timestamp | distinct starts | on a 10 ms grid |
| --- | --- | ---: | ---: |
| before | `"ts":1.81113e+09` | 1 | 1/1 |
| after | `"ts":0.000` | 3 | 0/3 |

The three commands in that trace really do start ~2.4 ms apart; before the fix
all three collapse onto one instant.

### Fix

Rebase every timestamp onto the first event and pin the formatting. Rebasing
removes the 1e9 magnitude the significant-digit rule was eating; `std::fixed`
with 3 decimals keeps nanosecond resolution regardless of magnitude.

```cpp
uint64_t t0 = copy.empty() ? 0u : copy[0].start_ns;
for (const auto &e : copy) t0 = std::min(t0, e.start_ns);
out << std::fixed << std::setprecision(3);
...
<< "\",\"ph\":\"X\",\"ts\":" << ((e.start_ns - t0) / 1000.0)
```

Relative timestamps are also what Perfetto and `chrome://tracing` expect for a
single-process trace, so this removes a second problem at the same time: the
absolute value was ~1.2e9 µs, i.e. 20 minutes of uptime, as the first thing
every reader had to scroll past.

### Verification

- `make -j` builds clean under the project's own `-Wall -Wextra`; no new
  diagnostics in `runtime.cpp`.
- `./bin/vnpu-selftest` → **16 tests, 0 failures**.
- Trace output as in the table above.

Only the trace writer is touched; nothing on a compute path changes.

### Note for consumers

The prebuilt `lib/libvnpu.a` and `lib/libvnpu.so` committed in the archive were
built before this fix, so a consumer linking those binaries keeps the 10 ms grid
until they rebuild from patched source.

---

## What is in this archive

`npuxl-improved-patched.zip` is the complete tree from `Maxnpuxl.zip.zip` with
patch 0001 applied. It differs from upstream in exactly two files:

```
src/core/runtime.cpp      +13 -1   (the fix)
PATCHES.md                new      (this file)
```

Nothing else is touched. The prebuilt `lib/libvnpu.a` and `lib/libvnpu.so` are
the **originals from the archive**, not rebuilt here — they are glibc-specific
and shipping rebuilt ones would only move the problem to someone else's machine.
So a consumer who links those binaries keeps the 10 ms grid until they rebuild
from this source (that is stated in the note under 0001 as well).

Build and verify the same way:

```sh
make -j
./bin/vnpu-selftest                                    # 16 tests, 0 failures
./bin/vnpu-cli trace --shape 512x512x512 --output t.json
grep -o '"ts":[0-9.e+-]*' t.json | head -3             # starts at 0.000
```
