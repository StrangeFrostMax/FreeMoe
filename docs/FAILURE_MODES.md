# Failure modes and typed errors

Every runtime failure is a `fm::Error` with a symbolic code, a human message and
a fixed CLI exit code. Malformed model fixtures may produce any typed error;
the manifest requires a non-OK code and a non-empty reason.

| Code | Exit | Typical cause | Required behaviour |
|---|---:|---|---|
| `MALFORMED_GGUF` | 10 | bad magic/version, truncation, oversized arrays, bad alignment, unknown quant type, overlapping/out-of-range tensor extents | reject file before allocation; no partial model object. |
| `UNSUPPORTED_ARCHITECTURE` | 11 | architecture/adapter mismatch | name the architecture; do not guess. |
| `UNSUPPORTED_QUANTISATION` | 11 | executable tensor is not F32 | reject adapter. |
| `INVALID_TOKENIZER` / `METADATA_MISMATCH` | 11 | unknown tokenizer/chat template, inconsistent expert metadata | reject model. |
| `MISSING_REQUIRED_TENSOR` / `UNKNOWN_TENSOR` / `DUPLICATE_TENSOR` / `SHAPE_MISMATCH` | 12 | role/shape/directory validation | name the tensor or shape. |
| `SOURCE_CHANGED` | 13 | size/inode/mtime/ctime/sparse content probe mismatch before a read batch | stop generation; do not reuse stale index. |
| `IO_ERROR` / `SHORT_READ` / `DIRECT_IO_FALLBACK` | 14 | read syscall failure, EOF, forced direct-I/O fallback | explicit error or logged fallback metric; never return partial expert bytes. |
| `ALLOCATION_FAILED` / `BUDGET_TOO_SMALL` | 15 | host/device allocation failure or unsafe budget | print reservations; refuse startup. |
| `INVALID_EXPERT_ID` / `EXPERT_CHECKSUM_MISMATCH` / `EXPERT_EXECUTION_MISMATCH` | 16 | bad router ID, corrupt source/slot payload, selected/executed count mismatch | stop generation; never substitute weights or skip experts. |
| `SLOTS_BUSY` | 17 | all slots referenced or pending after `T_wait` | return typed error, do not overwrite live weights. |
| `STALE_COMPLETION` | 18 | generation mismatch or unpublished upload | discard completion; leave slot state unchanged. |
| `TRANSFER_FAILED` / `FENCE_TIMEOUT` / `DEVICE_LOSS` | 19 | injected or backend-reported H2D/fence/device failure | retire fence before reuse; invalidate slots on device loss. |
| `GPU_BACKEND_UNIMPLEMENTED` / `CPU_FALLBACK_DISABLED` | 20 | GPU path requested without a real backend | fail loudly unless `--allow-cpu-fallback`; then log every fallback op. |
| `INVALID_ARGUMENT` | 2 | bad CLI option, prompt too long, invalid extent | reject request. |
| `NOT_IMPLEMENTED` | 21 | reserved for explicit future gaps | no silent approximation. |
| `INTERNAL_ERROR` | 70 | invariant violation or unexpected exception | stop; never continue with questionable state. |

## Cancellation and asynchronous safety

- A prefetch-only request can be cancelled; a shared request with a demand
  subscriber is never cancelled by prefetch pressure.
- Demand reads execute synchronously with priority; stale predictions are
  cancelled by age or queue pressure and counted as wasted bytes.
- Slot generation IDs prevent a late completion for expert A from publishing a
  slot now assigned to expert B.
- Uploads publish only after a fence completes; transfer/fence failures move
  the slot to `ERROR` before any reuse.
- Shutdown drains/cancels futures, releases backend buffers after the residency
  manager, and writes a final allocation snapshot with zero live allocations.

## Fault-injection coverage

`tests/unit/tests.cpp` injects transfer failure, fence timeout, deferred fence,
device loss, stale completion, all-slots-busy, referenced-slot eviction,
source-file modification and checksum cycling through a one-slot pool. The
malformed fixture suite exercises 54 files under `tests/malformed_models/`.
The ASan/UBSan fuzz target mutates the dense fixture for the declared `F`
duration.
