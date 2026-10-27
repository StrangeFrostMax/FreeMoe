# GGUF format and quantisation handling

## Source contract

FreeMoe reads an unchanged GGUF v3 file. It does not write a custom deployment
container. The parser is memory-safe and operates either on a read-only mapped
file or an in-memory byte slice (used by fuzzing).

Header fields parsed, in order:

1. magic `GGUF`;
2. `uint32` version (only 3 accepted);
3. `uint64` tensor count;
4. `uint64` metadata count;
5. metadata key/value records;
6. tensor descriptors (name, dimension count, dimensions, type, relative offset);
7. an aligned data section.

`general.alignment` must be a power of two no larger than 1 MiB. The data
section starts after **both** metadata and the tensor directory. Tensor offsets
are relative to that data section, must respect alignment and must not overlap.

## Metadata value types

The parser supports GGUF value types 0–12: UINT8, INT8, UINT16, INT16, UINT32,
INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64 and FLOAT64. Arrays may
contain scalars, strings or one nested array level (limited to depth 4).
`tests/fixtures/metadata-types.gguf` exercises every type. Oversized strings,
arrays, dimension products and offsets are rejected before allocation.

## Quantisation geometry

The parser knows the block geometry of these GGUF types so it can compute
tensor byte sizes and validate row divisibility:

| Type ID | Name | Block elements | Block bytes | Inference |
|---:|---|---:|---:|---|
| 0 | F32 | 1 | 4 | yes |
| 1 | F16 | 1 | 2 | no |
| 2 | Q4_0 | 32 | 18 | no |
| 3 | Q4_1 | 32 | 20 | no |
| 6 | Q5_0 | 32 | 22 | no |
| 7 | Q5_1 | 32 | 24 | no |
| 8 | Q8_0 | 32 | 34 | no |
| 9 | Q8_1 | 32 | 40 | no |
| 10–15 | Q2_K…Q8_K | 256 | 84/110/144/176/210/292 | no |
| 30 | BF16 | 1 | 2 | no |

Only F32 tensors are executable. `tests/fixtures/quant-blocks.gguf` proves
that Q8_0 size arithmetic is correct (two 32-element blocks = 68 bytes), and a
matching malformed fixture proves a 31-element row is rejected. Unknown type
IDs are hard errors, not guesses.

## Expert records

For `freemoe-tiny-v1` MoE, gate/up/down expert weights are stacked F32 tensors.
Each expert record is the concatenation of three independently validated byte
ranges:

```text
gate[expert] || up[expert] || down[expert]
```

The per-expert stride is computed from `ne[0] * ne[1] * bytes_per_element`;
naive `expert_index * logical_float_bytes` is not used for quantised types.
Each record's FNV-1a 64-bit checksum is stored in the GGUF metadata key
`freemoe.expert_checksums`. The runtime validates the freshly read record and
the uploaded slot payload against that checksum.

## Source identity and integrity

`FileIdentity` records size, device, inode, mtime, ctime and a sparse
FNV-1a probe of the first/middle/last 64 bytes. Before each read batch, the
scheduler compares the current identity with the value recorded at index
creation. A mismatch produces `SOURCE_CHANGED`. The probe is not cryptographic;
it catches payload edits on coarse-timestamp filesystems without reading the
whole model before every batch. Expert record checksums provide per-record
integrity. The metadata/directory prefix hash is reported as
`source_fingerprint` in `model inspect`.

## Direct I/O

On Linux the direct reader opens with `O_DIRECT`, derives block alignment from
`fstatvfs`, rounds the requested range down/up to aligned boundaries, uses a
`posix_memalign` bounce buffer and copies the useful subrange. Short reads and
unsupported alignment fall back to buffered I/O with an explicit
`direct_io_fallback` trace event and metric. Windows
`FILE_FLAG_NO_BUFFERING` is not implemented; Windows builds use buffered/mmap
paths and must label that fallback explicitly in results.
