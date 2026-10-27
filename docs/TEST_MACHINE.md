# D1 test machine record

Recorded on 2026-10-27. Two profiles are distinguished: the **validation
sandbox** used to build/test this repository, and the user's **target CPU
profile** (Zen 3) whose exact machine is not yet specified.

## Validation sandbox

| Field | Value |
|---|---|
| OS | Linux `e2b.local 6.1.158+ #1 SMP PREEMPT_DYNAMIC Mon May 11 18:48:24 UTC 2026 x86_64 GNU/Linux` |
| CPU | Intel(R) Xeon(R) Processor @ 2.60GHz, 2 logical CPUs, KVM guest |
| ISA flags used | AVX2 + FMA3 forced; AVX-512 disabled for FreeMoe backend tests |
| RAM | 3.8 GiB total, about 3.6 GiB available at record time |
| Compiler | `g++ (Debian 12.2.0-14+deb12u1) 12.2.0` |
| Physical GPU | none present in sandbox; model/driver/VRAM **unknown (D1)** |
| Virtual NPU | vendored CPU software device; no dedicated VRAM |
| Storage | container overlay filesystem; direct-I/O support/fallback recorded per run |

### `lscpu` output (abridged, non-ISA vulnerability rows omitted)

```text
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 48 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  2
On-line CPU(s) list:                     0,1
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Processor @ 2.60GHz
CPU family:                              6
Model:                                   106
Thread(s) per core:                      2
Core(s) per socket:                      1
Socket(s):                               1
Stepping:                                6
BogoMIPS:                                5200.02
Flags:                                   ... avx2 ... avx512f ... avx512_vnni ...
Hypervisor vendor:                       KVM
Virtualization type:                     full
L1d cache:                               48 KiB (1 instance)
L1i cache:                               32 KiB (1 instance)
L2 cache:                                1.3 MiB (1 instance)
L3 cache:                                54 MiB (1 instance)
NUMA node(s):                            1
NUMA node0 CPU(s):                       0,1
```

### `free -h`

```text
               total        used        free      shared  buff/cache   available
Mem:           3.8Gi       260Mi       3.5Gi       1.0Mi       274Mi       3.6Gi
Swap:             0B          0B          0B
```

### Device query output

`./build/freemoe device-info --backend vnpu --json`:

```json
{"name":"Vendored Virtual NPU 1.0 (patched)","backend":"AVX2+FMA dual-pipe(forced)","memory_type":"host_ram (virtual device buffers)","api_version":"VNPU ABI 1","driver_version":"vendored C++ runtime","notes":"Virtual NPU is a CPU software device; no dedicated VRAM. Dual independent 256-bit AVX2/FMA3 accumulators enabled. ","total_memory_bytes":4131278848,"free_memory_bytes":3789348864,"dedicated_vram_bytes":0,"simd_bits":256,"threads":2}
```

`./build/freemoe device-info --backend cpu --json`:

```json
{"name":"FreeMoe scalar CPU reference","backend":"Intel(R) Xeon(R) Processor @ 2.60GHz","memory_type":"host_ram","api_version":"C++17 scalar","driver_version":"none","notes":"Correctness oracle; no vendor acceleration.","total_memory_bytes":4131278848,"free_memory_bytes":3789348864,"dedicated_vram_bytes":0,"simd_bits":0,"threads":1}
```

The GPU placeholder reports `dedicated_vram_bytes=0` and `backend="not
implemented"`. This is not a successful physical GPU query.

## Target profile

The user identified the target CPU family as **Zen 3** and correctly noted
that its AVX-512 alternative is dual 256-bit AVX2/FMA3 execution units.
The exact CPU model, RAM size, NVMe model, OS version, GPU/NPU model, driver
version and usable VRAM are **unknown (D1)**. Consequently:

- no Zen 3 or 12 GiB GPU performance number is claimed;
- the physical-GPU backend gate remains closed;
- benchmark results produced on this sandbox are labelled process-cold with
  unknown OS cache state and are not generalised to the target machine.
