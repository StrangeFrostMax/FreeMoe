# Benchmark and correctness protocol

Status: committed provisional protocol for the **synthetic correctness
prototype**. This is not a performance-qualified release and does not claim
support for production GGUF checkpoints.

## Reference

| Field | Value |
|---|---|
| Reference runtime | Independent scalar Python oracle, `tools/reference_oracle.py` |
| D5 pinned tree | `deee8aa67b9615a0ef8cd73aaa4c8aca6469cfa5` |
| Oracle file SHA-256 | `bd5af7463c47139e08d8a3ec07e0cf8f196d2aa4368b03b0dc9bd3e262b2dfa7` |
| Fixture generator SHA-256 | `3b06ba80ae618074d6aefd1665ede19692be56b9c5f92cacbc3436868dea435c` |
| Full reference manifest SHA-256 | `6b946e2c48097d5d55e945c4772db3e24dd847318d4ec9fa63c393a55d38158a` |
| Python | 3.11.2 standard library only |
| Build command | none (Python source is executed directly) |
| Compiler/build flags | not applicable |
| Reference output command | `python3 tools/make_fixtures.py --all` |
| Reference logits files | `tests/reference/<kind>/prompt_XX.logits` |
| Reference token/routing files | `tests/reference/<kind>/prompt_XX.greedy.ids`, `tests/reference/<kind>/prompt_XX.routing.txt` |

`tests/reference/manifest.sha256` is the authoritative SHA-256 list for every
reference artefact, including logits and routing traces. The D8 prompt and ID
file hashes are also reproduced below.

## Tolerances (D4)

| Data type | Absolute tolerance | Relative tolerance | Used by |
|---|---:|---:|---|
| F32 logits | 0.005 | 0.005 | D2 dense and D3 MoE reference comparisons |

A comparison passes when `abs(actual-reference) <= abs_tol + rel_tol *
abs(reference)` for every logit, greedy token IDs match for `N_greedy`, and MoE
routing IDs match for every layer/token. F16/Q8_0 and K-quants are parsed and
size-checked but are not executable inference types in this release. Any
tolerance change requires a commit whose message states the reason and date.

## Parameters (D7)

| Parameter | Value | Notes |
|---|---:|---|
| `N_greedy` | 16 | generated tokens compared per prompt |
| `N_soak` | 256 | tokens per soak session; three consecutive sessions |
| `F` | 0.02 h (72 s) | declared ASan/UBSan fuzz duration |
| `T_wait` | 100 ms | slot wait/fence deadline in benchmark runs |
| `G_bytes` | 4096 | maximum gap for read coalescing |
| `R` | 5 | exploratory benchmark repetitions |

Deterministic unit tests override `T_wait` to 5 ms to keep fault-injection
tests fast. Benchmark and soak commands use the committed value unless a config
explicitly declares otherwise.

## Provisional performance target

The user delegated parameter choice on 2026-10-27. The following provisional
targets apply only to the synthetic D3 MoE fixture on the recorded sandbox:

- median prefill throughput >= 1000 prompt tokens/s;
- median p95 inter-token latency <= 50 ms;
- 20/20 prompt correctness comparisons pass;
- no invalid state transitions, checksum failures or unbounded allocations.

These are project inputs, not measured hardware claims. The exact target
Zen 3 CPU model, RAM, storage, GPU/NPU SDK, driver and usable VRAM are **unknown
(D1)**, so the performance-qualified release gate remains closed until a real
D1 machine record is committed and benchmarked.

## Benchmark controls

- Fix model, quantisation, prompts, generated token count, context, backend,
  reader, cache slots, replacement policy, prefetch, queue depth, coalescing
  gap and thread count in `configs/benchmark.env`.
- Record commit, compiler, OS, CPU, RAM, backend/API/driver and device query.
- Separate model load, prefill and decode in `results/<run-id>/report.json`.
- Run `R=5` repetitions and report median plus min/max spread using
  `scripts/summarize_benchmark.py`.
- Validate generated tokens/routing and logits; do not accept throughput alone.
- No benchmark number from a paper, upstream issue or different backend is
  transplanted into this project's results.

## Cold and warm protocol

- A process restart is **not** an OS-cold run.
- The sandbox cannot safely control OS page cache. Its runs are labelled
  `process-cold first run; OS cache state unknown`.
- Warm runs first read the model in a separate unmeasured warming pass and
  record the method. Cold/warm run sets are never mixed in one comparison.
- Runs with unknown cache state are labelled `unknown` and excluded from
  cold-versus-warm comparisons.
- Failed runs remain in `results/` and are listed by
  `scripts/update_results_index.py` in `docs/RESULTS_INDEX.md`.

## Commands

```sh
make clean && make -j2
make test
make fuzz FUZZ_SECONDS=72
make soak
bash scripts/run_record.sh configs/example.env
bash benchmarks/run_repeats.sh configs/benchmark.env
```

## D8 prompt and ID file hashes

| File | SHA-256 |
|---|---|
| `tests/reference/prompt_00.txt` | `ca077a1d608b40280cd357ce937d9fbabb20feea0e7b25960073cc4242bc4974` |
| `tests/reference/prompt_00.input.ids` | `20d7a96c634b4a66cd17f532ca0a3ad159f5dbd9346b4b3014b60a0a1a99f312` |
| `tests/reference/dense/prompt_00.greedy.ids` | `09a37bf388aa54ccc342b6bab8654b7813821d40f288cbf9963741a386a4973e` |
| `tests/reference/moe/prompt_00.greedy.ids` | `7cf4f7fcd00da26d6faf4cab346717d4aa174492feacbafb4732e237c51f10fe` |
| `tests/reference/prompt_01.txt` | `eeecd3a839a0ea8ced33203653aaddf2459cc4e6d222848d9f3a1ffa70e3a250` |
| `tests/reference/prompt_01.input.ids` | `a19188bf500191a62e12ac72f17753ca04f063842cef2cb81025b16c03b91f8d` |
| `tests/reference/dense/prompt_01.greedy.ids` | `9c0f15f0ab4725d2272df8fd7758c5b1a8dd9e1a0eae559d44e4498758058f53` |
| `tests/reference/moe/prompt_01.greedy.ids` | `a5d7640cf49ca111824ada68985a534eb980f43be94c654ba640a2062891fe7f` |
| `tests/reference/prompt_02.txt` | `d6fe99d42077f579d6d84f910d476b6551524972060571b0735d714c04fa110e` |
| `tests/reference/prompt_02.input.ids` | `ef0d10b7ac46913e8e237e3f1bed41e39e54e9b5855f900f4237ad4fe32247c8` |
| `tests/reference/dense/prompt_02.greedy.ids` | `7052ff7b1de546193f51a244b6004a6bf86d9503e92c3db2ab63213df5a83fa7` |
| `tests/reference/moe/prompt_02.greedy.ids` | `3aa6f05e5da96dec32b3fbb1c6c4bb1b7705f9875f188b4c97c0620b42b563e8` |
| `tests/reference/prompt_03.txt` | `0b001ac1cda8515f7d82ad8b0ca8d406727ef20b64d7a7b4d393e2c795da6456` |
| `tests/reference/prompt_03.input.ids` | `c5cf305b97afce9cd81f370756239f571a5d7402a48b72b1924bb7e399c640ad` |
| `tests/reference/dense/prompt_03.greedy.ids` | `85c08e0daa1fc670f4fa4897c9592b57a473d5e232c7f249ab2e22fe1cddd3cc` |
| `tests/reference/moe/prompt_03.greedy.ids` | `1223ffdb8f8647651ece3d79256d3695657d05fd0a4bddba0f426d1508def12a` |
| `tests/reference/prompt_04.txt` | `61e5e97bf214f84603791392e19f43f632140f1686f963580bcda28310da0967` |
| `tests/reference/prompt_04.input.ids` | `ae0029fad1e46461569963e7f2fba9930a00600ff00c8682723ee1569aca7302` |
| `tests/reference/dense/prompt_04.greedy.ids` | `2ecfdf82b6f17d2acd578271facc03cca8f8f61f853d5c18e615a2e75cbcb391` |
| `tests/reference/moe/prompt_04.greedy.ids` | `a37f378e36f4de59e4ba0105f50439d5947e6333f459bba0bfca7224a7104fe8` |
| `tests/reference/prompt_05.txt` | `4e2e3e01101dfa990d853180a3050d6c5778b26f431193b10520aaba268fe8e2` |
| `tests/reference/prompt_05.input.ids` | `e3a7725afad5ec88a8c959fab805c04b83e86b82a23fa1ef1c5cd9e029ad9048` |
| `tests/reference/dense/prompt_05.greedy.ids` | `5c326fb6feae109796e6847ce8967a71a1411a987b12962ac0f08eca571b164e` |
| `tests/reference/moe/prompt_05.greedy.ids` | `190a347a6a78e73045afcf1c5c2084e1ab65570ee05e4cc0c66e7a3fccc04f9f` |
| `tests/reference/prompt_06.txt` | `50c10878ffdebf6571ed158c3e76357575a2bff5d3d3d6ea747165aa747aca66` |
| `tests/reference/prompt_06.input.ids` | `dc4481189247f9f2c87c7e62b5f19800f08f755425a32269b605fd459b29a2b3` |
| `tests/reference/dense/prompt_06.greedy.ids` | `5c326fb6feae109796e6847ce8967a71a1411a987b12962ac0f08eca571b164e` |
| `tests/reference/moe/prompt_06.greedy.ids` | `c8fcd1d05e1a77be3d1cb9239d686758b04f93ba6110f96abf9c0fb841dbee13` |
| `tests/reference/prompt_07.txt` | `3ceb7879c89e76f172eea24a41163f9503b5873609ab97fbef0de2b2520a53cd` |
| `tests/reference/prompt_07.input.ids` | `7fb70c933a1618fb5952e5b3ace9c1945bfb498ed53049257c684dbca428d5ed` |
| `tests/reference/dense/prompt_07.greedy.ids` | `646fa9a27a4dfa6d78fe0c7de936f566e8612eed65117e9092441502d9c95f54` |
| `tests/reference/moe/prompt_07.greedy.ids` | `70af51f31878b65f489217754211527f876b071ba344fd7000aaeb132fd0778a` |
| `tests/reference/prompt_08.txt` | `84ab4aa18405354f98b848b22971532e1c0e953688dc810b2d5186ecb095d636` |
| `tests/reference/prompt_08.input.ids` | `1416d711eca74ac5c819a9a6fe48403121f9023f0be69848a0d442bd3cdacf31` |
| `tests/reference/dense/prompt_08.greedy.ids` | `f964b5e1902efacdbbf673ea853c9da4525296506bbaa2d9716237fc7948f029` |
| `tests/reference/moe/prompt_08.greedy.ids` | `8ac35f52bc3c161c7497078d0c2c8ee630a8e1c9b42df3c8f959e0a9eb57ef4a` |
| `tests/reference/prompt_09.txt` | `9d58c3ff233cc7e3238af874e7b83a1b0e745714bb67cb7b223fb1f5828708f8` |
| `tests/reference/prompt_09.input.ids` | `b04d69ca7ea94280ebdab216853391d479efad6128e457fb7504f26d12800723` |
| `tests/reference/dense/prompt_09.greedy.ids` | `cfbd15fa626bd38a51ce2d9dadd4c2b08aa7416fe0a035b552155eee8b55ee3c` |
| `tests/reference/moe/prompt_09.greedy.ids` | `3f0997b5b479bf3ab9561e8cf9f58b6922843c58e81178cd4037943e077dccdc` |
| `tests/reference/prompt_10.txt` | `93b053396f6a83fdf0b461650a2d523230bcc434eef00ac14130d00fcdc57d12` |
| `tests/reference/prompt_10.input.ids` | `9bbfa65c2e525252b4775009e6e9563af83c968e0280bdd0605c16ac13f40758` |
| `tests/reference/dense/prompt_10.greedy.ids` | `9ad25840434dbcd978c2da76754a69c4570bf6c61e2ab068b29a6aa426b968a4` |
| `tests/reference/moe/prompt_10.greedy.ids` | `cc4217cf1d033e0cec4dee4600391fcbd5583163dbb07097b758a793ab79e7ec` |
| `tests/reference/prompt_11.txt` | `b212c4404b3f94468d28b54c31dc5bec0ba1d94e632817cdede427b0204bca90` |
| `tests/reference/prompt_11.input.ids` | `33a5b7a26bf718d68c03e72e8adc5ced52005bf3afac20c23357f90e57a3ed09` |
| `tests/reference/dense/prompt_11.greedy.ids` | `12e176511edd9c6a5a4c70a1f4f6ec4f676eea7cfd886f16108470f735ab3617` |
| `tests/reference/moe/prompt_11.greedy.ids` | `22417fdd561953ede2aa086923e92dc2dbf4d81afa20b3e8de178451541df92e` |
| `tests/reference/prompt_12.txt` | `484df221bede61a8c9aee3fbc6f25bd043f6014c05d49d483f20f2c249ef261c` |
| `tests/reference/prompt_12.input.ids` | `050f4721b5741981727b7a17389c8f11a8b459079dd155cc1efda9d333b9cc1a` |
| `tests/reference/dense/prompt_12.greedy.ids` | `4bd468e40995b799e79a697877532e8052ffa39adbb40b72793d43fe47476aba` |
| `tests/reference/moe/prompt_12.greedy.ids` | `b27293367144aef3aa1e02e8cf65f77ffa2580b5475e47d06f5e423062f05ccf` |
| `tests/reference/prompt_13.txt` | `a573db00032c72d4114fed2253e4bbfb7dd4e3c966b2297325cdaf636d2ce0ab` |
| `tests/reference/prompt_13.input.ids` | `76d782b8acfebd69a5ed961f6563290d95f048217304428db064cbad1fdcd307` |
| `tests/reference/dense/prompt_13.greedy.ids` | `f68185bf4922e0df7804ede145ddc3654a3e2ae8273063145147cc7f2f94ec10` |
| `tests/reference/moe/prompt_13.greedy.ids` | `564739ea9eb7283606f67eb6459a529d6646950eb1e2de95418d2b5da3b308e5` |
| `tests/reference/prompt_14.txt` | `10eebba0246c3beadd0358af6a63618c3d5eaa83c3caae4473d51d015921bc10` |
| `tests/reference/prompt_14.input.ids` | `0bc39bd322c532f9d0fb377e11035338fb4646e8a40b909c717f06c1636d33ac` |
| `tests/reference/dense/prompt_14.greedy.ids` | `3bfc884bf38e190c54abbd871e39301b84ab3eda1279ae5d397ac4d825c19967` |
| `tests/reference/moe/prompt_14.greedy.ids` | `44139d90eb6977b3456dcea03d1c0fbb94ab33394e7533324184297797c03f1f` |
| `tests/reference/prompt_15.txt` | `9ec36f8f4ccd0b9800fb34ab4c3808137c28cc4325b1de5c00cca9780a4112e0` |
| `tests/reference/prompt_15.input.ids` | `3d91c1091bce1b11ce65a302a9fa5674aa81afd4eee6ce678a61f5f7b0e64ea2` |
| `tests/reference/dense/prompt_15.greedy.ids` | `2e47f173dca5a61f31c566ff9456c3b4778bba84905436887565740faf339245` |
| `tests/reference/moe/prompt_15.greedy.ids` | `faea3695e0166b6784008cf548bc39781b5506e3b44285190c599206381c33fc` |
| `tests/reference/prompt_16.txt` | `beac8fe6cbadcc4397b8da06bf6991ac31b2821e0a8c4f65470606c51390b1a6` |
| `tests/reference/prompt_16.input.ids` | `0b732b0dd64b09e0c758296f5a2690c0de780081183da5c58ac8add6de1d488d` |
| `tests/reference/dense/prompt_16.greedy.ids` | `f55e0d29d8d92bc7fb0729acef405c0632450ca1756d8660d763de4193a9929d` |
| `tests/reference/moe/prompt_16.greedy.ids` | `71f027ffc8faec1ce7513ad0e89a0884955797368139abb5983c99a1d4d1e58e` |
| `tests/reference/prompt_17.txt` | `40164fb36abdc430ccf5e1ed8f16efc69b5fd7db29b61e900c44ab34aa8d820f` |
| `tests/reference/prompt_17.input.ids` | `bbed4cbf19b6db032614bcebf19d397e093d716bdd661a268e380e0c4a5f667b` |
| `tests/reference/dense/prompt_17.greedy.ids` | `f964b5e1902efacdbbf673ea853c9da4525296506bbaa2d9716237fc7948f029` |
| `tests/reference/moe/prompt_17.greedy.ids` | `64edbadd3cfb9561d731a4cacb564159a0deee07e3ecc8de4dd7b4b458a2e9f8` |
| `tests/reference/prompt_18.txt` | `16dc6aecfebb45e877ab089ecbb879419b97703939c97dd9a53683f8c3c28672` |
| `tests/reference/prompt_18.input.ids` | `6ee70ebc84d82014c112d7f3f90065eab2fe9cc3d4b100ea3787017038a53e8a` |
| `tests/reference/dense/prompt_18.greedy.ids` | `2e47f173dca5a61f31c566ff9456c3b4778bba84905436887565740faf339245` |
| `tests/reference/moe/prompt_18.greedy.ids` | `7792ebc9106c1569076c476fc120088d92771e2843081c3158760d7417294689` |
| `tests/reference/prompt_19.txt` | `a0e9b10451a3796199fe94f25ef8b4a1dfcc609fbd6e0d02e6b3a1820c8caa01` |
| `tests/reference/prompt_19.input.ids` | `5d21fd5062b23e61bb8bd350463568f9c2b3cedad21bf4a0e0f3f099620e7daa` |
| `tests/reference/dense/prompt_19.greedy.ids` | `a79771fe7c85776f401393364f43f9847338053af1a935e2c9d7dc6f98861d0f` |
| `tests/reference/moe/prompt_19.greedy.ids` | `0a7d6ef44683bdeaa90e6cbdadd431e5ea1be2d590459538fda095c28ad3b092` |

The 20 prompts range from 8 to 512 byte tokens. Prompt whitespace is
significant; readers strip CR/LF only, never trailing spaces.
