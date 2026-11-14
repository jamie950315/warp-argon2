# Microarchitecture & Optimization Notes

`warp-argon2` optimizes the fixed RFC 9106 Argon2id profile (`m=65536` KiB = 64 MiB, `t=1`, `p=1`, `outlen=32`) using a 32-lane warp/SIMD-group cooperative execution model across **Apple Silicon Metal** and **NVIDIA CUDA**.

## 1. Cooperative 32-Lane Register Model (`REG_IO` + `DIAG_OUT`)

One 1,024-byte Argon2 block (`128 × uint64_t`) maps across 32 threads (`tid = 0..31`) in natural column-round ownership:

```text
tile = tid >> 2   (0..7)
lane = tid & 3    (0..3)
registers (a, b, c, d) hold words:
  16*tile + lane + {0, 4, 8, 12}
```

Key design wins over classic 1-thread-per-hash and 8-lane-per-warp kernels:

1. **Register-resident `prev` forwarding (`REG_IO`)**: Within each segment, `prev` (`block[i-1]`) never leaves the 4 `uint64_t` registers per lane. Every block step performs only **one** 1 KiB global read (`ref`) and **one** 1 KiB global write (`next`).
2. **Full 32-lane BlaMka utilization**: All 32 threads remain active throughout both column and row compression rounds (8 parallel 4-lane tiles), removing the 75% idle lane penalty of naive 8-lane shared-memory Argon2 kernels.
3. **`DIAG_OUT` shuffle elision**: Each 4-lane tile round requires two BlaMka half-rounds separated by a diagonal word permutation:
   - `b` rotates by `+1 mod 4`
   - `c` rotates by `+2 mod 4`
   - `d` rotates by `+3 mod 4`
   Instead of running three inverse `__shfl_sync` / `simd_shuffle` instructions after every half-round to restore natural ownership, `DIAG_OUT` omits all three inverse shuffles and folds the diagonal permutation `(lane + k) & 3` into the 1 KiB `stage[4][32]` transpose scatter index.
   - As a structural bonus, after phase-2 BlaMka in DD slices, lane 0's `(t0 ^ a)` already holds natural `word0` (`pseudo_rand`) without finishing the second transpose first.

## 2. RFC 9106 `H0` Boundary Decoupling

Supporting arbitrary passwords, salts, secrets, and associated data lengths on GPU without dynamic heap allocation or global byte-streaming loops:

- **Arbitrary inputs (`hash_batch` / `hash_batch_h0`)**: Host computes the 64-byte RFC 9106 pre-hashing seed `H0` via Blake2b-512 (`compute_h0`). Each GPU init warp expands `H0 || LE32(0) || LE32(0)` and `H0 || LE32(1) || LE32(0)` via `blake2b_long(..., 1024)` into `B[0]` and `B[1]`.
- **Sequential range workloads (`hash_indexed_range`)**: Each threadgroup/warp computes `H0` directly on-device for `pwd = LE64(start_index + idx)` and `salt = salt16` using one Blake2b-512 compression (`40 + 8 + 16 = 64` bytes <= 128 B).

## 3. Data-Independent Table & Prefetch Safety Bound

For `t=1, p=1`, slices 0 and 1 (`blocks 2..32767`) depend solely on counter blocks `(pass=0, lane=0, slice, m=65536, t=1, type=2)`. Both backends precompute this 32,768-entry `uint32_t` index table once at initialization (`128 KiB`).

### Prefetch Distance Proof

Across all 32,768 DI indices (`tests/test_correctness.cpp` enforces this exhaustively):

- Minimum gap between `curr` and `ref(curr)` is **2** (`curr - ref >= 2`, with gap = 2 occurring explicitly).
- Therefore:
  - **One-ahead prefetch (`ref(j+1)` loaded while computing `j`)**: unconditionally safe, since `ref(j+1) <= (j+1) - 2 = j - 1 < j`.
  - **Two-ahead prefetch (`ref(j+2)` loaded before `j` completes)**: invalid, because `ref(j+2)` can equal `j` before block `j` is written.

## 4. Metal vs CUDA Hardware Adaptation

| Dimension | Apple Silicon Metal | NVIDIA CUDA |
| :--- | :--- | :--- |
| Memory pool layout | **2 GiB-chunked `BLOCK_MAJOR`** (`kChunkHashes = 32`, `pool[(b * 32 + local) * 128]`) | **`HASH_MAJOR`** (`pool[(size_t)hash_idx * 65536 + b]`) |
| Rationale | Respects per-`MTLBuffer` max length while turning concurrent stores into contiguous cross-hash UMA bursts | Contiguous per-hash addressing minimizes address arithmetic (`HASH_MAJOR` beat `BLOCK_MAJOR` on discrete GDDR6X) |
| Coherent ref loads | Normal device loads (`MTLResourceHazardTrackingModeUntracked` across synced segments) | `__ldcg` (`ld.global.cg`) bypasses L1 for mutable 64 MiB DRAM workspace while preserving L2 coherency |
| Kernel launch | Single dispatch per chunk (`argon2id_batch_h0` / `argon2id_indexed_range`) | Split `init` -> `fill` -> `finalize` (`__launch_bounds__(32, 16)`, L1 preference, 0 B shared carveout) |

## 5. Measured Negative Experiments (Kept Out of Production)

Several alternatives were benchmarked and intentionally rejected on hardware:

1. **Pure 4-color register-shuffle transpose (0 B shared memory)**: Correct bijection verified by device oracle, yet ~3–6% slower than the `1 KiB` `stage[4][32]` scatter/gather on both RTX 4070 SUPER and Apple M5 Pro due to instruction/register shuffle pressure.
2. **CUDA `cp.async` global-to-shared ref staging (`sm_80+`)**: Increased shared footprint from 1 KiB to 3 KiB, cutting warp occupancy and regressing net H/s by ~0.8%.
3. **Inline PTX `st.global.*` stores on CUDA 13.3 + `sm_89`**: Triggered a sticky driver `cudaErrorInvalidAddressSpace` on subsequent `cudaMalloc` calls; compiler-default stores (`*p = v`) remain clean and optimal.
4. **Two-ahead DI prefetch window (`>= 3`)**: Provably invalid on gap-2 DI blocks.

## 6. Measured Hardware Telemetry

Verified after final edits on both GPUs (`m=65536` KiB, `t=1`, `p=1`, `outlen=32`):

### Apple M5 Pro (Apple Silicon Metal, Unified Memory)
- **Self-test**: `16/16` arbitrary `hash_batch` vectors (`password`/`salt`/`secret`/`AD`) and `24/24` multi-wave `hash_indexed_range` outputs byte-identical to `libargon2`; `warp_argon2_correctness` passed in `5.97 s`.
- **Throughput (`c=192`, 12,288 MiB across six 2 GiB chunks)**: **1,812.41 H/s** (`9,216` hashes in `5.08 s`).
- **Host CPU reference (`libargon2`, 1 thread)**: **44.54 H/s** (`40.7×` speedup on Metal GPU).

### NVIDIA GeForce RTX 4070 SUPER (`sm_89`, CUDA 13.3 on WSL2)
- **PTXAS compilation (`argon2_fill_kernel`)**: **56 registers/thread**, **1,024 B shared memory/CTA**, **0 bytes stack frame**, **0 spill stores**, **0 spill loads**.
- **Self-test & CTest**: `2/2` tests passed (`warp_argon2_correctness` + `cuda_fill_block`); `16/16` batch vectors and `24/24` multi-wave range vectors byte-identical to `libargon2`.
- **Throughput & WDDM cliff (`--autotune`)**:
  - Peak autotuned throughput at `c=166` (`10,624 MiB`): **3,674.87 H/s** (just below WDDM non-dedicated VRAM paging).
  - Sustained 5s run at `c=168` (`10,752 MiB`): **3,656.47 H/s** (`18,312` hashes in `5.01 s`); above ~10.7 GiB under desktop WDDM load, paging drops throughput (`c=172` -> `2,648.45 H/s`, `c=176` -> `2,488.06 H/s`).
- **CUDA Stage Breakdown (`--stage-timing --concurrency 168`, 66 waves, 3,663.30 H/s)**:
  - `H2D`: `0.02 ms`
  - `Init`: `0.31 ms`
  - `Fill`: `45.41 ms` (`99.2%` of wave time)
  - `Final`: `0.05 ms`
  - `D2H`: `0.00 ms`
  - `Total`: `45.79 ms` / wave
