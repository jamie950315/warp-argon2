# warp-argon2

High-throughput warp-cooperative **RFC 9106 Argon2id** (`m=65536` KiB / 64 MiB, `t=1`, `p=1`, `outlen=32`) for **Apple Silicon Metal** and **NVIDIA CUDA**.

Rather than running one thread per Argon2 instance (which spills a 1 KiB block across local memory and stalls on uncoalesced 64 MiB indexing), `warp-argon2` assigns **one 32-lane warp / SIMD group per hash**, keeping the full 1,024-byte `G` state across registers (4 × `uint64_t` per lane) and a 1 KiB transpose stage in on-chip shared/threadgroup memory.

## Key Architectural Features

- **32-lane full-warp cooperative compression (`REG_IO` + `DIAG_OUT`)**
  - Every lane holds 4 contiguous `uint64_t` registers (`WarpBlockWords`) covering the full 128-qword Argon2 block, carrying the `prev` block in registers across entire segments instead of reloading it from DRAM.
  - 8 parallel 4-lane tiles execute column and row Blake2b/BlaMka `G` rounds in registers via warp/SIMD shuffles (`__shfl_sync` / `simd_shuffle`).
  - `DIAG_OUT` elides the 3 post-round inverse diagonal shuffles (`b`, `c`, `d`) by absorbing the diagonal lane rotation directly into the 1 KiB transpose scatter permutation.
- **Data-independent table & safe one-ahead prefetch (`DI` + `DD`)**
  - For `t=1, p=1`, slices 0–1 (32,768 blocks) are data-independent; `warp-argon2` precomputes the exact RFC 9106 reference table once at initialization and pipelines reference loads one iteration ahead (`ref <= curr - 2`).
  - Slices 2–3 (data-dependent) overlap address generation with diagonal-out phase-2 output and broadcast via lane 0.
- **Platform-specific memory layouts**
  - **Apple Silicon Metal (UMA)**: uses a **2 GiB-chunked `BLOCK_MAJOR` pool** (`32 hashes × 64 MiB` per `MTLBuffer`, up to 12 chunks) in `MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked`, turning concurrent store traffic into coalesced multi-hash writes without discrete PCIe copies.
  - **NVIDIA CUDA**: uses **`HASH_MAJOR`** (`pool + hash_idx * 65536 * 128`) with L1-bypass coherent global reference loads (`__ldcg`) and a split 3-kernel pipeline (`init` → `fill` → `finalize`) configured with `cudaFuncCachePreferL1` and `__launch_bounds__(32, 16)` (**56 registers/thread, 1024 B shared/CTA, 0 spills, 0 local bytes**).
- **Zero external dependencies**
  - Self-contained C++20 codebase bundling only the minimal `phc-winner-argon2` C reference implementation in [`third_party/argon2/`](third_party/argon2) for CPU verification and RFC 9106 `H0` compatibility.
  - Arbitrary `password`, `salt`, `secret`, and `associated data` lengths are supported on both GPUs by computing RFC 9106 `H0` on host (`hash_batch`) or accepting precomputed `H0` (`hash_batch_h0`), while sequential range workloads compute `H0` directly on-GPU (`hash_indexed_range`).

## Verified Benchmarks

Profile: RFC 9106 Argon2id (`v0x13`, `m=65536` KiB, `t=1`, `p=1`, `outlen=32`).

| Target | Architecture / Backend | Concurrency | Memory Allocation | Measured Throughput |
| :--- | :--- | :--- | :--- | :--- |
| **Apple M5 Pro** | Apple Silicon Metal (UMA) | `c=192` | 12,288 MiB (6 × 2 GiB chunks) | **1,812.41 H/s** (`40.7×` 1T CPU) |
| **NVIDIA GeForce RTX 4070 SUPER** | Ada Lovelace `sm_89` CUDA | `c=166` (autotuned) / `c=168` | 10,624 MiB / 10,752 MiB | **3,674.87 H/s** (`3,656.47 H/s` @ `c=168`) |
| **Host CPU (`libargon2`, 1 thread)** | Apple M5 Pro reference C | 1 thread | 64 MiB | **44.54 H/s** |

*(See [`docs/microarchitecture.md`](docs/microarchitecture.md) for PTXAS register/shared-memory metrics, CUDA stage breakdown, WDDM paging cliff behavior, and negative A/B findings.)*

## Building

### macOS (Apple Silicon Metal)

Requires Xcode Command Line Tools (`xcrun metal` / `metallib`) and CMake >= 3.22:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### Linux / Windows (NVIDIA CUDA)

Requires CUDA Toolkit (>= 11.8) and CMake >= 3.22:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

To target a specific SM architecture (for example `89` for RTX 40-series Ada Lovelace):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
```

## CLI Usage

```bash
# Run self-test verifying RFC 9106 KATs, H0 pre-hash, and GPU == libargon2 equality
./build/warp-argon2 --self-test

# Hash a password & salt through the active GPU backend (auto-detects Metal or CUDA)
./build/warp-argon2 --password "password" --salt "somesalt"

# Benchmark GPU throughput
./build/warp-argon2 --bench --seconds 5

# Benchmark single-thread / multi-thread CPU baseline
./build/warp-argon2 --backend cpu --bench --seconds 3 --cpu-threads 1

# CUDA-only concurrency autotuner & stage-timing profiler
./build/warp-argon2 --autotune --seconds 2
./build/warp-argon2 --stage-timing --seconds 3
```

## C++ API

```cpp
#include "warp_argon2/argon2id.hpp"
#include "warp_argon2/metal_backend.hpp"  // or cuda_backend.hpp

warp_argon2::MetalBackend gpu;
std::string err;
if (!gpu.init(0, err)) { /* handle error */ }

const std::string pwd = "correct horse battery staple";
const std::string salt = "warp-argon2-salt";
std::vector<warp_argon2::InputView> batch{
    warp_argon2::InputView{
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(pwd.data()), pwd.size()),
        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(salt.data()), salt.size()),
    },
};

std::vector<warp_argon2::Digest> out;
if (gpu.hash_batch(batch, out, err)) {
    std::cout << warp_argon2::to_hex(out[0]) << "\n";
}
```

## License

MIT License ([`LICENSE`](LICENSE)). Bundled minimal reference files under [`third_party/argon2/`](third_party/argon2) retain their upstream CC0 1.0 Universal / Apache 2.0 dual license.
