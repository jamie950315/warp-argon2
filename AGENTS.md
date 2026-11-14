# AGENTS.md

## Overview

`warp-argon2` is a standalone, dependency-free C++20 implementation of **RFC 9106 Argon2id** (`m=65536` KiB, `t=1`, `p=1`, `outlen=32`) accelerated with 32-lane warp/SIMD-group cooperative kernels for **Apple Silicon Metal** and **NVIDIA CUDA**.

## Architecture

- **`include/warp_argon2/`**: Public C++20 API (`argon2id.hpp`, `metal_backend.hpp`, `cuda_backend.hpp`).
- **`src/cpu_ref.cpp`**: RFC 9106 `compute_h0`, `build_di_ref_table`, and `phc-winner-argon2` CPU reference wrapper.
- **`src/metal/`**: Apple Silicon Metal compute shaders (`argon2id.metal`) and Objective-C++ runtime (`metal_backend.mm`) using 2 GiB-chunked `BLOCK_MAJOR` UMA buffers, `REG_IO`, `DIAG_OUT`, and precomputed 32K DI index table. `warp_argon2.metallib` is built exclusively in `${CMAKE_BINARY_DIR}` (never inside the source tree).
- **`src/cuda/`**: NVIDIA CUDA kernels (`blake2b.cuh`, `argon2_fill_block.cuh`, `argon2_device.cuh`, `cuda_backend.cu`) using split `init`/`fill`/`finalize`, `HASH_MAJOR` layout, `__ldcg` ref loads, `REG_IO`, `DIAG_OUT`, and pinned multi-stream overlap.
- **`third_party/argon2/`**: Minimal vendored C sources from `phc-winner-argon2` (zero external runtime dependencies).
- **`tests/`**: `test_correctness.cpp` (RFC 9106 KATs, `H0` checks, DI gap-2 bound proof, `DIAG_OUT` transpose proof, Metal/CUDA equality against `libargon2`) and `test_cuda_fill_block.cu` (CUDA full-warp unit test).

## Commands

### macOS (Metal)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/warp-argon2 --self-test
./build/warp-argon2 --bench --seconds 5
```

### Linux / Windows (CUDA)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
ctest --test-dir build --output-on-failure
./build/warp-argon2 --self-test
./build/warp-argon2 --bench --seconds 5
```
