#include "warp_argon2/cuda_backend.hpp"
#include "argon2_device.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <vector>

namespace warp_argon2 {

namespace {

constexpr int kThreadsPerHash = 32;
constexpr int kMaxStreams = 4;
constexpr int kHardMaxConcurrency = 512;
constexpr std::size_t kWorkspaceBytes =
    static_cast<std::size_t>(kMemBlocks) * sizeof(cuda_detail::Block);

#ifndef WARP_A2_MIN_BLOCKS_PER_SM
#define WARP_A2_MIN_BLOCKS_PER_SM 16
#endif
#if WARP_A2_MIN_BLOCKS_PER_SM > 0
#define WARP_A2_LAUNCH_BOUNDS __launch_bounds__(32, WARP_A2_MIN_BLOCKS_PER_SM)
#else
#define WARP_A2_LAUNCH_BOUNDS __launch_bounds__(32)
#endif

__global__ void __launch_bounds__(32)
argon2_init_h0_kernel(const uint8_t* __restrict__ d_h0,
                      cuda_detail::Block* __restrict__ memory_pool,
                      int count) {
    const int idx = blockIdx.x;
    if (idx >= count) return;
    const int tid = threadIdx.x;
    __shared__ cuda_detail::InitShared sh;
    __shared__ uint8_t h0_local[64];
    for (int i = tid; i < 64; i += kThreadsPerHash) {
        h0_local[i] = d_h0[static_cast<std::size_t>(idx) * 64 + static_cast<std::size_t>(i)];
    }
    __syncwarp();
    cuda_detail::argon2id_init_from_h0(h0_local, memory_pool, idx, &sh, tid);
}

__global__ void __launch_bounds__(32)
argon2_init_range_kernel(const uint8_t* __restrict__ d_salt16,
                         uint64_t start_index,
                         cuda_detail::Block* __restrict__ memory_pool,
                         int count) {
    const int idx = blockIdx.x;
    if (idx >= count) return;
    const int tid = threadIdx.x;
    __shared__ cuda_detail::InitShared sh;
    __shared__ uint8_t salt_local[16];
    if (tid < 16) {
        salt_local[tid] = d_salt16[tid];
    }
    __syncwarp();
    cuda_detail::argon2id_init_indexed(salt_local, start_index + static_cast<uint64_t>(idx),
                                       memory_pool, idx, &sh, tid);
}

__global__ void WARP_A2_LAUNCH_BOUNDS
argon2_fill_kernel(cuda_detail::Block* __restrict__ memory_pool,
                   const uint32_t* __restrict__ di_ref_table,
                   int count) {
    const int idx = blockIdx.x;
    if (idx >= count) return;
    __shared__ cuda_detail::FillShared sh;
    cuda_detail::argon2id_fill_matrix_mainnet(memory_pool, idx, &sh, threadIdx.x, di_ref_table);
}

__global__ void __launch_bounds__(32)
argon2_finalize_kernel(uint8_t* __restrict__ digests,
                       cuda_detail::Block* __restrict__ memory_pool,
                       int count) {
    const int idx = blockIdx.x;
    if (idx >= count) return;
    __shared__ cuda_detail::FinalShared sh;
    cuda_detail::argon2id_finalize(digests + static_cast<std::size_t>(idx) * 32,
                                   memory_pool, idx, &sh, threadIdx.x);
}

struct DeviceBuffers {
    cuda_detail::Block* memory = nullptr;
    uint32_t* d_di_ref_table = nullptr;
    uint8_t* d_salt16 = nullptr;

    // Per-stream staging buffers so H2D / D2H on stream s can overlap cleanly.
    uint8_t* d_h0[kMaxStreams]{};
    uint8_t* d_digests[kMaxStreams]{};
    uint8_t* h_h0[kMaxStreams]{};
    uint8_t* h_digests[kMaxStreams]{};
    cudaStream_t streams[kMaxStreams]{};
    cudaEvent_t pool_done[kMaxStreams]{};

    cudaEvent_t te_h2d_start{};
    cudaEvent_t te_h2d_end{};
    cudaEvent_t te_init_end{};
    cudaEvent_t te_fill_end{};
    cudaEvent_t te_final_end{};
    cudaEvent_t te_d2h_end{};
    bool timing_events_ok = false;

    int n_streams = 2;
    int concurrency = 0;
};

void destroy_timing_events(DeviceBuffers* b) {
    auto kill = [](cudaEvent_t& e) {
        if (e) {
            cudaEventDestroy(e);
            e = nullptr;
        }
    };
    kill(b->te_h2d_start);
    kill(b->te_h2d_end);
    kill(b->te_init_end);
    kill(b->te_fill_end);
    kill(b->te_final_end);
    kill(b->te_d2h_end);
    b->timing_events_ok = false;
}

bool create_timing_events(DeviceBuffers* b) {
    destroy_timing_events(b);
    auto mk = [](cudaEvent_t& e) -> bool {
        return cudaEventCreate(&e) == cudaSuccess;
    };
    if (!mk(b->te_h2d_start) || !mk(b->te_h2d_end) || !mk(b->te_init_end) ||
        !mk(b->te_fill_end) || !mk(b->te_final_end) || !mk(b->te_d2h_end)) {
        destroy_timing_events(b);
        return false;
    }
    b->timing_events_ok = true;
    return true;
}

void configure_fill_kernel_cache() {
    const void* fn = reinterpret_cast<const void*>(&argon2_fill_kernel);
    (void)cudaFuncSetCacheConfig(fn, cudaFuncCachePreferL1);
    (void)cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    (void)cudaGetLastError();
}

}  // namespace

CudaBackend::~CudaBackend() {
    free_buffers();
    delete static_cast<DeviceBuffers*>(impl_);
    impl_ = nullptr;
}

std::vector<CudaDeviceInfo> CudaBackend::enumerate_devices() {
    std::vector<CudaDeviceInfo> out;
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n <= 0) {
        (void)cudaGetLastError();
        return out;
    }
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) continue;
        CudaDeviceInfo d;
        d.index = i;
        d.name = prop.name;
        d.compute_major = prop.major;
        d.compute_minor = prop.minor;
        d.total_mem_bytes = prop.totalGlobalMem;
        d.sm89_native = (prop.major == 8 && prop.minor == 9);
        out.push_back(std::move(d));
    }
    return out;
}

int CudaBackend::concurrency() const {
    if (!impl_) return 0;
    return static_cast<const DeviceBuffers*>(impl_)->concurrency;
}

void CudaBackend::free_buffers() {
    ready_ = false;
    if (!impl_) return;
    auto* b = static_cast<DeviceBuffers*>(impl_);
    destroy_timing_events(b);
    for (int i = 0; i < kMaxStreams; ++i) {
        if (b->pool_done[i]) {
            cudaEventDestroy(b->pool_done[i]);
            b->pool_done[i] = nullptr;
        }
        if (b->streams[i]) {
            cudaStreamDestroy(b->streams[i]);
            b->streams[i] = nullptr;
        }
        if (b->d_h0[i]) {
            cudaFree(b->d_h0[i]);
            b->d_h0[i] = nullptr;
        }
        if (b->d_digests[i]) {
            cudaFree(b->d_digests[i]);
            b->d_digests[i] = nullptr;
        }
        if (b->h_h0[i]) {
            cudaFreeHost(b->h_h0[i]);
            b->h_h0[i] = nullptr;
        }
        if (b->h_digests[i]) {
            cudaFreeHost(b->h_digests[i]);
            b->h_digests[i] = nullptr;
        }
    }
    if (b->memory) {
        cudaFree(b->memory);
        b->memory = nullptr;
    }
    if (b->d_di_ref_table) {
        cudaFree(b->d_di_ref_table);
        b->d_di_ref_table = nullptr;
    }
    if (b->d_salt16) {
        cudaFree(b->d_salt16);
        b->d_salt16 = nullptr;
    }
    b->concurrency = 0;
}

bool CudaBackend::ensure_di_table(std::string& err) {
    if (!impl_) {
        err = "DeviceBuffers not allocated";
        return false;
    }
    auto* b = static_cast<DeviceBuffers*>(impl_);
    if (b->d_di_ref_table) return true;

    std::vector<uint32_t> host_table(kDiTableBlocks);
    build_di_ref_table(host_table.data());

    const std::size_t bytes = kDiTableBlocks * sizeof(uint32_t);
    cudaError_t e = cudaMalloc(&b->d_di_ref_table, bytes);
    if (e != cudaSuccess) {
        err = std::string("cudaMalloc di_ref_table: ") + cudaGetErrorString(e);
        return false;
    }
    e = cudaMemcpy(b->d_di_ref_table, host_table.data(), bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        err = std::string("cudaMemcpy di_ref_table: ") + cudaGetErrorString(e);
        cudaFree(b->d_di_ref_table);
        b->d_di_ref_table = nullptr;
        return false;
    }
    return true;
}

bool CudaBackend::allocate_buffers(int concurrency, int n_streams, std::string& err) {
    free_buffers();
    auto* b = impl_ ? static_cast<DeviceBuffers*>(impl_) : new DeviceBuffers();
    if (!impl_) impl_ = b;

    n_streams = std::clamp(n_streams, 1, kMaxStreams);
    concurrency = std::max(1, concurrency);
    b->concurrency = concurrency;
    b->n_streams = n_streams;
    n_streams_ = n_streams;

    (void)cudaGetLastError();
    const std::size_t pool_bytes = static_cast<std::size_t>(concurrency) * kWorkspaceBytes;
    cudaError_t e = cudaMalloc(&b->memory, pool_bytes);
    if (e != cudaSuccess) {
        err = std::string("cudaMalloc pool: ") + cudaGetErrorString(e);
        free_buffers();
        return false;
    }

    e = cudaMalloc(&b->d_salt16, 16);
    if (e != cudaSuccess) {
        err = std::string("cudaMalloc salt16: ") + cudaGetErrorString(e);
        free_buffers();
        return false;
    }

    for (int i = 0; i < n_streams; ++i) {
        if (cudaMalloc(&b->d_h0[i], static_cast<std::size_t>(concurrency) * kH0Bytes) != cudaSuccess ||
            cudaMalloc(&b->d_digests[i], static_cast<std::size_t>(concurrency) * kTagLen) != cudaSuccess ||
            cudaHostAlloc(&b->h_h0[i], static_cast<std::size_t>(concurrency) * kH0Bytes,
                          cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc(&b->h_digests[i], static_cast<std::size_t>(concurrency) * kTagLen,
                          cudaHostAllocDefault) != cudaSuccess ||
            cudaStreamCreate(&b->streams[i]) != cudaSuccess ||
            cudaEventCreateWithFlags(&b->pool_done[i], cudaEventDisableTiming) != cudaSuccess) {
            err = std::string("CUDA stream/staging alloc failed: ") +
                  cudaGetErrorString(cudaGetLastError());
            free_buffers();
            return false;
        }
    }

    if (!create_timing_events(b)) {
        err = "cudaEventCreate failed for stage timing";
        free_buffers();
        return false;
    }

    if (!ensure_di_table(err)) {
        free_buffers();
        return false;
    }

    return true;
}

int CudaBackend::probe_max_concurrency(int device_index, std::string& err) {
    cudaDeviceProp prop{};
    if (cudaSetDevice(device_index) != cudaSuccess ||
        cudaGetDeviceProperties(&prop, device_index) != cudaSuccess) {
        err = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cudaGetLastError());
        return 0;
    }

    int lo = 1;
    int hi = static_cast<int>(prop.totalGlobalMem / kWorkspaceBytes);
    hi = std::clamp(hi, 1, kHardMaxConcurrency);

    int best = 0;
    while (lo <= hi) {
        const int mid = lo + (hi - lo) / 2;
        void* tmp = nullptr;
        const std::size_t bytes =
            static_cast<std::size_t>(mid) * kWorkspaceBytes +
            static_cast<std::size_t>(mid) * (kH0Bytes + kTagLen) * 2 +
            (32u << 20);
        if (cudaMalloc(&tmp, bytes) == cudaSuccess) {
            cudaFree(tmp);
            best = mid;
            lo = mid + 1;
        } else {
            (void)cudaGetLastError();
            hi = mid - 1;
        }
    }
    return best;
}

bool CudaBackend::init(int device_index, int concurrency, int n_streams, std::string& err) {
    free_buffers();

    const auto devs = enumerate_devices();
    if (device_index < 0 || static_cast<std::size_t>(device_index) >= devs.size()) {
        err = "invalid CUDA device index";
        return false;
    }

    cudaError_t e = cudaSetDevice(device_index);
    if (e != cudaSuccess) {
        err = std::string("cudaSetDevice failed: ") + cudaGetErrorString(e);
        return false;
    }

    device_index_ = device_index;
    device_name_ = devs[static_cast<std::size_t>(device_index)].name;
    configure_fill_kernel_cache();

    if (concurrency <= 0) {
        return init_auto(device_index, err);
    }

    if (!allocate_buffers(concurrency, n_streams, err)) {
        return false;
    }
    ready_ = true;
    return true;
}

bool CudaBackend::init_auto(int device_index, std::string& err) {
    free_buffers();

    const auto devs = enumerate_devices();
    if (device_index < 0 || static_cast<std::size_t>(device_index) >= devs.size()) {
        err = "invalid CUDA device index";
        return false;
    }
    if (cudaSetDevice(device_index) != cudaSuccess) {
        err = std::string("cudaSetDevice failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }

    device_index_ = device_index;
    device_name_ = devs[static_cast<std::size_t>(device_index)].name;
    configure_fill_kernel_cache();

    const int max_c = probe_max_concurrency(device_index, err);
    if (max_c < 1) {
        if (err.empty()) err = "insufficient GPU VRAM for single 64 MiB Argon2 workspace";
        return false;
    }

    int try_c = std::max(1, (max_c * 85) / 100);
    while (try_c >= 1) {
        if (allocate_buffers(try_c, 2, err)) {
            ready_ = true;
            return true;
        }
        try_c = (try_c > 8) ? (try_c - 8) : (try_c - 1);
    }
    return false;
}

bool CudaBackend::hash_batch_h0(std::span<const H0Block> h0_batch,
                                std::vector<Digest>& out,
                                std::string& err) {
    if (h0_batch.empty()) {
        out.clear();
        return true;
    }
    if (!ready_ || !impl_) {
        err = "CUDA backend not initialized";
        return false;
    }

    auto* b = static_cast<DeviceBuffers*>(impl_);
    out.resize(h0_batch.size());

    const int max_wave = std::max(1, b->concurrency);
    const int ns = std::clamp(b->n_streams, 1, kMaxStreams);

    struct InFlight {
        int s = 0;
        std::size_t offset = 0;
        int count = 0;
    };
    std::vector<InFlight> ring(static_cast<std::size_t>(ns));
    std::vector<bool> active(static_cast<std::size_t>(ns), false);

    std::size_t offset = 0;
    std::size_t remaining = h0_batch.size();
    int wave_idx = 0;

    auto finish_slot = [&](int s) -> bool {
        if (!active[static_cast<std::size_t>(s)]) return true;
        cudaError_t e = cudaStreamSynchronize(b->streams[s]);
        if (e != cudaSuccess) {
            err = std::string("CUDA stream sync failed: ") + cudaGetErrorString(e);
            return false;
        }
        const auto& inf = ring[static_cast<std::size_t>(s)];
        for (int i = 0; i < inf.count; ++i) {
            std::memcpy(out[inf.offset + static_cast<std::size_t>(i)].data(),
                        b->h_digests[s] + static_cast<std::size_t>(i) * kTagLen,
                        kTagLen);
        }
        active[static_cast<std::size_t>(s)] = false;
        return true;
    };

    while (remaining > 0) {
        const int s = wave_idx % ns;
        if (!finish_slot(s)) return false;

        const int chunk = static_cast<int>(std::min<std::size_t>(remaining, static_cast<std::size_t>(max_wave)));
        cudaStream_t st = b->streams[s];

        for (int i = 0; i < chunk; ++i) {
            std::memcpy(b->h_h0[s] + static_cast<std::size_t>(i) * kH0Bytes,
                        h0_batch[offset + static_cast<std::size_t>(i)].data(),
                        kH0Bytes);
        }

        if (cudaMemcpyAsync(b->d_h0[s], b->h_h0[s], static_cast<std::size_t>(chunk) * kH0Bytes,
                            cudaMemcpyHostToDevice, st) != cudaSuccess) {
            err = std::string("CUDA H2D failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }

        if (wave_idx > 0 && ns > 1) {
            const int prev_s = (wave_idx - 1) % ns;
            if (cudaStreamWaitEvent(st, b->pool_done[prev_s], 0) != cudaSuccess) {
                err = "cudaStreamWaitEvent failed";
                return false;
            }
        }

        argon2_init_h0_kernel<<<chunk, kThreadsPerHash, 0, st>>>(b->d_h0[s], b->memory, chunk);
        argon2_fill_kernel<<<chunk, kThreadsPerHash, 0, st>>>(b->memory, b->d_di_ref_table, chunk);
        argon2_finalize_kernel<<<chunk, kThreadsPerHash, 0, st>>>(b->d_digests[s], b->memory, chunk);
        if (cudaEventRecord(b->pool_done[s], st) != cudaSuccess) {
            err = "cudaEventRecord failed";
            return false;
        }

        if (cudaMemcpyAsync(b->h_digests[s], b->d_digests[s],
                            static_cast<std::size_t>(chunk) * kTagLen,
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            err = std::string("CUDA D2H failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }

        ring[static_cast<std::size_t>(s)] = {s, offset, chunk};
        active[static_cast<std::size_t>(s)] = true;
        offset += static_cast<std::size_t>(chunk);
        remaining -= static_cast<std::size_t>(chunk);
        ++wave_idx;
    }

    for (int s = 0; s < ns; ++s) {
        if (!finish_slot(s)) return false;
    }
    return true;
}

bool CudaBackend::hash_batch(std::span<const InputView> inputs,
                             std::vector<Digest>& out,
                             std::string& err) {
    if (inputs.empty()) {
        out.clear();
        return true;
    }
    std::vector<H0Block> h0s(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        h0s[i] = compute_h0(inputs[i]);
    }
    return hash_batch_h0(h0s, out, err);
}

bool CudaBackend::hash_indexed_range(std::span<const uint8_t, 16> salt16,
                                     uint64_t start_index,
                                     int count,
                                     std::vector<Digest>& out,
                                     std::string& err) {
    if (count < 1) {
        out.clear();
        return true;
    }
    if (!ready_ || !impl_) {
        err = "CUDA backend not initialized";
        return false;
    }

    auto* b = static_cast<DeviceBuffers*>(impl_);
    if (cudaMemcpy(b->d_salt16, salt16.data(), 16, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = std::string("CUDA salt16 H2D failed: ") + cudaGetErrorString(cudaGetLastError());
        return false;
    }

    out.resize(static_cast<std::size_t>(count));
    const int max_wave = std::max(1, b->concurrency);
    const int ns = std::clamp(b->n_streams, 1, kMaxStreams);

    struct InFlight {
        int offset = 0;
        int count = 0;
    };
    std::vector<InFlight> ring(static_cast<std::size_t>(ns));
    std::vector<bool> active(static_cast<std::size_t>(ns), false);

    int done = 0;
    int wave_idx = 0;

    auto finish_slot = [&](int s) -> bool {
        if (!active[static_cast<std::size_t>(s)]) return true;
        cudaError_t e = cudaStreamSynchronize(b->streams[s]);
        if (e != cudaSuccess) {
            err = std::string("CUDA range stream sync failed: ") + cudaGetErrorString(e);
            return false;
        }
        const auto& inf = ring[static_cast<std::size_t>(s)];
        std::memcpy(out.data() + inf.offset, b->h_digests[s],
                    static_cast<std::size_t>(inf.count) * kTagLen);
        active[static_cast<std::size_t>(s)] = false;
        return true;
    };

    while (done < count) {
        const int s = wave_idx % ns;
        if (!finish_slot(s)) return false;

        const int chunk = std::min(count - done, max_wave);
        cudaStream_t st = b->streams[s];

        if (wave_idx > 0 && ns > 1) {
            const int prev_s = (wave_idx - 1) % ns;
            if (cudaStreamWaitEvent(st, b->pool_done[prev_s], 0) != cudaSuccess) {
                err = "cudaStreamWaitEvent failed";
                return false;
            }
        }

        argon2_init_range_kernel<<<chunk, kThreadsPerHash, 0, st>>>(
            b->d_salt16, start_index + static_cast<uint64_t>(done), b->memory, chunk);
        argon2_fill_kernel<<<chunk, kThreadsPerHash, 0, st>>>(
            b->memory, b->d_di_ref_table, chunk);
        argon2_finalize_kernel<<<chunk, kThreadsPerHash, 0, st>>>(
            b->d_digests[s], b->memory, chunk);
        if (cudaEventRecord(b->pool_done[s], st) != cudaSuccess) {
            err = "cudaEventRecord failed";
            return false;
        }
        if (cudaMemcpyAsync(b->h_digests[s], b->d_digests[s],
                            static_cast<std::size_t>(chunk) * kTagLen,
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            err = std::string("CUDA D2H failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }

        ring[static_cast<std::size_t>(s)] = {done, chunk};
        active[static_cast<std::size_t>(s)] = true;
        done += chunk;
        ++wave_idx;
    }

    for (int s = 0; s < ns; ++s) {
        if (!finish_slot(s)) return false;
    }
    return true;
}

CudaBenchResult CudaBackend::bench_throughput(double seconds) {
    CudaBenchResult res{};
    if (!ready_ || !impl_) {
        res.error = "CUDA backend not initialized";
        return res;
    }

    auto* b = static_cast<DeviceBuffers*>(impl_);
    const int wave = std::max(1, b->concurrency);

    std::array<uint8_t, 16> salt{};
    for (uint8_t i = 0; i < 16; ++i) salt[i] = static_cast<uint8_t>(0x21u + i);
    if (cudaMemcpy(b->d_salt16, salt.data(), 16, cudaMemcpyHostToDevice) != cudaSuccess) {
        res.error = std::string("CUDA salt16 H2D failed: ") + cudaGetErrorString(cudaGetLastError());
        return res;
    }

    auto run_wave = [&](uint64_t start_idx, cudaStream_t st) -> bool {
        argon2_init_range_kernel<<<wave, kThreadsPerHash, 0, st>>>(
            b->d_salt16, start_idx, b->memory, wave);
        argon2_fill_kernel<<<wave, kThreadsPerHash, 0, st>>>(
            b->memory, b->d_di_ref_table, wave);
        argon2_finalize_kernel<<<wave, kThreadsPerHash, 0, st>>>(
            b->d_digests[0], b->memory, wave);
        if (cudaMemcpyAsync(b->h_digests[0], b->d_digests[0],
                            static_cast<std::size_t>(wave) * kTagLen,
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) {
            res.error = std::string("CUDA D2H failed: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        const cudaError_t e = cudaStreamSynchronize(st);
        if (e != cudaSuccess) {
            res.error = std::string("CUDA bench sync failed: ") + cudaGetErrorString(e);
            return false;
        }
        return true;
    };

    // Warmup one full wave.
    if (!run_wave(0, b->streams[0])) return res;

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t idx = static_cast<uint64_t>(wave);
    uint64_t total = 0;

    while (true) {
        if (!run_wave(idx, b->streams[0])) return res;
        idx += static_cast<uint64_t>(wave);
        total += static_cast<uint64_t>(wave);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= seconds) {
            res.ok = true;
            res.elapsed_sec = elapsed;
            res.hashes = total;
            res.hashes_per_sec = elapsed > 0.0 ? static_cast<double>(total) / elapsed : 0.0;
            return res;
        }
    }
}

StageTimingReport CudaBackend::stage_timing_run(double seconds) {
    StageTimingReport rpt{};
    if (!ready_ || !impl_) {
        rpt.error = "CUDA backend not initialized";
        return rpt;
    }

    auto* b = static_cast<DeviceBuffers*>(impl_);
    if (!b->timing_events_ok) {
        rpt.error = "CUDA timing events unavailable";
        return rpt;
    }

    const int wave = std::max(1, b->concurrency);
    cudaStream_t st = b->streams[0];

    std::array<uint8_t, 16> salt{};
    for (uint8_t i = 0; i < 16; ++i) salt[i] = static_cast<uint8_t>(0x55u + i);
    std::vector<H0Block> h0_vec(static_cast<std::size_t>(wave));
    for (int i = 0; i < wave; ++i) {
        uint8_t pwd[8];
        for (int k = 0; k < 8; ++k) pwd[k] = static_cast<uint8_t>((i >> (8 * k)) & 0xffu);
        h0_vec[static_cast<std::size_t>(i)] = compute_h0(InputView{pwd, salt});
        std::memcpy(b->h_h0[0] + static_cast<std::size_t>(i) * kH0Bytes,
                    h0_vec[static_cast<std::size_t>(i)].data(), kH0Bytes);
    }

    // Warmup once cleanly.
    argon2_init_h0_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->d_h0[0], b->memory, wave);
    argon2_fill_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->memory, b->d_di_ref_table, wave);
    argon2_finalize_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->d_digests[0], b->memory, wave);
    if (cudaStreamSynchronize(st) != cudaSuccess) {
        rpt.error = std::string("CUDA warmup failed: ") + cudaGetErrorString(cudaGetLastError());
        return rpt;
    }

    const auto t0 = std::chrono::steady_clock::now();
    while (true) {
        cudaEventRecord(b->te_h2d_start, st);
        cudaMemcpyAsync(b->d_h0[0], b->h_h0[0], static_cast<std::size_t>(wave) * kH0Bytes,
                        cudaMemcpyHostToDevice, st);
        cudaEventRecord(b->te_h2d_end, st);

        argon2_init_h0_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->d_h0[0], b->memory, wave);
        cudaEventRecord(b->te_init_end, st);

        argon2_fill_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->memory, b->d_di_ref_table, wave);
        cudaEventRecord(b->te_fill_end, st);

        argon2_finalize_kernel<<<wave, kThreadsPerHash, 0, st>>>(b->d_digests[0], b->memory, wave);
        cudaEventRecord(b->te_final_end, st);

        cudaMemcpyAsync(b->h_digests[0], b->d_digests[0],
                        static_cast<std::size_t>(wave) * kTagLen,
                        cudaMemcpyDeviceToHost, st);
        cudaEventRecord(b->te_d2h_end, st);

        cudaError_t e = cudaStreamSynchronize(st);
        if (e != cudaSuccess) {
            rpt.error = std::string("CUDA stage timing sync failed: ") + cudaGetErrorString(e);
            return rpt;
        }

        float h2d_ms = 0, init_ms = 0, fill_ms = 0, final_ms = 0, d2h_ms = 0, tot_ms = 0;
        cudaEventElapsedTime(&h2d_ms, b->te_h2d_start, b->te_h2d_end);
        cudaEventElapsedTime(&init_ms, b->te_h2d_end, b->te_init_end);
        cudaEventElapsedTime(&fill_ms, b->te_init_end, b->te_fill_end);
        cudaEventElapsedTime(&final_ms, b->te_fill_end, b->te_final_end);
        cudaEventElapsedTime(&d2h_ms, b->te_final_end, b->te_d2h_end);
        cudaEventElapsedTime(&tot_ms, b->te_h2d_start, b->te_d2h_end);

        WaveStageSample ws{};
        ws.hashes = static_cast<uint64_t>(wave);
        ws.h2d_ms = h2d_ms;
        ws.init_ms = init_ms;
        ws.fill_ms = fill_ms;
        ws.final_ms = final_ms;
        ws.d2h_ms = d2h_ms;
        ws.total_ms = tot_ms;
        rpt.waves.push_back(ws);

        rpt.hashes += ws.hashes;
        rpt.h2d_ms_sum += ws.h2d_ms;
        rpt.init_ms_sum += ws.init_ms;
        rpt.fill_ms_sum += ws.fill_ms;
        rpt.final_ms_sum += ws.final_ms;
        rpt.d2h_ms_sum += ws.d2h_ms;
        rpt.wave_total_ms_sum += ws.total_ms;

        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= seconds) {
            rpt.ok = true;
            rpt.elapsed_sec = elapsed;
            rpt.h_per_s = elapsed > 0.0 ? static_cast<double>(rpt.hashes) / elapsed : 0.0;
            return rpt;
        }
    }
}

AutotuneResult CudaBackend::autotune(
    int device_index,
    double trial_sec,
    std::function<void(const AutotuneEntry&)> on_trial) {
    AutotuneResult res{};

    const auto devs = enumerate_devices();
    if (device_index < 0 || static_cast<std::size_t>(device_index) >= devs.size()) {
        res.error = "invalid CUDA device index";
        return res;
    }
    res.gpu_total_bytes = devs[static_cast<std::size_t>(device_index)].total_mem_bytes;

    std::string err;
    const int max_c = probe_max_concurrency(device_index, err);
    if (max_c < 1) {
        res.error = err.empty() ? "GPU VRAM cannot hold 1 Argon2 workspace" : err;
        return res;
    }

    auto test_c = [&](int c) -> AutotuneEntry {
        AutotuneEntry ent{};
        ent.concurrency = c;
        ent.streams = 1;
        ent.mem_mb = static_cast<double>(c) * 64.0;
        std::string init_err;
        if (!init(device_index, c, 1, init_err)) {
            ent.ok = false;
            ent.err = init_err;
            return ent;
        }
        const CudaBenchResult br = bench_throughput(trial_sec);
        ent.ok = br.ok;
        ent.hashes = br.hashes;
        ent.sec = br.elapsed_sec;
        ent.h_per_s = br.hashes_per_sec;
        ent.err = br.error;
        if (ent.ok && ent.h_per_s > res.best_h_per_s) {
            res.best_h_per_s = ent.h_per_s;
            res.best_concurrency = c;
            res.best_streams = 1;
            res.ok = true;
        }
        return ent;
    };

    std::vector<int> tested;
    auto already_tested = [&](int c) {
        return std::find(tested.begin(), tested.end(), c) != tested.end();
    };
    auto record = [&](int c) {
        if (c < 1 || c > max_c || already_tested(c)) return;
        tested.push_back(c);
        AutotuneEntry ent = test_c(c);
        res.trials.push_back(ent);
        if (on_trial) on_trial(ent);
    };

    // Coarse sweep (step 16, from ~32 up to max_c).
    for (int c = std::min(32, max_c); c <= max_c; c += 16) {
        record(c);
    }
    record(max_c);

    // Fine sweep around best (+-16 step 4).
    const int best_coarse = res.best_concurrency > 0 ? res.best_concurrency : max_c;
    for (int c = best_coarse - 12; c <= best_coarse + 12; c += 4) {
        record(c);
    }

    // Tight sweep around best (+-4 step 2).
    const int best_fine = res.best_concurrency > 0 ? res.best_concurrency : best_coarse;
    for (int c = best_fine - 4; c <= best_fine + 4; c += 2) {
        record(c);
    }

    free_buffers();
    if (!res.ok && res.error.empty()) {
        res.error = "all CUDA autotune trials failed";
    }
    return res;
}

}  // namespace warp_argon2
