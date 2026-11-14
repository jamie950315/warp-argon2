#include "../src/cuda/argon2_fill_block.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace warp_argon2::cuda_detail;

namespace {

__global__ void fill_single_regio_kernel(const uint64_t* __restrict__ prev_in,
                                         const uint64_t* __restrict__ ref_in,
                                         uint64_t* __restrict__ next_out) {
    __shared__ uint64_t stage[4][WARP_A2_STAGE_STRIDE];
    const int tid = threadIdx.x;
    WarpBlockWords prev = load_warp_block_natural_raw(prev_in, tid);
    WarpBlockWords ref = load_warp_block_natural_raw(ref_in, tid);
    fill_block_fullwarp_regio(prev, ref, next_out, stage, tid);
}

__global__ void fill_chain_dd_kernel(const uint64_t* __restrict__ prev0,
                                     const uint64_t* __restrict__ ref_pool,
                                     int pool_n,
                                     uint64_t* __restrict__ final_out,
                                     int steps) {
    __shared__ uint64_t stage[4][WARP_A2_STAGE_STRIDE];
    const int tid = threadIdx.x;
    WarpBlockWords prev = load_warp_block_natural_raw(prev0, tid);
    for (int s = 0; s < steps; ++s) {
        const uint64_t w0 = __shfl_sync(0xffffffffu, prev.a, 0);
        const int ri = static_cast<int>(w0 % static_cast<uint64_t>(pool_n));
        WarpBlockWords ref = load_warp_block_natural_raw(
            ref_pool + static_cast<size_t>(ri) * FB_QWORDS, tid);
        fill_block_fullwarp_regio(prev, ref, final_out, stage, tid);
    }
}

__global__ void transpose_oracle_kernel(int* __restrict__ fail_flag) {
    __shared__ uint64_t stage[4][WARP_A2_STAGE_STRIDE];
    const int tid = threadIdx.x;
    const uint64_t a = (static_cast<uint64_t>(tid) << 32) | 0ull;
    const uint64_t b = (static_cast<uint64_t>(tid) << 32) | 1ull;
    const uint64_t c = (static_cast<uint64_t>(tid) << 32) | 2ull;
    const uint64_t d = (static_cast<uint64_t>(tid) << 32) | 3ull;

    uint64_t sa = a, sb = b, sc = c, sd = d;
    transpose_diag_shared(sa, sb, sc, sd, stage, tid);

    uint64_t ha = a, hb = b, hc = c, hd = d;
    const int group = tid >> 3;
    const int base = regio_shuffle_route_base(tid);
    transpose_diag_shuffle4_in_place(
        ha, hb, hc, hd, group,
        base + ((group ^ 0) << 3),
        base + ((group ^ 1) << 3),
        base + ((group ^ 2) << 3),
        base + ((group ^ 3) << 3));

    if (sa != ha || sb != hb || sc != hc || sd != hd) {
        atomicOr(fail_flag, 1);
    }
}

uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count <= 0) {
        std::fprintf(stderr, "[FAIL] No CUDA device available\n");
        return 1;
    }

    uint64_t seed = 0x123456789abcdef0ULL;
    std::vector<uint64_t> h_prev(FB_QWORDS);
    std::vector<uint64_t> h_ref(FB_QWORDS);
    std::vector<uint64_t> h_cpu(FB_QWORDS, 0);
    std::vector<uint64_t> h_gpu(FB_QWORDS, 0);

    for (int i = 0; i < FB_QWORDS; ++i) {
        h_prev[static_cast<size_t>(i)] = splitmix64(seed);
        h_ref[static_cast<size_t>(i)] = splitmix64(seed);
    }
    fill_block_cpu(h_prev.data(), h_ref.data(), h_cpu.data(), /*with_xor=*/0);

    uint64_t *d_prev = nullptr, *d_ref = nullptr, *d_out = nullptr;
    int* d_fail = nullptr;
    cudaMalloc(&d_prev, FB_QWORDS * sizeof(uint64_t));
    cudaMalloc(&d_ref, FB_QWORDS * sizeof(uint64_t));
    cudaMalloc(&d_out, FB_QWORDS * sizeof(uint64_t));
    cudaMalloc(&d_fail, sizeof(int));

    cudaMemcpy(d_prev, h_prev.data(), FB_QWORDS * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ref, h_ref.data(), FB_QWORDS * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // 1. Single full-warp REG_IO fill_block check.
    fill_single_regio_kernel<<<1, 32>>>(d_prev, d_ref, d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[FAIL] fill_single_regio_kernel failed: %s\n",
                     cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    cudaMemcpy(h_gpu.data(), d_out, FB_QWORDS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < FB_QWORDS; ++i) {
        if (h_gpu[static_cast<size_t>(i)] != h_cpu[static_cast<size_t>(i)]) {
            std::fprintf(stderr, "[FAIL] fill_block mismatch at qword %d\n", i);
            return 1;
        }
    }
    std::printf("[PASS] CUDA full-warp REG_IO+DIAG_OUT fill_block (128/128 qwords match CPU)\n");

    // 2. Device transpose oracle (shared stage vs 4-color shuffle).
    cudaMemset(d_fail, 0, sizeof(int));
    transpose_oracle_kernel<<<1, 32>>>(d_fail);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[FAIL] transpose_oracle_kernel failed\n");
        return 1;
    }
    int h_fail = 1;
    cudaMemcpy(&h_fail, d_fail, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_fail != 0) {
        std::fprintf(stderr, "[FAIL] transpose oracle reported layout divergence\n");
        return 1;
    }
    std::printf("[PASS] CUDA DIAG_OUT shared stage == 4-color pure-shuffle transpose oracle\n");

    // 3. Multi-step DD chain check.
    constexpr int kPoolN = 16;
    constexpr int kSteps = 32;
    std::vector<uint64_t> h_pool(kPoolN * FB_QWORDS);
    for (size_t i = 0; i < h_pool.size(); ++i) h_pool[i] = splitmix64(seed);
    uint64_t* d_pool = nullptr;
    cudaMalloc(&d_pool, h_pool.size() * sizeof(uint64_t));
    cudaMemcpy(d_pool, h_pool.data(), h_pool.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    std::vector<uint64_t> cpu_chain = h_prev;
    std::vector<uint64_t> cpu_next(FB_QWORDS);
    for (int s = 0; s < kSteps; ++s) {
        const int ri = static_cast<int>(cpu_chain[0] % static_cast<uint64_t>(kPoolN));
        fill_block_cpu(cpu_chain.data(), &h_pool[static_cast<size_t>(ri) * FB_QWORDS],
                       cpu_next.data(), /*with_xor=*/0);
        cpu_chain.swap(cpu_next);
    }

    fill_chain_dd_kernel<<<1, 32>>>(d_prev, d_pool, kPoolN, d_out, kSteps);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "[FAIL] fill_chain_dd_kernel failed\n");
        return 1;
    }
    cudaMemcpy(h_gpu.data(), d_out, FB_QWORDS * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < FB_QWORDS; ++i) {
        if (h_gpu[static_cast<size_t>(i)] != cpu_chain[static_cast<size_t>(i)]) {
            std::fprintf(stderr, "[FAIL] DD chain mismatch at qword %d\n", i);
            return 1;
        }
    }
    std::printf("[PASS] CUDA %d-step DD chain matched CPU reference\n", kSteps);

    cudaFree(d_pool);
    cudaFree(d_prev);
    cudaFree(d_ref);
    cudaFree(d_out);
    cudaFree(d_fail);
    return 0;
}
