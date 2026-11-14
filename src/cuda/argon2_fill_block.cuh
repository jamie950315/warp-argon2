#pragma once

#include "blake2b.cuh"
#include <cstdint>

#ifndef WARP_A2_FULL_WARP_BLAMKA
#define WARP_A2_FULL_WARP_BLAMKA 1
#endif
#ifndef WARP_A2_FULL_WARP_REG_IO
#define WARP_A2_FULL_WARP_REG_IO 1
#endif
#ifndef WARP_A2_FULL_WARP_BARRIER_MODE
#define WARP_A2_FULL_WARP_BARRIER_MODE 0
#endif
#ifndef WARP_A2_FULL_WARP_DIAG_OUT
#define WARP_A2_FULL_WARP_DIAG_OUT 1
#endif
#ifndef WARP_A2_STAGE_STRIDE
#define WARP_A2_STAGE_STRIDE 32
#endif
#ifndef WARP_A2_WORKSPACE_LOAD_POLICY
#define WARP_A2_WORKSPACE_LOAD_POLICY 2
#endif
#ifndef WARP_A2_WORKSPACE_LOAD_IMPL
#define WARP_A2_WORKSPACE_LOAD_IMPL 2
#endif
#ifndef WARP_A2_WORKSPACE_VISIBILITY_MODE
#define WARP_A2_WORKSPACE_VISIBILITY_MODE 1
#endif
#ifndef WARP_A2_WORKSPACE_STORE_POLICY
#define WARP_A2_WORKSPACE_STORE_POLICY 0
#endif
#ifndef WARP_A2_REGIO_TRANSPOSE_MASK
#define WARP_A2_REGIO_TRANSPOSE_MASK 0
#endif
#ifndef WARP_A2_DD_PIPELINE_POINT
#define WARP_A2_DD_PIPELINE_POINT 0
#endif
#ifndef WARP_A2_DD_POST_STORE_SYNC
#define WARP_A2_DD_POST_STORE_SYNC 1
#endif

#if (WARP_A2_REGIO_TRANSPOSE_MASK > 0) && \
    (!(WARP_A2_FULL_WARP_REG_IO) || !(WARP_A2_FULL_WARP_DIAG_OUT))
#error "WARP_A2_REGIO_TRANSPOSE_MASK > 0 requires FULL_WARP_REG_IO and FULL_WARP_DIAG_OUT"
#endif

#define WARP_A2_REGIO_TRANSPOSE_P1_SHUFFLE ((WARP_A2_REGIO_TRANSPOSE_MASK & 1) != 0)
#define WARP_A2_REGIO_TRANSPOSE_P2_SHUFFLE ((WARP_A2_REGIO_TRANSPOSE_MASK & 2) != 0)

namespace warp_argon2 {
namespace cuda_detail {

constexpr int FB_QWORDS = 128;
constexpr unsigned FB_WMASK = 0xffffffffu;

struct WarpBlockWords {
    uint64_t a;
    uint64_t b;
    uint64_t c;
    uint64_t d;
};

__device__ __forceinline__ uint64_t load_workspace_u64(const uint64_t* __restrict__ p) {
#if WARP_A2_WORKSPACE_LOAD_POLICY == 0
    return __ldg(p);
#elif WARP_A2_WORKSPACE_LOAD_POLICY == 3 || WARP_A2_WORKSPACE_LOAD_IMPL == 3
    return *p;
#elif WARP_A2_WORKSPACE_LOAD_POLICY == 1
  #if WARP_A2_WORKSPACE_LOAD_IMPL == 2
    return static_cast<uint64_t>(__ldca(reinterpret_cast<const unsigned long long*>(p)));
  #elif WARP_A2_WORKSPACE_LOAD_IMPL == 1
    uint64_t v;
    asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
  #else
    uint64_t v;
    asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
  #endif
#else
  #if WARP_A2_WORKSPACE_LOAD_IMPL == 2
    return static_cast<uint64_t>(__ldcg(reinterpret_cast<const unsigned long long*>(p)));
  #elif WARP_A2_WORKSPACE_LOAD_IMPL == 1
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
  #else
    uint64_t v;
    asm volatile("ld.global.cg.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
  #endif
#endif
}

__device__ __forceinline__ ulonglong2 load_workspace_u2(const ulonglong2* __restrict__ p) {
#if WARP_A2_WORKSPACE_LOAD_POLICY == 0
    return __ldg(p);
#elif WARP_A2_WORKSPACE_LOAD_POLICY == 3 || WARP_A2_WORKSPACE_LOAD_IMPL == 3
    return *p;
#elif WARP_A2_WORKSPACE_LOAD_POLICY == 1
  #if WARP_A2_WORKSPACE_LOAD_IMPL == 2
    return __ldca(p);
  #elif WARP_A2_WORKSPACE_LOAD_IMPL == 1
    ulonglong2 v;
    asm volatile("ld.global.ca.v2.u64 {%0, %1}, [%2];"
                 : "=l"(v.x), "=l"(v.y)
                 : "l"(p)
                 : "memory");
    return v;
  #else
    ulonglong2 v;
    asm volatile("ld.global.ca.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
    return v;
  #endif
#else
  #if WARP_A2_WORKSPACE_LOAD_IMPL == 2
    return __ldcg(p);
  #elif WARP_A2_WORKSPACE_LOAD_IMPL == 1
    ulonglong2 v;
    asm volatile("ld.global.cg.v2.u64 {%0, %1}, [%2];"
                 : "=l"(v.x), "=l"(v.y)
                 : "l"(p)
                 : "memory");
    return v;
  #else
    ulonglong2 v;
    asm volatile("ld.global.cg.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p));
    return v;
  #endif
#endif
}

__device__ __forceinline__ uint32_t load_immutable_di_index(const uint32_t* __restrict__ p) {
    return __ldg(p);
}

__device__ __forceinline__ WarpBlockWords load_warp_block_natural(
    const uint64_t* __restrict__ blk, int tid) {
    const int tile = tid >> 2;
    const int l = tid & 3;
    WarpBlockWords w;
    w.a = load_workspace_u64(blk + 16 * tile + l + 0);
    w.b = load_workspace_u64(blk + 16 * tile + l + 4);
    w.c = load_workspace_u64(blk + 16 * tile + l + 8);
    w.d = load_workspace_u64(blk + 16 * tile + l + 12);
    return w;
}

__device__ __forceinline__ WarpBlockWords load_warp_block_natural_raw(
    const uint64_t* __restrict__ blk, int tid) {
    const int tile = tid >> 2;
    const int l = tid & 3;
    WarpBlockWords w;
    w.a = blk[16 * tile + l + 0];
    w.b = blk[16 * tile + l + 4];
    w.c = blk[16 * tile + l + 8];
    w.d = blk[16 * tile + l + 12];
    return w;
}

// Keep compiler-default global stores: inline PTX st.global.* triggers a sticky driver
// InvalidAddressSpace error on CUDA 13.3 + sm_89 when mixed into the fill module.
__device__ __forceinline__ void store_workspace_u64(uint64_t* __restrict__ p, uint64_t v) {
    *p = v;
}

__device__ __forceinline__ void store_workspace_u2(ulonglong2* __restrict__ p, ulonglong2 v) {
    *p = v;
}

__device__ __forceinline__ void store_warp_block_natural(uint64_t* __restrict__ blk,
                                                         const WarpBlockWords& w, int tid) {
    const int tile = tid >> 2;
    const int l = tid & 3;
    blk[16 * tile + l + 0] = w.a;
    blk[16 * tile + l + 4] = w.b;
    blk[16 * tile + l + 8] = w.c;
    blk[16 * tile + l + 12] = w.d;
}

__device__ __forceinline__ void store_warp_block_workspace(uint64_t* __restrict__ blk,
                                                           const WarpBlockWords& w, int tid) {
    const int tile = tid >> 2;
    const int l = tid & 3;
    store_workspace_u64(blk + 16 * tile + l + 0, w.a);
    store_workspace_u64(blk + 16 * tile + l + 4, w.b);
    store_workspace_u64(blk + 16 * tile + l + 8, w.c);
    store_workspace_u64(blk + 16 * tile + l + 12, w.d);
}

__device__ __forceinline__ void workspace_post_store_visibility() {
#if WARP_A2_WORKSPACE_VISIBILITY_MODE == 0
    __threadfence_block();
    __syncwarp();
#elif WARP_A2_WORKSPACE_VISIBILITY_MODE == 1
    __syncwarp();
#endif
}

inline uint64_t host_rotr64(uint64_t x, unsigned n) {
    return (x >> n) | (x << (64u - n));
}
inline uint64_t host_fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = UINT64_C(0xFFFFFFFF);
    return x + y + 2 * ((x & m) * (y & m));
}
inline void host_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d) {
    a = host_fBlaMka(a, b);
    d = host_rotr64(d ^ a, 32);
    c = host_fBlaMka(c, d);
    b = host_rotr64(b ^ c, 24);
    a = host_fBlaMka(a, b);
    d = host_rotr64(d ^ a, 16);
    c = host_fBlaMka(c, d);
    b = host_rotr64(b ^ c, 63);
}
inline void host_blake2_round_16(uint64_t* v) {
    host_G(v[0], v[4], v[8], v[12]);
    host_G(v[1], v[5], v[9], v[13]);
    host_G(v[2], v[6], v[10], v[14]);
    host_G(v[3], v[7], v[11], v[15]);
    host_G(v[0], v[5], v[10], v[15]);
    host_G(v[1], v[6], v[11], v[12]);
    host_G(v[2], v[7], v[8], v[13]);
    host_G(v[3], v[4], v[9], v[14]);
}

inline void fill_block_cpu(const uint64_t prev[FB_QWORDS],
                           const uint64_t ref[FB_QWORDS],
                           uint64_t next_io[FB_QWORDS],
                           int with_xor) {
    uint64_t blockR[FB_QWORDS];
    uint64_t block_tmp[FB_QWORDS];
    for (int i = 0; i < FB_QWORDS; ++i) {
        blockR[i] = ref[i] ^ prev[i];
        block_tmp[i] = blockR[i];
    }
    if (with_xor) {
        for (int i = 0; i < FB_QWORDS; ++i) block_tmp[i] ^= next_io[i];
    }
    for (int i = 0; i < 8; ++i) host_blake2_round_16(blockR + 16 * i);
    for (int i = 0; i < 8; ++i) {
        uint64_t v[16] = {
            blockR[2 * i],       blockR[2 * i + 1],   blockR[2 * i + 16],  blockR[2 * i + 17],
            blockR[2 * i + 32],  blockR[2 * i + 33],  blockR[2 * i + 48],  blockR[2 * i + 49],
            blockR[2 * i + 64],  blockR[2 * i + 65],  blockR[2 * i + 80],  blockR[2 * i + 81],
            blockR[2 * i + 96],  blockR[2 * i + 97],  blockR[2 * i + 112], blockR[2 * i + 113]};
        host_blake2_round_16(v);
        blockR[2 * i] = v[0];
        blockR[2 * i + 1] = v[1];
        blockR[2 * i + 16] = v[2];
        blockR[2 * i + 17] = v[3];
        blockR[2 * i + 32] = v[4];
        blockR[2 * i + 33] = v[5];
        blockR[2 * i + 48] = v[6];
        blockR[2 * i + 49] = v[7];
        blockR[2 * i + 64] = v[8];
        blockR[2 * i + 65] = v[9];
        blockR[2 * i + 80] = v[10];
        blockR[2 * i + 81] = v[11];
        blockR[2 * i + 96] = v[12];
        blockR[2 * i + 97] = v[13];
        blockR[2 * i + 112] = v[14];
        blockR[2 * i + 113] = v[15];
    }
    for (int i = 0; i < FB_QWORDS; ++i) next_io[i] = block_tmp[i] ^ blockR[i];
}

__device__ __forceinline__ uint64_t fb_fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = UINT64_C(0xFFFFFFFF);
    return x + y + 2 * ((x & m) * (y & m));
}

__device__ __forceinline__ void fb_blamka_g(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d) {
    a = fb_fBlaMka(a, b);
    d = rotr64(d ^ a, 32);
    c = fb_fBlaMka(c, d);
    b = rotr64(b ^ c, 24);
    a = fb_fBlaMka(a, b);
    d = rotr64(d ^ a, 16);
    c = fb_fBlaMka(c, d);
    b = rotr64(b ^ c, 63);
}

__device__ __forceinline__ void fb_blake2_round_16(
    uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3,
    uint64_t& v4, uint64_t& v5, uint64_t& v6, uint64_t& v7,
    uint64_t& v8, uint64_t& v9, uint64_t& v10, uint64_t& v11,
    uint64_t& v12, uint64_t& v13, uint64_t& v14, uint64_t& v15) {
    fb_blamka_g(v0, v4, v8, v12);
    fb_blamka_g(v1, v5, v9, v13);
    fb_blamka_g(v2, v6, v10, v14);
    fb_blamka_g(v3, v7, v11, v15);
    fb_blamka_g(v0, v5, v10, v15);
    fb_blamka_g(v1, v6, v11, v12);
    fb_blamka_g(v2, v7, v8, v13);
    fb_blamka_g(v3, v4, v9, v14);
}

__device__ __forceinline__ void blamka_tile_round(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d,
                                                   int l) {
    fb_blamka_g(a, b, c, d);
    uint64_t bd = __shfl_sync(FB_WMASK, b, (l + 1) & 3, 4);
    uint64_t cd = __shfl_sync(FB_WMASK, c, (l + 2) & 3, 4);
    uint64_t dd = __shfl_sync(FB_WMASK, d, (l + 3) & 3, 4);
    fb_blamka_g(a, bd, cd, dd);
    b = __shfl_sync(FB_WMASK, bd, (l + 3) & 3, 4);
    c = __shfl_sync(FB_WMASK, cd, (l + 2) & 3, 4);
    d = __shfl_sync(FB_WMASK, dd, (l + 1) & 3, 4);
}

__device__ __forceinline__ void blamka_tile_round_diag_out(uint64_t& a, uint64_t& b, uint64_t& c,
                                                            uint64_t& d, int l) {
    fb_blamka_g(a, b, c, d);
    b = __shfl_sync(FB_WMASK, b, (l + 1) & 3, 4);
    c = __shfl_sync(FB_WMASK, c, (l + 2) & 3, 4);
    d = __shfl_sync(FB_WMASK, d, (l + 3) & 3, 4);
    fb_blamka_g(a, b, c, d);
}

__device__ __forceinline__ void transpose_diag_shared(uint64_t& a, uint64_t& b, uint64_t& c,
                                                       uint64_t& d,
                                                       uint64_t stage[4][WARP_A2_STAGE_STRIDE],
                                                       int tid) {
    const int tile = tid >> 2;
    const int l = tid & 3;
    {
        uint64_t x[4] = {a, b, c, d};
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int logical_l = (l + k) & 3;
            const int o = logical_l + 4 * k;
            const int row = o >> 1;
            const int pos = 2 * tile + (o & 1);
            stage[pos >> 2][4 * row + (pos & 3)] = x[k];
        }
    }
    __syncwarp();
    a = stage[0][tid];
    b = stage[1][tid];
    c = stage[2][tid];
    d = stage[3][tid];
}

__device__ __forceinline__ int regio_shuffle_route_base(int tid) {
    const int group = tid >> 3;
    const int logical_l = ((tid >> 1) & 2) | (tid & 1);
    return (((tid >> 1) & 1) << 2) | ((logical_l - group) & 3);
}

__device__ __forceinline__ void xor_permute4_in_place(uint64_t& a, uint64_t& b, uint64_t& c,
                                                      uint64_t& d, int group) {
    if (group == 1) {
        uint64_t t = a; a = b; b = t; t = c; c = d; d = t;
    } else if (group == 2) {
        uint64_t t = a; a = c; c = t; t = b; b = d; d = t;
    } else if (group == 3) {
        uint64_t t = a; a = d; d = t; t = b; b = c; c = t;
    }
}

__device__ __forceinline__ void transpose_diag_shuffle4_in_place(uint64_t& a, uint64_t& b,
                                                                  uint64_t& c, uint64_t& d,
                                                                  int group, int src0, int src1,
                                                                  int src2, int src3) {
    xor_permute4_in_place(a, b, c, d, group);
    a = __shfl_sync(FB_WMASK, a, src0);
    b = __shfl_sync(FB_WMASK, b, src1);
    c = __shfl_sync(FB_WMASK, c, src2);
    d = __shfl_sync(FB_WMASK, d, src3);
    xor_permute4_in_place(a, b, c, d, group);
}

struct FullWarpRegioPhase2 {
    uint64_t t0, t1, t2, t3;
    uint64_t a, b, c, d;
};

__device__ __forceinline__ FullWarpRegioPhase2 fill_block_fullwarp_regio_to_phase2(
    const WarpBlockWords& prev,
    const WarpBlockWords& ref,
    uint64_t stage[4][WARP_A2_STAGE_STRIDE],
    int tid) {
    FullWarpRegioPhase2 p;
    const int l = tid & 3;
    p.a = prev.a ^ ref.a;
    p.b = prev.b ^ ref.b;
    p.c = prev.c ^ ref.c;
    p.d = prev.d ^ ref.d;
    p.t0 = p.a;
    p.t1 = p.b;
    p.t2 = p.c;
    p.t3 = p.d;

#if WARP_A2_FULL_WARP_DIAG_OUT
    blamka_tile_round_diag_out(p.a, p.b, p.c, p.d, l);
#else
    blamka_tile_round(p.a, p.b, p.c, p.d, l);
#endif
#if WARP_A2_FULL_WARP_BARRIER_MODE == 0
    __syncwarp();
#endif

#if WARP_A2_REGIO_TRANSPOSE_P1_SHUFFLE
    {
        const int group = tid >> 3;
        const int base = regio_shuffle_route_base(tid);
        (void)stage;
        transpose_diag_shuffle4_in_place(p.a, p.b, p.c, p.d, group, base + ((group ^ 0) << 3),
                                         base + ((group ^ 1) << 3), base + ((group ^ 2) << 3),
                                         base + ((group ^ 3) << 3));
    }
#else
    transpose_diag_shared(p.a, p.b, p.c, p.d, stage, tid);
#endif

#if WARP_A2_FULL_WARP_DIAG_OUT
    blamka_tile_round_diag_out(p.a, p.b, p.c, p.d, l);
#else
    blamka_tile_round(p.a, p.b, p.c, p.d, l);
#endif
#if WARP_A2_FULL_WARP_BARRIER_MODE == 0
    __syncwarp();
#endif
    return p;
}

__device__ __forceinline__ uint32_t dd_mainnet_ref_index(uint32_t slice, uint32_t i,
                                                         uint64_t word0) {
    const uint32_t pseudo32 = static_cast<uint32_t>(word0);
    const uint32_t area = slice * 16384u + i - 1u;
    const uint32_t sq_hi = __umulhi(pseudo32, pseudo32);
    return area - 1u - __umulhi(area, sq_hi);
}

__device__ __forceinline__ WarpBlockWords fill_block_fullwarp_regio_finish(
    FullWarpRegioPhase2 p,
    uint64_t* __restrict__ next,
    uint64_t stage[4][WARP_A2_STAGE_STRIDE],
    int tid,
    int do_post_store_sync) {
#if WARP_A2_REGIO_TRANSPOSE_P2_SHUFFLE
    {
        const int group = tid >> 3;
        const int base = regio_shuffle_route_base(tid);
        (void)stage;
        transpose_diag_shuffle4_in_place(p.a, p.b, p.c, p.d, group, base + ((group ^ 0) << 3),
                                         base + ((group ^ 1) << 3), base + ((group ^ 2) << 3),
                                         base + ((group ^ 3) << 3));
    }
#else
    transpose_diag_shared(p.a, p.b, p.c, p.d, stage, tid);
#endif
    WarpBlockWords out;
    out.a = p.t0 ^ p.a;
    out.b = p.t1 ^ p.b;
    out.c = p.t2 ^ p.c;
    out.d = p.t3 ^ p.d;
    store_warp_block_workspace(next, out, tid);
    if (do_post_store_sync) {
        workspace_post_store_visibility();
    }
    return out;
}

__device__ __forceinline__ void fill_block_fullwarp_regio(
    WarpBlockWords& prev,
    const WarpBlockWords& ref,
    uint64_t* __restrict__ next,
    uint64_t stage[4][WARP_A2_STAGE_STRIDE],
    int tid) {
    FullWarpRegioPhase2 ph = fill_block_fullwarp_regio_to_phase2(prev, ref, stage, tid);
    prev = fill_block_fullwarp_regio_finish(ph, next, stage, tid, /*do_post_store_sync=*/1);
}

__device__ __forceinline__ void fill_block_fullwarp_regio_dd(
    WarpBlockWords& prev,
    const WarpBlockWords& ref,
    uint64_t* __restrict__ next,
    uint64_t stage[4][WARP_A2_STAGE_STRIDE],
    int tid,
    int has_next,
    const uint64_t* __restrict__ hash_pool_base,
    uint32_t slice,
    uint32_t next_i,
    WarpBlockWords& next_ref_out) {
#if WARP_A2_DD_PIPELINE_POINT == 0
    (void)has_next;
    (void)hash_pool_base;
    (void)slice;
    (void)next_i;
    (void)next_ref_out;
    fill_block_fullwarp_regio(prev, ref, next, stage, tid);
#else
    const int l = tid & 3;
    uint64_t a = prev.a ^ ref.a;
    uint64_t b = prev.b ^ ref.b;
    uint64_t c = prev.c ^ ref.c;
    uint64_t d = prev.d ^ ref.d;
    const uint64_t t0 = a, t1 = b, t2 = c, t3 = d;

#if WARP_A2_FULL_WARP_DIAG_OUT
    blamka_tile_round_diag_out(a, b, c, d, l);
#else
    blamka_tile_round(a, b, c, d, l);
#endif
#if WARP_A2_FULL_WARP_BARRIER_MODE == 0
    __syncwarp();
#endif

#if WARP_A2_REGIO_TRANSPOSE_P1_SHUFFLE
    {
        const int group = tid >> 3;
        const int base = regio_shuffle_route_base(tid);
        transpose_diag_shuffle4_in_place(a, b, c, d, group, base + ((group ^ 0) << 3),
                                         base + ((group ^ 1) << 3), base + ((group ^ 2) << 3),
                                         base + ((group ^ 3) << 3));
    }
#else
    transpose_diag_shared(a, b, c, d, stage, tid);
#endif
#if WARP_A2_FULL_WARP_DIAG_OUT
    blamka_tile_round_diag_out(a, b, c, d, l);
#else
    blamka_tile_round(a, b, c, d, l);
#endif
#if WARP_A2_FULL_WARP_BARRIER_MODE == 0
    __syncwarp();
#endif

    const uint32_t nri =
        has_next ? dd_mainnet_ref_index(slice, next_i, __shfl_sync(FB_WMASK, t0 ^ a, 0)) : 0u;

#if WARP_A2_DD_PIPELINE_POINT == 3
    if (has_next) {
        next_ref_out = load_warp_block_natural(hash_pool_base + static_cast<size_t>(nri) * FB_QWORDS, tid);
    }
#endif

#if WARP_A2_REGIO_TRANSPOSE_P2_SHUFFLE
    {
        const int group = tid >> 3;
        const int base = regio_shuffle_route_base(tid);
        transpose_diag_shuffle4_in_place(a, b, c, d, group, base + ((group ^ 0) << 3),
                                         base + ((group ^ 1) << 3), base + ((group ^ 2) << 3),
                                         base + ((group ^ 3) << 3));
    }
#else
    transpose_diag_shared(a, b, c, d, stage, tid);
#endif

#if WARP_A2_DD_PIPELINE_POINT == 2
    if (has_next) {
        next_ref_out = load_warp_block_natural(hash_pool_base + static_cast<size_t>(nri) * FB_QWORDS, tid);
    }
#endif

    WarpBlockWords out;
    out.a = t0 ^ a;
    out.b = t1 ^ b;
    out.c = t2 ^ c;
    out.d = t3 ^ d;

#if WARP_A2_DD_PIPELINE_POINT == 1
    if (has_next) {
        next_ref_out = load_warp_block_natural(hash_pool_base + static_cast<size_t>(nri) * FB_QWORDS, tid);
    }
#endif

    store_warp_block_workspace(next, out, tid);
#if WARP_A2_DD_POST_STORE_SYNC
    workspace_post_store_visibility();
#endif
    prev = out;
#endif
}

__device__ __forceinline__ void fill_block_coop_legacy_sh(const uint64_t* __restrict__ prev,
                                                          const uint64_t* __restrict__ ref,
                                                          uint64_t* __restrict__ next,
                                                          int with_xor,
                                                          uint64_t* __restrict__ sh_R,
                                                          int tid) {
    const int j0 = tid * 2;
    const ulonglong2* __restrict__ rp = reinterpret_cast<const ulonglong2*>(ref);
    const ulonglong2* __restrict__ pp = reinterpret_cast<const ulonglong2*>(prev);
    ulonglong2 a0 = rp[j0];
    ulonglong2 b0 = pp[j0];
    ulonglong2 a1 = rp[j0 + 1];
    ulonglong2 b1 = pp[j0 + 1];
    ulonglong2 r0 = make_ulonglong2(a0.x ^ b0.x, a0.y ^ b0.y);
    ulonglong2 r1 = make_ulonglong2(a1.x ^ b1.x, a1.y ^ b1.y);
    ulonglong2 t0, t1;
    if (with_xor) {
        const ulonglong2* __restrict__ np = reinterpret_cast<const ulonglong2*>(next);
        ulonglong2 n0 = np[j0];
        ulonglong2 n1 = np[j0 + 1];
        t0 = make_ulonglong2(r0.x ^ n0.x, r0.y ^ n0.y);
        t1 = make_ulonglong2(r1.x ^ n1.x, r1.y ^ n1.y);
    } else {
        t0 = r0;
        t1 = r1;
    }
    ulonglong2* __restrict__ Rd = reinterpret_cast<ulonglong2*>(sh_R);
    Rd[j0] = r0;
    Rd[j0 + 1] = r1;
    __syncwarp();
    if (tid < 8) {
        const int i = tid * 16;
        fb_blake2_round_16(sh_R[i + 0], sh_R[i + 1], sh_R[i + 2], sh_R[i + 3],
                           sh_R[i + 4], sh_R[i + 5], sh_R[i + 6], sh_R[i + 7],
                           sh_R[i + 8], sh_R[i + 9], sh_R[i + 10], sh_R[i + 11],
                           sh_R[i + 12], sh_R[i + 13], sh_R[i + 14], sh_R[i + 15]);
    }
    __syncwarp();
    if (tid < 8) {
        const int i = tid;
        fb_blake2_round_16(sh_R[2 * i], sh_R[2 * i + 1], sh_R[2 * i + 16], sh_R[2 * i + 17],
                           sh_R[2 * i + 32], sh_R[2 * i + 33], sh_R[2 * i + 48], sh_R[2 * i + 49],
                           sh_R[2 * i + 64], sh_R[2 * i + 65], sh_R[2 * i + 80], sh_R[2 * i + 81],
                           sh_R[2 * i + 96], sh_R[2 * i + 97], sh_R[2 * i + 112], sh_R[2 * i + 113]);
    }
    __syncwarp();
    {
        ulonglong2* __restrict__ nd = reinterpret_cast<ulonglong2*>(next);
        ulonglong2 rr0 = Rd[j0];
        ulonglong2 rr1 = Rd[j0 + 1];
        nd[j0] = make_ulonglong2(t0.x ^ rr0.x, t0.y ^ rr0.y);
        nd[j0 + 1] = make_ulonglong2(t1.x ^ rr1.x, t1.y ^ rr1.y);
    }
    __syncwarp();
}

}  // namespace cuda_detail
}  // namespace warp_argon2
