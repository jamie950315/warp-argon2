#pragma once

#include "blake2b.cuh"
#include "argon2_fill_block.cuh"

#include <cstdint>

#ifndef WARP_A2_FORWARD_PREV
#define WARP_A2_FORWARD_PREV 1
#endif
#ifndef WARP_A2_DI_INDEX_TABLE
#define WARP_A2_DI_INDEX_TABLE 1
#endif
#ifndef WARP_A2_REF_PREFETCH
#define WARP_A2_REF_PREFETCH 1
#endif
#ifndef WARP_A2_KERNEL_SPLIT
#define WARP_A2_KERNEL_SPLIT 1
#endif
#ifndef WARP_A2_SPECIALIZED_FILL
#define WARP_A2_SPECIALIZED_FILL 1
#endif
#ifndef WARP_A2_FAST_FILL_SHARED
#define WARP_A2_FAST_FILL_SHARED 1
#endif
#ifndef WARP_A2_DI_PREFETCH_WINDOW
#define WARP_A2_DI_PREFETCH_WINDOW 1
#endif
#ifndef WARP_A2_STAGE_STRIDE
#define WARP_A2_STAGE_STRIDE 32
#endif

#if WARP_A2_DI_PREFETCH_WINDOW >= 3
#error "WARP_A2_DI_PREFETCH_WINDOW >= 3 is invalid: distance-2 DI ref(j+2) can read ungenerated block j"
#endif

#define WARP_A2_REGIO_NEED_STAGE_SMEM (WARP_A2_REGIO_TRANSPOSE_MASK != 3)

namespace warp_argon2 {
namespace cuda_detail {

constexpr uint32_t A2_BLOCK_SIZE = 1024;
constexpr uint32_t A2_QWORDS = 128;
constexpr uint32_t A2_SYNC_POINTS = 4;
constexpr uint32_t A2_MEMORY_BLOCKS = 65536;
constexpr uint32_t A2_SEGMENT_LENGTH = 16384;
constexpr uint32_t A2_PASSES = 1;
constexpr uint32_t A2_LANES = 1;
constexpr uint32_t A2_OUT_LEN = 32;
constexpr uint32_t A2_PREHASH_DIGEST_LENGTH = 64;
constexpr uint32_t A2_PREHASH_SEED_LENGTH = 72;
constexpr uint32_t A2_VERSION = 0x13;
constexpr uint32_t A2_TYPE_ID = 2;
constexpr int A2_THREADS = 32;
constexpr unsigned A2_WMASK = 0xffffffffu;

struct Block {
    uint64_t v[A2_QWORDS];
};

struct FillShared {
#if WARP_A2_REGIO_NEED_STAGE_SMEM
    alignas(128) uint64_t stage[4][WARP_A2_STAGE_STRIDE];
#else
    unsigned char unused;
#endif
};

struct InitShared {
    uint8_t blockhash[A2_PREHASH_SEED_LENGTH];
    uint8_t blockhash_bytes[A2_BLOCK_SIZE];
};

struct FinalShared {
    uint8_t blockhash_bytes[A2_BLOCK_SIZE];
    uint8_t out32[A2_OUT_LEN];
};

__device__ __forceinline__ void load_block(Block* dst, const uint8_t* input) {
    for (int i = 0; i < static_cast<int>(A2_QWORDS); ++i) {
        dst->v[i] = load64(input + i * 8);
    }
}

__device__ __forceinline__ void store_block(uint8_t* output, const Block* src) {
    for (int i = 0; i < static_cast<int>(A2_QWORDS); ++i) {
        store64(output + i * 8, src->v[i]);
    }
}

__device__ __forceinline__ uint64_t (*fill_sh_stage(FillShared* sh))[WARP_A2_STAGE_STRIDE] {
#if WARP_A2_REGIO_NEED_STAGE_SMEM
    return sh->stage;
#else
    (void)sh;
    return nullptr;
#endif
}

__device__ __forceinline__ void init_b0_b1_from_blockhash(Block* __restrict__ pool,
                                                          int hash_idx,
                                                          InitShared* sh) {
    Block* __restrict__ hash_pool = pool + (static_cast<size_t>(hash_idx) << 16);

    store32(sh->blockhash + A2_PREHASH_DIGEST_LENGTH, 0);
    store32(sh->blockhash + A2_PREHASH_DIGEST_LENGTH + 4, 0);
    blake2b_long(sh->blockhash_bytes, A2_BLOCK_SIZE, sh->blockhash, A2_PREHASH_SEED_LENGTH);
    load_block(&hash_pool[0], sh->blockhash_bytes);

    store32(sh->blockhash + A2_PREHASH_DIGEST_LENGTH, 1);
    blake2b_long(sh->blockhash_bytes, A2_BLOCK_SIZE, sh->blockhash, A2_PREHASH_SEED_LENGTH);
    load_block(&hash_pool[1], sh->blockhash_bytes);
}

// Initialize B[0], B[1] from precomputed 64-byte RFC 9106 H0.
__device__ void argon2id_init_from_h0(const uint8_t* h0,
                                      Block* __restrict__ pool,
                                      int hash_idx,
                                      InitShared* sh,
                                      int tid) {
    if (tid == 0) {
        for (uint32_t i = 0; i < A2_PREHASH_DIGEST_LENGTH; ++i) {
            sh->blockhash[i] = h0[i];
        }
        init_b0_b1_from_blockhash(pool, hash_idx, sh);
    }
    __syncwarp();
}

// Compute RFC 9106 H0 on-device for pwd = LE64(index), salt = salt16, empty secret/AD.
__device__ void argon2id_init_indexed(const uint8_t* salt16,
                                      uint64_t index,
                                      Block* __restrict__ pool,
                                      int hash_idx,
                                      InitShared* sh,
                                      int tid) {
    if (tid == 0) {
        uint8_t pwd[8];
        store64(pwd, index);

        Blake2bState state;
        blake2b_init(&state, A2_PREHASH_DIGEST_LENGTH);

        uint8_t value[4];
        store32(value, A2_LANES); blake2b_update(&state, value, 4);
        store32(value, A2_OUT_LEN); blake2b_update(&state, value, 4);
        store32(value, A2_MEMORY_BLOCKS); blake2b_update(&state, value, 4);
        store32(value, A2_PASSES); blake2b_update(&state, value, 4);
        store32(value, A2_VERSION); blake2b_update(&state, value, 4);
        store32(value, A2_TYPE_ID); blake2b_update(&state, value, 4);

        store32(value, 8);
        blake2b_update(&state, value, 4);
        blake2b_update(&state, pwd, 8);

        store32(value, 16);
        blake2b_update(&state, value, 4);
        blake2b_update(&state, salt16, 16);

        store32(value, 0);
        blake2b_update(&state, value, 4);  // secretlen = 0
        blake2b_update(&state, value, 4);  // adlen = 0

        blake2b_final(&state, sh->blockhash, A2_PREHASH_DIGEST_LENGTH);
        init_b0_b1_from_blockhash(pool, hash_idx, sh);
    }
    __syncwarp();
}

// Production mainnet fill: t=1, m=65536, lanes=1, HASH_MAJOR, REG_IO + DIAG_OUT + DI table.
__device__ void argon2id_fill_matrix_mainnet(Block* __restrict__ pool,
                                             int hash_idx,
                                             FillShared* sh,
                                             int tid,
                                             const uint32_t* __restrict__ di_ref_table) {
    constexpr uint32_t kSeg = A2_SEGMENT_LENGTH;
    Block* __restrict__ hash_pool = pool + (static_cast<size_t>(hash_idx) << 16);
    uint64_t(*stage)[WARP_A2_STAGE_STRIDE] = fill_sh_stage(sh);

    WarpBlockWords prev = load_warp_block_natural(hash_pool[1].v, tid);

    // Slice 0 (DI): blocks 2..16383 with one-ahead reference prefetch.
    {
        uint32_t r0 = load_immutable_di_index(&di_ref_table[2]);
        WarpBlockWords ref = load_warp_block_natural(hash_pool[r0].v, tid);
        for (uint32_t i = 2; i < kSeg; ++i) {
            const int has_next = (i + 1 < kSeg);
            WarpBlockWords nref;
            if (has_next) {
                uint32_t nr = load_immutable_di_index(&di_ref_table[i + 1]);
                nref = load_warp_block_natural(hash_pool[nr].v, tid);
            }
            fill_block_fullwarp_regio(prev, ref, hash_pool[i].v, stage, tid);
            if (has_next) ref = nref;
        }
    }

    // Slice 1 (DI): blocks 16384..32767 with one-ahead reference prefetch.
    {
        constexpr uint32_t base = kSeg;
        uint32_t r0 = load_immutable_di_index(&di_ref_table[kSeg + 0]);
        WarpBlockWords ref = load_warp_block_natural(hash_pool[r0].v, tid);
        for (uint32_t i = 0; i < kSeg; ++i) {
            const int has_next = (i + 1 < kSeg);
            WarpBlockWords nref;
            if (has_next) {
                uint32_t nr = load_immutable_di_index(&di_ref_table[kSeg + i + 1]);
                nref = load_warp_block_natural(hash_pool[nr].v, tid);
            }
            fill_block_fullwarp_regio(prev, ref, hash_pool[base + i].v, stage, tid);
            if (has_next) ref = nref;
        }
    }

    // Slices 2-3 (DD): blocks 32768..65535.
    {
        WarpBlockWords ref;
        {
            const uint64_t pr0 = __shfl_sync(A2_WMASK, prev.a, 0);
            const uint32_t r0 = dd_mainnet_ref_index(2u, 0u, pr0);
            ref = load_warp_block_natural(hash_pool[r0].v, tid);
        }
#pragma unroll 1
        for (uint32_t slice = 2; slice < A2_SYNC_POINTS; ++slice) {
            const uint32_t base = slice * kSeg;
            for (uint32_t i = 0; i < kSeg; ++i) {
                const int has_next_in_slice = (i + 1 < kSeg);
                const int has_next = has_next_in_slice || (slice == 2 && i + 1 == kSeg);
                const uint32_t next_slice = has_next_in_slice ? slice : 3u;
                const uint32_t next_i = has_next_in_slice ? (i + 1) : 0u;
#if WARP_A2_DD_PIPELINE_POINT == 0
                fill_block_fullwarp_regio(prev, ref, hash_pool[base + i].v, stage, tid);
                if (has_next) {
                    const uint64_t pr = __shfl_sync(A2_WMASK, prev.a, 0);
                    const uint32_t nri = dd_mainnet_ref_index(next_slice, next_i, pr);
                    ref = load_warp_block_natural(hash_pool[nri].v, tid);
                }
#else
                WarpBlockWords nref;
                fill_block_fullwarp_regio_dd(prev, ref, hash_pool[base + i].v, stage, tid,
                                             has_next, hash_pool[0].v, next_slice, next_i, nref);
                if (has_next) ref = nref;
#endif
            }
        }
    }
}

__device__ void argon2id_finalize(uint8_t* out32_global,
                                  Block* __restrict__ pool,
                                  int hash_idx,
                                  FinalShared* sh,
                                  int tid) {
    if (tid == 0) {
        const Block* last = &pool[(static_cast<size_t>(hash_idx) << 16) + (A2_MEMORY_BLOCKS - 1u)];
        store_block(sh->blockhash_bytes, last);
        blake2b_long(sh->out32, A2_OUT_LEN, sh->blockhash_bytes, A2_BLOCK_SIZE);
#pragma unroll
        for (uint32_t i = 0; i < A2_OUT_LEN; ++i) {
            out32_global[i] = sh->out32[i];
        }
    }
    __syncwarp();
}

}  // namespace cuda_detail
}  // namespace warp_argon2
