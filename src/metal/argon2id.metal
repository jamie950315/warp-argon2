// Warp-cooperative RFC 9106 Argon2id (m=65536 KiB, t=1, p=1, outlen=32) — Apple Silicon Metal
// 32 threads (one SIMD group) cooperate per hash over a 4-GiB-chunked BLOCK_MAJOR pool.

#include <metal_stdlib>
using namespace metal;

constant uint A2_QWORDS = 128;
constant uint A2_SYNC_POINTS = 4;
constant uint A2_MEMORY_BLOCKS = 65536;
constant uint A2_LANE_LENGTH = 65536;
constant uint A2_SEGMENT_LENGTH = 16384;
constant uint A2_PASSES = 1;
constant uint A2_LANES = 1;
constant uint A2_TYPE_ID = 2;
constant uint A2_VERSION = 0x13;
constant uint A2_H0_LEN = 64;
constant uint A2_OUT_LEN = 32;

constant ulong BLAKE2B_IV[8] = {
    0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL,
    0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL
};

constant uchar BLAKE2B_SIGMA[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}
};

inline ulong rotr64(ulong x, uint n) {
    return (x >> n) | (x << (64 - n));
}

inline ulong load64_t(const thread uchar* p) {
    ulong v = 0;
    for (uint i = 0; i < 8; ++i) v |= ((ulong)p[i]) << (8 * i);
    return v;
}

inline void store64_t(thread uchar* p, ulong v) {
    for (uint i = 0; i < 8; ++i) p[i] = (uchar)((v >> (8 * i)) & 0xff);
}

inline void store32_t(thread uchar* p, uint v) {
    p[0] = (uchar)(v & 0xff);
    p[1] = (uchar)((v >> 8) & 0xff);
    p[2] = (uchar)((v >> 16) & 0xff);
    p[3] = (uchar)((v >> 24) & 0xff);
}

inline ulong fBlaMka(ulong x, ulong y) {
    const ulong m = 0xFFFFFFFFUL;
    return x + y + 2UL * ((x & m) * (y & m));
}

inline ulong simd_shuffle_u64(ulong data, ushort lane) {
    uint2 v = as_type<uint2>(data);
    v.x = simd_shuffle(v.x, lane);
    v.y = simd_shuffle(v.y, lane);
    return as_type<ulong>(v);
}

inline ulong simd_broadcast_u64(ulong data, ushort lane) {
    uint2 v = as_type<uint2>(data);
    v.x = simd_broadcast(v.x, lane);
    v.y = simd_broadcast(v.y, lane);
    return as_type<ulong>(v);
}

inline void blamka_g(thread ulong& a, thread ulong& b, thread ulong& c, thread ulong& d) {
    a = fBlaMka(a, b);
    d = rotr64(d ^ a, 32);
    c = fBlaMka(c, d);
    b = rotr64(b ^ c, 24);
    a = fBlaMka(a, b);
    d = rotr64(d ^ a, 16);
    c = fBlaMka(c, d);
    b = rotr64(b ^ c, 63);
}

struct Blake2bState {
    ulong h[8];
    ulong t[2];
    ulong f[2];
    uchar buf[128];
    uint buflen;
    uint outlen;
};

inline void blake2b_init(thread Blake2bState& S, uint outlen) {
    for (uint i = 0; i < 8; ++i) S.h[i] = BLAKE2B_IV[i];
    S.t[0] = S.t[1] = S.f[0] = S.f[1] = 0;
    S.buflen = 0;
    S.outlen = outlen;
    for (uint i = 0; i < 128; ++i) S.buf[i] = 0;
    uchar P[64];
    for (uint i = 0; i < 64; ++i) P[i] = 0;
    P[0] = (uchar)outlen;
    P[2] = 1;
    P[3] = 1;
    for (uint i = 0; i < 8; ++i) S.h[i] ^= load64_t(&P[i * 8]);
}

inline void blake2b_compress(thread Blake2bState& S, const thread uchar* block) {
    ulong m[16], v[16];
    for (uint i = 0; i < 16; ++i) m[i] = load64_t(block + i * 8);
    for (uint i = 0; i < 8; ++i) v[i] = S.h[i];
    v[8] = BLAKE2B_IV[0];
    v[9] = BLAKE2B_IV[1];
    v[10] = BLAKE2B_IV[2];
    v[11] = BLAKE2B_IV[3];
    v[12] = BLAKE2B_IV[4] ^ S.t[0];
    v[13] = BLAKE2B_IV[5] ^ S.t[1];
    v[14] = BLAKE2B_IV[6] ^ S.f[0];
    v[15] = BLAKE2B_IV[7] ^ S.f[1];
    for (uint r = 0; r < 12; ++r) {
#define B2G(i, a, b, c, d)                                  \
    do {                                                    \
        a = a + b + m[BLAKE2B_SIGMA[r][2 * (i)]];           \
        d = rotr64(d ^ a, 32);                              \
        c = c + d;                                          \
        b = rotr64(b ^ c, 24);                              \
        a = a + b + m[BLAKE2B_SIGMA[r][2 * (i) + 1]];       \
        d = rotr64(d ^ a, 16);                              \
        c = c + d;                                          \
        b = rotr64(b ^ c, 63);                              \
    } while (0)
        B2G(0, v[0], v[4], v[8], v[12]);
        B2G(1, v[1], v[5], v[9], v[13]);
        B2G(2, v[2], v[6], v[10], v[14]);
        B2G(3, v[3], v[7], v[11], v[15]);
        B2G(4, v[0], v[5], v[10], v[15]);
        B2G(5, v[1], v[6], v[11], v[12]);
        B2G(6, v[2], v[7], v[8], v[13]);
        B2G(7, v[3], v[4], v[9], v[14]);
#undef B2G
    }
    for (uint i = 0; i < 8; ++i) S.h[i] = S.h[i] ^ v[i] ^ v[i + 8];
}

inline void blake2b_update(thread Blake2bState& S, const thread uchar* in, uint inlen) {
    uint pin = 0;
    if (S.buflen + inlen > 128) {
        uint left = S.buflen;
        uint fill = 128 - left;
        for (uint i = 0; i < fill; ++i) S.buf[left + i] = in[pin + i];
        S.t[0] += 128;
        if (S.t[0] < 128) S.t[1]++;
        blake2b_compress(S, S.buf);
        S.buflen = 0;
        inlen -= fill;
        pin += fill;
        while (inlen > 128) {
            S.t[0] += 128;
            if (S.t[0] < 128) S.t[1]++;
            uchar tmp[128];
            for (uint i = 0; i < 128; ++i) tmp[i] = in[pin + i];
            blake2b_compress(S, tmp);
            inlen -= 128;
            pin += 128;
        }
    }
    for (uint i = 0; i < inlen; ++i) S.buf[S.buflen + i] = in[pin + i];
    S.buflen += inlen;
}

inline void blake2b_final(thread Blake2bState& S, thread uchar* out, uint outlen) {
    S.t[0] += S.buflen;
    if (S.t[0] < S.buflen) S.t[1]++;
    S.f[0] = 0xFFFFFFFFFFFFFFFFUL;
    for (uint i = S.buflen; i < 128; ++i) S.buf[i] = 0;
    blake2b_compress(S, S.buf);
    uchar buffer[64];
    for (uint i = 0; i < 8; ++i) store64_t(buffer + i * 8, S.h[i]);
    for (uint i = 0; i < outlen; ++i) out[i] = buffer[i];
}

inline void blake2b_simple(thread uchar* out, uint outlen, const thread uchar* in, uint inlen) {
    Blake2bState S;
    blake2b_init(S, outlen);
    blake2b_update(S, in, inlen);
    blake2b_final(S, out, outlen);
}

inline void blake2b_long(thread uchar* out, uint outlen, const thread uchar* in, uint inlen) {
    uchar outlen_bytes[4];
    store32_t(outlen_bytes, outlen);
    if (outlen <= 64) {
        Blake2bState S;
        blake2b_init(S, outlen);
        blake2b_update(S, outlen_bytes, 4);
        blake2b_update(S, in, inlen);
        blake2b_final(S, out, outlen);
    } else {
        uchar out_buffer[64], in_buffer[64];
        Blake2bState S;
        blake2b_init(S, 64);
        blake2b_update(S, outlen_bytes, 4);
        blake2b_update(S, in, inlen);
        blake2b_final(S, out_buffer, 64);
        for (uint i = 0; i < 32; ++i) out[i] = out_buffer[i];
        uint toproduce = outlen - 32;
        uint out_off = 32;
        while (toproduce > 64) {
            for (uint i = 0; i < 64; ++i) in_buffer[i] = out_buffer[i];
            blake2b_simple(out_buffer, 64, in_buffer, 64);
            for (uint i = 0; i < 32; ++i) out[out_off + i] = out_buffer[i];
            out_off += 32;
            toproduce -= 32;
        }
        for (uint i = 0; i < 64; ++i) in_buffer[i] = out_buffer[i];
        blake2b_simple(out_buffer, toproduce, in_buffer, 64);
        for (uint i = 0; i < toproduce; ++i) out[out_off + i] = out_buffer[i];
    }
}

struct WarpWords {
    ulong a, b, c, d;
};

// 4-lane tile BlaMka round leaving diagonals rotated by lane index (DIAG_OUT).
inline void blamka_tile_round_diag_out(thread ulong& a, thread ulong& b, thread ulong& c,
                                       thread ulong& d, int l, uint tid) {
    blamka_g(a, b, c, d);
    ushort base = ushort(tid & ~3u);
    b = simd_shuffle_u64(b, ushort(base | ((l + 1) & 3)));
    c = simd_shuffle_u64(c, ushort(base | ((l + 2) & 3)));
    d = simd_shuffle_u64(d, ushort(base | ((l + 3) & 3)));
    blamka_g(a, b, c, d);
}

// 1 KiB threadgroup transpose stage keeps high Apple GPU occupancy (~5% faster than pure shuffle).
inline void transpose_diag_shared(
    thread ulong& a, thread ulong& b, thread ulong& c, thread ulong& d,
    threadgroup ulong stage[4][32], uint tid
) {
    const int tile = int(tid >> 2);
    const int l = int(tid & 3);
    ulong x[4] = {a, b, c, d};
    for (int k = 0; k < 4; ++k) {
        const int logical_l = (l + k) & 3;
        const int o = logical_l + 4 * k;
        const int row = o >> 1;
        const int pos = 2 * tile + (o & 1);
        stage[pos >> 2][4 * row + (pos & 3)] = x[k];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    a = stage[0][tid];
    b = stage[1][tid];
    c = stage[2][tid];
    d = stage[3][tid];
}

// Block-major pool within each 4 GiB chunk: concurrent hashes write block b side by side.
inline device ulong* block_ptr(device ulong* pool, uint local, uint block_idx, uint H) {
    return pool + ((ulong)block_idx * (ulong)H + (ulong)local) * A2_QWORDS;
}

inline device const ulong* block_ptr_c(device const ulong* pool, uint local, uint block_idx, uint H) {
    return pool + ((ulong)block_idx * (ulong)H + (ulong)local) * A2_QWORDS;
}

inline WarpWords load_warp_words(device const ulong* blk, uint tid) {
    const int tile = int(tid >> 2);
    const int l = int(tid & 3);
    WarpWords w;
    w.a = blk[16 * tile + l + 0];
    w.b = blk[16 * tile + l + 4];
    w.c = blk[16 * tile + l + 8];
    w.d = blk[16 * tile + l + 12];
    return w;
}

inline void store_warp_words(device ulong* blk, WarpWords w, uint tid) {
    const int tile = int(tid >> 2);
    const int l = int(tid & 3);
    blk[16 * tile + l + 0] = w.a;
    blk[16 * tile + l + 4] = w.b;
    blk[16 * tile + l + 8] = w.c;
    blk[16 * tile + l + 12] = w.d;
}

inline uint index_alpha_mainnet_dd(uint slice, uint i, uint pseudo32) {
    uint area = slice * A2_SEGMENT_LENGTH + i - 1u;
    ulong sq = pseudo32;
    sq = (sq * sq) >> 32;
    return (uint)(area - 1u - ((ulong)area * sq >> 32));
}

inline void fill_block_regio(
    thread WarpWords& prev, WarpWords ref, device ulong* next,
    threadgroup ulong stage[4][32], uint tid
) {
    const int l = int(tid & 3);
    ulong a = prev.a ^ ref.a;
    ulong b = prev.b ^ ref.b;
    ulong c = prev.c ^ ref.c;
    ulong d = prev.d ^ ref.d;
    ulong t0 = a, t1 = b, t2 = c, t3 = d;

    blamka_tile_round_diag_out(a, b, c, d, l, tid);
    transpose_diag_shared(a, b, c, d, stage, tid);

    blamka_tile_round_diag_out(a, b, c, d, l, tid);
    transpose_diag_shared(a, b, c, d, stage, tid);

    prev.a = t0 ^ a;
    prev.b = t1 ^ b;
    prev.c = t2 ^ c;
    prev.d = t3 ^ d;
    store_warp_words(next, prev, tid);
}

inline void fill_segment_fast(
    device ulong* pool, uint local, uint H, uint slice,
    device const uint* di_table,
    threadgroup ulong stage[4][32], uint tid
) {
    const int data_independent = (slice < A2_SYNC_POINTS / 2);
    const uint starting_index = (slice == 0) ? 2u : 0u;
    uint curr_offset = slice * A2_SEGMENT_LENGTH + starting_index;
    uint prev_offset = curr_offset - 1;
    if (curr_offset % A2_LANE_LENGTH == 0) {
        prev_offset = curr_offset + A2_LANE_LENGTH - 1;
    }

    WarpWords prev = load_warp_words(block_ptr_c(pool, local, prev_offset, H), tid);

    if (data_independent) {
        WarpWords ref = load_warp_words(
            block_ptr_c(pool, local, di_table[slice * A2_SEGMENT_LENGTH + starting_index], H), tid);
        for (uint i = starting_index; i < A2_SEGMENT_LENGTH; ++i, ++curr_offset) {
            const int has_next = (i + 1 < A2_SEGMENT_LENGTH);
            WarpWords nref;
            if (has_next) {
                nref = load_warp_words(
                    block_ptr_c(pool, local, di_table[slice * A2_SEGMENT_LENGTH + (i + 1)], H), tid);
            }
            fill_block_regio(prev, ref, block_ptr(pool, local, curr_offset, H), stage, tid);
            if (has_next) ref = nref;
        }
        return;
    }

    // DD slices 2-3 with one-ahead reference prefetch before the second transpose & store.
    ulong pseudo = simd_broadcast_u64(prev.a, 0);
    uint ref_index = index_alpha_mainnet_dd(slice, starting_index, (uint)(pseudo & 0xFFFFFFFFUL));
    WarpWords ref = load_warp_words(block_ptr_c(pool, local, ref_index, H), tid);
    for (uint i = starting_index; i < A2_SEGMENT_LENGTH; ++i, ++curr_offset) {
        const int has_next = (i + 1 < A2_SEGMENT_LENGTH);
        WarpWords nref;
        const int l = int(tid & 3);
        ulong a = prev.a ^ ref.a;
        ulong b = prev.b ^ ref.b;
        ulong c = prev.c ^ ref.c;
        ulong d = prev.d ^ ref.d;
        ulong t0 = a, t1 = b, t2 = c, t3 = d;

        blamka_tile_round_diag_out(a, b, c, d, l, tid);
        transpose_diag_shared(a, b, c, d, stage, tid);

        blamka_tile_round_diag_out(a, b, c, d, l, tid);
        if (has_next) {
            ulong word0 = simd_broadcast_u64(t0 ^ a, 0);
            uint nri = index_alpha_mainnet_dd(slice, i + 1, (uint)(word0 & 0xFFFFFFFFUL));
            nref = load_warp_words(block_ptr_c(pool, local, nri, H), tid);
        }
        transpose_diag_shared(a, b, c, d, stage, tid);

        prev.a = t0 ^ a;
        prev.b = t1 ^ b;
        prev.c = t2 ^ c;
        prev.d = t3 ^ d;
        store_warp_words(block_ptr(pool, local, curr_offset, H), prev, tid);
        if (has_next) ref = nref;
    }
}

inline device ulong* select_pool(
    uint gid, uint chunk_hashes,
    device ulong* mem0, device ulong* mem1, device ulong* mem2, device ulong* mem3,
    device ulong* mem4, device ulong* mem5, device ulong* mem6, device ulong* mem7,
    device ulong* mem8, device ulong* mem9, device ulong* mem10, device ulong* mem11,
    thread uint& local_out
) {
    uint chunk = (chunk_hashes == 0) ? 0u : (gid / chunk_hashes);
    local_out = (chunk_hashes == 0) ? gid : (gid % chunk_hashes);
    if (chunk == 0) return mem0;
    else if (chunk == 1) return mem1;
    else if (chunk == 2) return mem2;
    else if (chunk == 3) return mem3;
    else if (chunk == 4) return mem4;
    else if (chunk == 5) return mem5;
    else if (chunk == 6) return mem6;
    else if (chunk == 7) return mem7;
    else if (chunk == 8) return mem8;
    else if (chunk == 9) return mem9;
    else if (chunk == 10) return mem10;
    return mem11;
}

inline void init_b0_b1_from_h0(device ulong* pool, uint local, uint H, thread uchar blockhash[72]) {
    uchar blockhash_bytes[1024];
    store32_t(blockhash + 64, 0);
    store32_t(blockhash + 68, 0);
    blake2b_long(blockhash_bytes, 1024, blockhash, 72);
    device ulong* b0 = block_ptr(pool, local, 0, H);
    for (uint i = 0; i < 128; ++i) b0[i] = load64_t(&blockhash_bytes[i * 8]);

    store32_t(blockhash + 64, 1);
    blake2b_long(blockhash_bytes, 1024, blockhash, 72);
    device ulong* b1 = block_ptr(pool, local, 1, H);
    for (uint i = 0; i < 128; ++i) b1[i] = load64_t(&blockhash_bytes[i * 8]);
}

inline void finalize_last_block(device ulong* pool, uint local, uint H, device uchar* digest) {
    device const ulong* blast = block_ptr_c(pool, local, A2_LANE_LENGTH - 1, H);
    uchar blockhash_bytes[1024];
    for (uint i = 0; i < 128; ++i) store64_t(&blockhash_bytes[i * 8], blast[i]);
    uchar out32[32];
    blake2b_long(out32, 32, blockhash_bytes, 1024);
    for (uint i = 0; i < 32; ++i) digest[i] = out32[i];
}

// General batch entry point: arbitrary password/salt/secret/AD pre-hashed to RFC 9106 H0 on host.
kernel void argon2id_batch_h0(
    device const uchar* h0_batch [[buffer(0)]],
    device uchar* digests [[buffer(1)]],
    device ulong* mem0 [[buffer(2)]],
    device ulong* mem1 [[buffer(3)]],
    device ulong* mem2 [[buffer(4)]],
    device ulong* mem3 [[buffer(5)]],
    device ulong* mem4 [[buffer(6)]],
    device ulong* mem5 [[buffer(7)]],
    device ulong* mem6 [[buffer(8)]],
    device ulong* mem7 [[buffer(9)]],
    device ulong* mem8 [[buffer(10)]],
    device ulong* mem9 [[buffer(11)]],
    device ulong* mem10 [[buffer(12)]],
    device ulong* mem11 [[buffer(13)]],
    constant uint& batch_size [[buffer(14)]],
    constant uint& chunk_hashes [[buffer(15)]],
    device const uint* di_table [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= batch_size) return;
    threadgroup ulong stage[4][32];

    uint local = 0;
    device ulong* pool = select_pool(
        gid, chunk_hashes,
        mem0, mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8, mem9, mem10, mem11, local);
    const uint H = (chunk_hashes == 0) ? 1u : chunk_hashes;

    if (tid == 0) {
        const device uchar* h0_in = h0_batch + gid * A2_H0_LEN;
        uchar blockhash[72];
        for (uint i = 0; i < A2_H0_LEN; ++i) blockhash[i] = h0_in[i];
        init_b0_b1_from_h0(pool, local, H, blockhash);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint s = 0; s < A2_SYNC_POINTS; ++s) {
        fill_segment_fast(pool, local, H, s, di_table, stage, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tid == 0) {
        finalize_last_block(pool, local, H, digests + gid * A2_OUT_LEN);
    }
}

// Indexed range kernel: computes RFC 9106 H0 directly on-device for pwd = LE64(start_index + gid) and salt = salt16.
kernel void argon2id_indexed_range(
    device const uchar* salt16 [[buffer(0)]],
    device uchar* digests [[buffer(1)]],
    device ulong* mem0 [[buffer(2)]],
    device ulong* mem1 [[buffer(3)]],
    device ulong* mem2 [[buffer(4)]],
    device ulong* mem3 [[buffer(5)]],
    device ulong* mem4 [[buffer(6)]],
    device ulong* mem5 [[buffer(7)]],
    device ulong* mem6 [[buffer(8)]],
    device ulong* mem7 [[buffer(9)]],
    device ulong* mem8 [[buffer(10)]],
    device ulong* mem9 [[buffer(11)]],
    device ulong* mem10 [[buffer(12)]],
    device ulong* mem11 [[buffer(13)]],
    constant uint& batch_size [[buffer(14)]],
    constant uint& chunk_hashes [[buffer(15)]],
    constant ulong& start_index [[buffer(16)]],
    device const uint* di_table [[buffer(17)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]]
) {
    if (gid >= batch_size) return;
    threadgroup ulong stage[4][32];

    uint local = 0;
    device ulong* pool = select_pool(
        gid, chunk_hashes,
        mem0, mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8, mem9, mem10, mem11, local);
    const uint H = (chunk_hashes == 0) ? 1u : chunk_hashes;

    if (tid == 0) {
        const ulong idx = start_index + (ulong)gid;
        uchar pwd[8];
        store64_t(pwd, idx);
        uchar salt[16];
        for (uint i = 0; i < 16; ++i) salt[i] = salt16[i];

        uchar blockhash[72];
        for (uint i = 0; i < 72; ++i) blockhash[i] = 0;

        Blake2bState S;
        blake2b_init(S, A2_H0_LEN);
        uchar value[4];
        store32_t(value, A2_LANES); blake2b_update(S, value, 4);
        store32_t(value, A2_OUT_LEN); blake2b_update(S, value, 4);
        store32_t(value, A2_MEMORY_BLOCKS); blake2b_update(S, value, 4);
        store32_t(value, A2_PASSES); blake2b_update(S, value, 4);
        store32_t(value, A2_VERSION); blake2b_update(S, value, 4);
        store32_t(value, A2_TYPE_ID); blake2b_update(S, value, 4);
        store32_t(value, 8); blake2b_update(S, value, 4);
        blake2b_update(S, pwd, 8);
        store32_t(value, 16); blake2b_update(S, value, 4);
        blake2b_update(S, salt, 16);
        store32_t(value, 0); blake2b_update(S, value, 4);
        store32_t(value, 0); blake2b_update(S, value, 4);
        blake2b_final(S, blockhash, A2_H0_LEN);

        init_b0_b1_from_h0(pool, local, H, blockhash);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint s = 0; s < A2_SYNC_POINTS; ++s) {
        fill_segment_fast(pool, local, H, s, di_table, stage, tid);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (tid == 0) {
        finalize_last_block(pool, local, H, digests + gid * A2_OUT_LEN);
    }
}
