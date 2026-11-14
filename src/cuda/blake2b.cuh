#pragma once

#include <cstdint>

namespace warp_argon2 {
namespace cuda_detail {

__device__ __constant__ const uint64_t kBlake2bIV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

__device__ __constant__ const uint8_t kBlake2bSigma[12][16] = {
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

__device__ __forceinline__ uint64_t rotr64(uint64_t x, unsigned c) {
    return (x >> c) | (x << (64u - c));
}

__device__ __forceinline__ uint64_t load64(const uint8_t* p) {
    uint64_t v = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    return v;
}

__device__ __forceinline__ void store64(uint8_t* p, uint64_t v) {
#pragma unroll
    for (int i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>((v >> (8 * i)) & 0xffu);
}

__device__ __forceinline__ void store32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xffu);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}

struct Blake2bState {
    uint64_t h[8];
    uint64_t t[2];
    uint64_t f[2];
    uint8_t buf[128];
    unsigned buflen;
    unsigned outlen;
};

__device__ void blake2b_compress(Blake2bState* S, const uint8_t block[128]) {
    uint64_t m[16];
    uint64_t v[16];

#pragma unroll
    for (int i = 0; i < 16; ++i) m[i] = load64(block + i * 8);
#pragma unroll
    for (int i = 0; i < 8; ++i) v[i] = S->h[i];

    v[8] = kBlake2bIV[0];
    v[9] = kBlake2bIV[1];
    v[10] = kBlake2bIV[2];
    v[11] = kBlake2bIV[3];
    v[12] = kBlake2bIV[4] ^ S->t[0];
    v[13] = kBlake2bIV[5] ^ S->t[1];
    v[14] = kBlake2bIV[6] ^ S->f[0];
    v[15] = kBlake2bIV[7] ^ S->f[1];

#define G(r, i, a, b, c, d)                                       \
    do {                                                          \
        (a) = (a) + (b) + m[kBlake2bSigma[(r)][2 * (i)]];         \
        (d) = rotr64((d) ^ (a), 32);                              \
        (c) = (c) + (d);                                          \
        (b) = rotr64((b) ^ (c), 24);                              \
        (a) = (a) + (b) + m[kBlake2bSigma[(r)][2 * (i) + 1]];     \
        (d) = rotr64((d) ^ (a), 16);                              \
        (c) = (c) + (d);                                          \
        (b) = rotr64((b) ^ (c), 63);                              \
    } while (0)

    for (int r = 0; r < 12; ++r) {
        G(r, 0, v[0], v[4], v[8], v[12]);
        G(r, 1, v[1], v[5], v[9], v[13]);
        G(r, 2, v[2], v[6], v[10], v[14]);
        G(r, 3, v[3], v[7], v[11], v[15]);
        G(r, 4, v[0], v[5], v[10], v[15]);
        G(r, 5, v[1], v[6], v[11], v[12]);
        G(r, 6, v[2], v[7], v[8], v[13]);
        G(r, 7, v[3], v[4], v[9], v[14]);
    }
#undef G

#pragma unroll
    for (int i = 0; i < 8; ++i) S->h[i] = S->h[i] ^ v[i] ^ v[i + 8];
}

__device__ void blake2b_init(Blake2bState* S, size_t outlen) {
    for (int i = 0; i < 8; ++i) S->h[i] = kBlake2bIV[i];
    S->t[0] = 0;
    S->t[1] = 0;
    S->f[0] = 0;
    S->f[1] = 0;
    S->buflen = 0;
    S->outlen = static_cast<unsigned>(outlen);
    for (int i = 0; i < 128; ++i) S->buf[i] = 0;

    uint8_t param[64] = {};
    param[0] = static_cast<uint8_t>(outlen);
    param[2] = 1;  // fanout
    param[3] = 1;  // depth
    for (int i = 0; i < 8; ++i) S->h[i] ^= load64(param + i * 8);
}

__device__ void blake2b_update(Blake2bState* S, const uint8_t* in, size_t inlen) {
    if (inlen == 0) return;
    size_t left = S->buflen;
    size_t fill = 128u - left;
    if (inlen > fill) {
        S->buflen = 0;
        for (size_t i = 0; i < fill; ++i) S->buf[left + i] = in[i];
        S->t[0] += 128;
        if (S->t[0] < 128) S->t[1]++;
        blake2b_compress(S, S->buf);
        in += fill;
        inlen -= fill;
        while (inlen > 128) {
            S->t[0] += 128;
            if (S->t[0] < 128) S->t[1]++;
            blake2b_compress(S, in);
            in += 128;
            inlen -= 128;
        }
    }
    for (size_t i = 0; i < inlen; ++i) S->buf[S->buflen + i] = in[i];
    S->buflen += static_cast<unsigned>(inlen);
}

__device__ void blake2b_final(Blake2bState* S, uint8_t* out, size_t outlen) {
    uint8_t buffer[64];
    S->t[0] += S->buflen;
    if (S->t[0] < S->buflen) S->t[1]++;
    S->f[0] = UINT64_C(0xffffffffffffffff);
    for (unsigned i = S->buflen; i < 128; ++i) S->buf[i] = 0;
    blake2b_compress(S, S->buf);
    for (int i = 0; i < 8; ++i) store64(buffer + i * 8, S->h[i]);
    for (size_t i = 0; i < outlen; ++i) out[i] = buffer[i];
}

__device__ void blake2b_simple(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    Blake2bState S;
    blake2b_init(&S, outlen);
    blake2b_update(&S, in, inlen);
    blake2b_final(&S, out, outlen);
}

__device__ void blake2b_long(uint8_t* out, uint32_t outlen, const uint8_t* in, size_t inlen) {
    uint8_t outlen_bytes[4];
    store32(outlen_bytes, outlen);

    if (outlen <= 64) {
        Blake2bState S;
        blake2b_init(&S, outlen);
        blake2b_update(&S, outlen_bytes, sizeof(outlen_bytes));
        blake2b_update(&S, in, inlen);
        blake2b_final(&S, out, outlen);
        return;
    }

    uint8_t out_buffer[64];
    uint8_t in_buffer[64];
    Blake2bState S;
    blake2b_init(&S, 64);
    blake2b_update(&S, outlen_bytes, sizeof(outlen_bytes));
    blake2b_update(&S, in, inlen);
    blake2b_final(&S, out_buffer, 64);

    for (int i = 0; i < 32; ++i) out[i] = out_buffer[i];
    uint32_t toproduce = outlen - 32;
    size_t out_off = 32;

    while (toproduce > 64) {
        for (int i = 0; i < 64; ++i) in_buffer[i] = out_buffer[i];
        blake2b_simple(out_buffer, 64, in_buffer, 64);
        for (int i = 0; i < 32; ++i) out[out_off + i] = out_buffer[i];
        out_off += 32;
        toproduce -= 32;
    }

    for (int i = 0; i < 64; ++i) in_buffer[i] = out_buffer[i];
    blake2b_simple(out_buffer, toproduce, in_buffer, 64);
    for (uint32_t i = 0; i < toproduce; ++i) out[out_off + i] = out_buffer[i];
}

}  // namespace cuda_detail
}  // namespace warp_argon2
