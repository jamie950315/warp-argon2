#include "warp_argon2/argon2id.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

extern "C" {
#include "argon2.h"
#include "blake2/blake2-impl.h"
#include "blake2/blake2.h"
}

namespace warp_argon2 {

std::string to_hex(std::span<const uint8_t> bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string s(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        s[2 * i] = kHex[(bytes[i] >> 4) & 0x0f];
        s[2 * i + 1] = kHex[bytes[i] & 0x0f];
    }
    return s;
}

H0Block compute_h0(const InputView& in) {
    H0Block h0{};
    blake2b_state S;
    blake2b_init(&S, kH0Bytes);

    uint8_t le[4];
    auto put_u32 = [&](uint32_t v) {
        store32(le, v);
        blake2b_update(&S, le, sizeof(le));
    };

    put_u32(kLanes);
    put_u32(kTagLen);
    put_u32(kMemCostKiB);
    put_u32(kPasses);
    put_u32(kVersion);
    put_u32(kTypeArgon2id);

    put_u32(static_cast<uint32_t>(in.password.size()));
    if (!in.password.empty()) {
        blake2b_update(&S, in.password.data(), in.password.size());
    }

    put_u32(static_cast<uint32_t>(in.salt.size()));
    if (!in.salt.empty()) {
        blake2b_update(&S, in.salt.data(), in.salt.size());
    }

    put_u32(static_cast<uint32_t>(in.secret.size()));
    if (!in.secret.empty()) {
        blake2b_update(&S, in.secret.data(), in.secret.size());
    }

    put_u32(static_cast<uint32_t>(in.ad.size()));
    if (!in.ad.empty()) {
        blake2b_update(&S, in.ad.data(), in.ad.size());
    }

    blake2b_final(&S, h0.data(), h0.size());
    return h0;
}

void compute_h0_batch(std::span<const InputView> inputs, uint8_t* out_h0_flat) {
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const H0Block h0 = compute_h0(inputs[i]);
        std::memcpy(out_h0_flat + i * kH0Bytes, h0.data(), kH0Bytes);
    }
}

bool cpu_argon2id_custom(uint32_t t_cost, uint32_t m_cost, uint32_t parallelism,
                         const InputView& in, std::vector<uint8_t>& out, std::string& err) {
    if (out.empty()) {
        err = "cpu_argon2id_custom: output length must be non-empty";
        return false;
    }

    // Copy into temporary buffers because argon2_context takes non-const pointers.
    std::vector<uint8_t> pwd_buf(in.password.begin(), in.password.end());
    std::vector<uint8_t> salt_buf(in.salt.begin(), in.salt.end());
    std::vector<uint8_t> secret_buf(in.secret.begin(), in.secret.end());
    std::vector<uint8_t> ad_buf(in.ad.begin(), in.ad.end());

    argon2_context ctx{};
    ctx.out = out.data();
    ctx.outlen = static_cast<uint32_t>(out.size());
    ctx.pwd = pwd_buf.empty() ? nullptr : pwd_buf.data();
    ctx.pwdlen = static_cast<uint32_t>(pwd_buf.size());
    ctx.salt = salt_buf.empty() ? nullptr : salt_buf.data();
    ctx.saltlen = static_cast<uint32_t>(salt_buf.size());
    ctx.secret = secret_buf.empty() ? nullptr : secret_buf.data();
    ctx.secretlen = static_cast<uint32_t>(secret_buf.size());
    ctx.ad = ad_buf.empty() ? nullptr : ad_buf.data();
    ctx.adlen = static_cast<uint32_t>(ad_buf.size());
    ctx.t_cost = t_cost;
    ctx.m_cost = m_cost;
    ctx.lanes = parallelism;
    ctx.threads = parallelism;
    ctx.version = kVersion;
    ctx.flags = ARGON2_DEFAULT_FLAGS;

    const int rc = argon2id_ctx(&ctx);
    if (rc != ARGON2_OK) {
        err = argon2_error_message(rc);
        return false;
    }
    return true;
}

bool cpu_argon2id(const InputView& in, Digest& out, std::string& err) {
    std::vector<uint8_t> tmp(kTagLen);
    if (!cpu_argon2id_custom(kPasses, kMemCostKiB, kLanes, in, tmp, err)) {
        return false;
    }
    std::memcpy(out.data(), tmp.data(), kTagLen);
    return true;
}

namespace {

inline uint64_t host_rotr64(uint64_t x, unsigned n) {
    return (x >> n) | (x << (64u - n));
}

inline uint64_t host_fBlaMka(uint64_t x, uint64_t y) {
    const uint64_t m = 0xFFFFFFFFull;
    return x + y + 2ull * ((x & m) * (y & m));
}

inline void host_G(uint64_t& a, uint64_t& b, uint64_t& c, uint64_t& d) {
    a = host_fBlaMka(a, b);
    d = host_rotr64(d ^ a, 32u);
    c = host_fBlaMka(c, d);
    b = host_rotr64(b ^ c, 24u);
    a = host_fBlaMka(a, b);
    d = host_rotr64(d ^ a, 16u);
    c = host_fBlaMka(c, d);
    b = host_rotr64(b ^ c, 63u);
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

inline void host_fill_zero_prev(const uint64_t ref[128], uint64_t next[128]) {
    uint64_t R[128], T[128];
    for (int i = 0; i < 128; ++i) {
        R[i] = ref[i];
        T[i] = ref[i];
    }
    for (int i = 0; i < 8; ++i) {
        host_blake2_round_16(R + 16 * i);
    }
    for (int i = 0; i < 8; ++i) {
        uint64_t v[16] = {
            R[2 * i],       R[2 * i + 1],   R[2 * i + 16],  R[2 * i + 17],
            R[2 * i + 32],  R[2 * i + 33],  R[2 * i + 48],  R[2 * i + 49],
            R[2 * i + 64],  R[2 * i + 65],  R[2 * i + 80],  R[2 * i + 81],
            R[2 * i + 96],  R[2 * i + 97],  R[2 * i + 112], R[2 * i + 113],
        };
        host_blake2_round_16(v);
        R[2 * i] = v[0];
        R[2 * i + 1] = v[1];
        R[2 * i + 16] = v[2];
        R[2 * i + 17] = v[3];
        R[2 * i + 32] = v[4];
        R[2 * i + 33] = v[5];
        R[2 * i + 48] = v[6];
        R[2 * i + 49] = v[7];
        R[2 * i + 64] = v[8];
        R[2 * i + 65] = v[9];
        R[2 * i + 80] = v[10];
        R[2 * i + 81] = v[11];
        R[2 * i + 96] = v[12];
        R[2 * i + 97] = v[13];
        R[2 * i + 112] = v[14];
        R[2 * i + 113] = v[15];
    }
    for (int i = 0; i < 128; ++i) {
        next[i] = T[i] ^ R[i];
    }
}

inline uint32_t host_index_alpha(uint32_t pass, uint32_t slice, uint32_t index,
                                 uint32_t pseudo_rand) {
    uint32_t reference_area_size;
    if (pass == 0) {
        if (slice == 0) {
            reference_area_size = index - 1;
        } else {
            reference_area_size = slice * kSegmentLength + index - 1;
        }
    } else {
        reference_area_size = kMemBlocks - kSegmentLength + index - 1;
    }
    uint64_t relative_position = pseudo_rand;
    relative_position = (relative_position * relative_position) >> 32;
    relative_position =
        reference_area_size - 1 -
        ((static_cast<uint64_t>(reference_area_size) * relative_position) >> 32);
    uint32_t start_position = 0;
    if (pass != 0) {
        start_position = (slice == 3) ? 0 : (slice + 1) * kSegmentLength;
    }
    return static_cast<uint32_t>((start_position + relative_position) % kMemBlocks);
}

}  // namespace

void build_di_ref_table(uint32_t* out_table_32k) {
    std::memset(out_table_32k, 0, kDiTableBlocks * sizeof(uint32_t));
    for (uint32_t slice = 0; slice < 2; ++slice) {
        uint64_t input_block[128] = {};
        uint64_t address_block[128] = {};
        input_block[0] = 0;             // pass
        input_block[1] = 0;             // lane
        input_block[2] = slice;         // slice
        input_block[3] = kMemBlocks;    // memory blocks
        input_block[4] = kPasses;       // passes
        input_block[5] = kTypeArgon2id; // type Argon2id
        const uint32_t starting = (slice == 0) ? 2u : 0u;
        if (slice == 0) {
            input_block[6]++;
            host_fill_zero_prev(input_block, address_block);
            host_fill_zero_prev(address_block, address_block);
        }
        for (uint32_t i = starting; i < kSegmentLength; ++i) {
            if (i % 128u == 0) {
                input_block[6]++;
                host_fill_zero_prev(input_block, address_block);
                host_fill_zero_prev(address_block, address_block);
            }
            const uint32_t pseudo =
                static_cast<uint32_t>(address_block[i % 128u] & 0xFFFFFFFFull);
            out_table_32k[slice * kSegmentLength + i] = host_index_alpha(0, slice, i, pseudo);
        }
    }
}

double cpu_hash_bench(double seconds, int threads) {
    if (threads < 1) threads = 1;
    std::array<uint8_t, 16> salt{};
    for (uint8_t i = 0; i < 16; ++i) salt[i] = static_cast<uint8_t>(0x30u + i);

    std::atomic<uint64_t> total_hashes{0};
    std::atomic<bool> stop{false};

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    const auto t0 = std::chrono::steady_clock::now();

    for (int t = 0; t < threads; ++t) {
        workers.emplace_back([&, t]() {
            uint64_t counter = static_cast<uint64_t>(t) << 40;
            uint8_t pwd[8];
            uint8_t out[kTagLen];
            while (!stop.load(std::memory_order_relaxed)) {
                for (int b = 0; b < 8; ++b) {
                    pwd[b] = static_cast<uint8_t>((counter >> (8 * b)) & 0xffu);
                }
                argon2id_hash_raw(kPasses, kMemCostKiB, kLanes,
                                  pwd, sizeof(pwd), salt.data(), salt.size(),
                                  out, sizeof(out));
                counter++;
                total_hashes.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::this_thread::sleep_for(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::duration<double>(seconds)));
    stop.store(true, std::memory_order_relaxed);
    for (auto& w : workers) w.join();

    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return elapsed > 0.0 ? static_cast<double>(total_hashes.load()) / elapsed : 0.0;
}

}  // namespace warp_argon2
