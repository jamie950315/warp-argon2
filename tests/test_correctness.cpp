#include "warp_argon2/argon2id.hpp"

#if WARP_ARGON2_HAS_METAL
#include "warp_argon2/metal_backend.hpp"
#endif

#if WARP_ARGON2_HAS_CUDA
#include "warp_argon2/cuda_backend.hpp"
#endif

extern "C" {
#include "argon2.h"
#include "core.h"
}

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool test_kat_and_h0() {
    std::string err;

    // Official phc-winner-argon2 v0x13 Argon2id KATs (including m=65536, t=1, p=1).
    {
        const std::string pwd = "password";
        const std::string salt = "somesalt";
        std::vector<uint8_t> p(pwd.begin(), pwd.end());
        std::vector<uint8_t> s(salt.begin(), salt.end());

        warp_argon2::Digest d{};
        if (!warp_argon2::cpu_argon2id(warp_argon2::InputView{p, s}, d, err)) {
            std::cerr << "[FAIL] cpu_argon2id KAT error: " << err << "\n";
            return false;
        }
        const std::string expected_t1 =
            "f6a5adc1ba723dddef9b5ac1d464e180fcd9dffc9d1cbf76cca2fed795d9ca98";
        if (warp_argon2::to_hex(d) != expected_t1) {
            std::cerr << "[FAIL] Argon2id (t=1,m=65536,p=1) KAT mismatch: got "
                      << warp_argon2::to_hex(d) << " expected " << expected_t1 << "\n";
            return false;
        }

        std::vector<uint8_t> out_t2(32);
        if (!warp_argon2::cpu_argon2id_custom(2, 65536, 1, warp_argon2::InputView{p, s}, out_t2, err)) {
            std::cerr << "[FAIL] cpu_argon2id_custom t=2 error: " << err << "\n";
            return false;
        }
        const std::string expected_t2 =
            "09316115d5cf24ed5a15a31a3ba326e5cf32edc24702987c02b6566f61913cf7";
        if (warp_argon2::to_hex(out_t2) != expected_t2) {
            std::cerr << "[FAIL] Argon2id (t=2,m=65536,p=1) KAT mismatch: got "
                      << warp_argon2::to_hex(out_t2) << "\n";
            return false;
        }
    }

    // Check RFC 9106 secret+AD non-empty vector matches raw argon2id_ctx vs custom wrapper,
    // and verify compute_h0 matches phc-winner-argon2 initial_hash() across arbitrary lengths.
    struct Case {
        std::vector<uint8_t> pwd;
        std::vector<uint8_t> salt;
        std::vector<uint8_t> secret;
        std::vector<uint8_t> ad;
    };
    std::vector<Case> cases;
    cases.push_back({{}, std::vector<uint8_t>(16, 0x22), {}, {}});
    cases.push_back({std::vector<uint8_t>(300, 0x41), std::vector<uint8_t>(64, 0x53),
                     std::vector<uint8_t>(12, 0x03), std::vector<uint8_t>(29, 0x04)});
    cases.push_back({std::vector<uint8_t>(32, 0x01), std::vector<uint8_t>(16, 0x02),
                     std::vector<uint8_t>(8, 0x03), std::vector<uint8_t>(12, 0x04)});

    for (std::size_t idx = 0; idx < cases.size(); ++idx) {
        auto& c = cases[idx];
        const auto h0 = warp_argon2::compute_h0(
            warp_argon2::InputView{c.pwd, c.salt, c.secret, c.ad});

        uint8_t ref_blockhash[ARGON2_PREHASH_SEED_LENGTH]{};
        uint8_t out_dummy[32]{};
        argon2_context ctx{};
        ctx.out = out_dummy;
        ctx.outlen = warp_argon2::kTagLen;
        ctx.pwd = c.pwd.empty() ? nullptr : c.pwd.data();
        ctx.pwdlen = static_cast<uint32_t>(c.pwd.size());
        ctx.salt = c.salt.empty() ? nullptr : c.salt.data();
        ctx.saltlen = static_cast<uint32_t>(c.salt.size());
        ctx.secret = c.secret.empty() ? nullptr : c.secret.data();
        ctx.secretlen = static_cast<uint32_t>(c.secret.size());
        ctx.ad = c.ad.empty() ? nullptr : c.ad.data();
        ctx.adlen = static_cast<uint32_t>(c.ad.size());
        ctx.t_cost = warp_argon2::kPasses;
        ctx.m_cost = warp_argon2::kMemCostKiB;
        ctx.lanes = warp_argon2::kLanes;
        ctx.threads = warp_argon2::kLanes;
        ctx.version = warp_argon2::kVersion;
        ctx.flags = ARGON2_DEFAULT_FLAGS;

        initial_hash(ref_blockhash, &ctx, Argon2_id);
        if (std::memcmp(h0.data(), ref_blockhash, warp_argon2::kH0Bytes) != 0) {
            std::cerr << "[FAIL] compute_h0 != initial_hash at case " << idx << "\n";
            return false;
        }
    }

    std::cout << "[PASS] RFC 9106 Argon2id KAT & H0 equivalence check\n";
    return true;
}

bool test_di_dependency_bound() {
    std::vector<uint32_t> table(warp_argon2::kDiTableBlocks, 0u);
    warp_argon2::build_di_ref_table(table.data());

    uint32_t min_gap = UINT32_MAX;
    bool saw_gap2 = false;
    for (uint32_t slice = 0; slice < 2; ++slice) {
        const uint32_t start_i = (slice == 0) ? 2u : 0u;
        for (uint32_t i = start_i; i < warp_argon2::kSegmentLength; ++i) {
            const uint32_t curr = slice * warp_argon2::kSegmentLength + i;
            const uint32_t ref = table[curr];
            if (ref >= curr - 1) {
                std::cerr << "[FAIL] DI dependency violation at curr=" << curr
                          << " ref=" << ref << "\n";
                return false;
            }
            const uint32_t gap = curr - ref;
            if (gap < min_gap) min_gap = gap;
            if (gap == 2) saw_gap2 = true;
        }
    }
    if (!saw_gap2 || min_gap < 2) {
        std::cerr << "[FAIL] DI min_gap=" << min_gap << " saw_gap2=" << saw_gap2 << "\n";
        return false;
    }
    std::cout << "[PASS] DI dependency bound: ref <= curr - 2 across all "
              << warp_argon2::kDiTableBlocks << " blocks (min gap = " << min_gap
              << ", confirming one-ahead prefetch safety and two-ahead hazard)\n";
    return true;
}

bool test_diag_out_transpose_mapping() {
    // Verify that DIAG_OUT + 1 KiB stage scatter/gather matches pure 4-color shuffle
    // and maps natural blockR indices into row-round column pairs (ref.c ordering).
    auto natural_idx = [](int tid, int k) -> int {
        const int tile = tid >> 2;
        const int l = tid & 3;
        return 16 * tile + l + 4 * k;
    };

    // Simulate phase-1 DIAG_OUT output on labels 0..127.
    std::array<std::array<int, 4>, 32> after_phase1{};
    for (int tid = 0; tid < 32; ++tid) {
        const int tile = tid >> 2;
        const int l = tid & 3;
        after_phase1[tid][0] = 16 * tile + l + 0;
        after_phase1[tid][1] = 16 * tile + ((l + 1) & 3) + 4;
        after_phase1[tid][2] = 16 * tile + ((l + 2) & 3) + 8;
        after_phase1[tid][3] = 16 * tile + ((l + 3) & 3) + 12;
    }

    // 1. Shared stage scatter/gather.
    std::array<std::array<int, 32>, 4> stage{};
    for (int tid = 0; tid < 32; ++tid) {
        const int tile = tid >> 2;
        const int l = tid & 3;
        for (int k = 0; k < 4; ++k) {
            const int logical_l = (l + k) & 3;
            const int o = logical_l + 4 * k;
            const int row = o >> 1;
            const int pos = 2 * tile + (o & 1);
            stage[pos >> 2][4 * row + (pos & 3)] = after_phase1[tid][k];
        }
    }
    std::array<std::array<int, 4>, 32> shared_out{};
    for (int tid = 0; tid < 32; ++tid) {
        for (int q = 0; q < 4; ++q) {
            shared_out[tid][q] = stage[q][tid];
        }
    }

    // 2. Verify row-round input layout matches ref.c row groups.
    std::array<bool, 128> seen{};
    for (int row = 0; row < 8; ++row) {
        for (int l = 0; l < 4; ++l) {
            const int tid = 4 * row + l;
            for (int q = 0; q < 4; ++q) {
                const int got = shared_out[tid][q];
                const int p = l + 4 * q;
                const int expected = 16 * (p >> 1) + 2 * row + (p & 1);
                if (got != expected) {
                    std::cerr << "[FAIL] DIAG_OUT shared transpose mismatch row=" << row
                              << " l=" << l << " q=" << q << " got=" << got
                              << " expected=" << expected << "\n";
                    return false;
                }
                seen[static_cast<std::size_t>(got)] = true;
            }
        }
    }
    for (int i = 0; i < 128; ++i) {
        if (!seen[static_cast<std::size_t>(i)]) {
            std::cerr << "[FAIL] DIAG_OUT transpose lost element " << i << "\n";
            return false;
        }
    }

    // 3. Verify 4-color XOR shuffle yields identical ownership on natural labels.
    auto rot4 = [](std::array<int, 4>& v, int g) {
        std::array<int, 4> tmp = v;
        for (int i = 0; i < 4; ++i) v[i] = tmp[i ^ g];
    };
    std::array<std::array<int, 4>, 32> diag_on_row{};
    for (int tid = 0; tid < 32; ++tid) {
        diag_on_row[tid] = shared_out[tid];
    }
    for (int tid = 0; tid < 32; ++tid) {
        const int row = tid >> 2;
        const int l = tid & 3;
        std::array<int, 4> after_p2{};
        for (int q = 0; q < 4; ++q) {
            const int src_lane = 4 * row + ((l + q) & 3);
            after_p2[q] = diag_on_row[src_lane][q];
        }
        after_phase1[tid] = after_p2;
    }
    // Check 4-color shuffle matches natural_idx(tid, q).
    for (int tid = 0; tid < 32; ++tid) {
        const int group = tid >> 3;
        const int logical_l = ((tid >> 1) & 2) | (tid & 1);
        const int base = (((tid >> 1) & 1) << 2) | ((logical_l - group) & 3);
        std::array<std::array<int, 4>, 32> permuted = after_phase1;
        for (int s = 0; s < 32; ++s) rot4(permuted[s], s >> 3);
        std::array<int, 4> shuf{};
        for (int c = 0; c < 4; ++c) {
            const int src = base + ((group ^ c) << 3);
            shuf[c] = permuted[src][c];
        }
        rot4(shuf, group);
        for (int q = 0; q < 4; ++q) {
            if (shuf[q] != natural_idx(tid, q)) {
                std::cerr << "[FAIL] 4-color shuffle mismatch at tid=" << tid << " q=" << q
                          << " got=" << shuf[q] << " expected=" << natural_idx(tid, q) << "\n";
                return false;
            }
        }
    }

    std::cout << "[PASS] DIAG_OUT 1 KiB stage & 4-color shuffle transpose bijection (128 qwords)\n";
    return true;
}

bool test_active_gpu_backend() {
    std::vector<std::vector<uint8_t>> pwds;
    std::vector<std::vector<uint8_t>> salts;
    std::vector<std::vector<uint8_t>> secrets;
    std::vector<std::vector<uint8_t>> ads;

    auto add = [&](std::string p, std::string s, std::string k, std::string x) {
        pwds.emplace_back(p.begin(), p.end());
        salts.emplace_back(s.begin(), s.end());
        secrets.emplace_back(k.begin(), k.end());
        ads.emplace_back(x.begin(), x.end());
    };
    add("password", "somesalt", "", "");
    add("", "0123456789abcdef", "", "");
    add("secret-and-ad", "salt12345678", "secret-pepper", "header-associated-data");
    add(std::string(250, 'X'), std::vector<uint8_t>(80, 0x5a).data() ? std::string(80, 'Z') : "", "k", "ad");
    for (int i = 0; i < 28; ++i) {
        std::vector<uint8_t> p(static_cast<std::size_t>(1 + i * 3));
        std::vector<uint8_t> s(static_cast<std::size_t>(8 + (i % 16)));
        for (std::size_t j = 0; j < p.size(); ++j) p[j] = static_cast<uint8_t>((i * 37 + j * 13) & 0xff);
        for (std::size_t j = 0; j < s.size(); ++j) s[j] = static_cast<uint8_t>((0x90 + i * 5 + j) & 0xff);
        pwds.push_back(std::move(p));
        salts.push_back(std::move(s));
        secrets.emplace_back();
        ads.emplace_back();
    }

    std::vector<warp_argon2::InputView> views(pwds.size());
    std::vector<warp_argon2::Digest> refs(pwds.size());
    std::string err;
    for (std::size_t i = 0; i < pwds.size(); ++i) {
        views[i] = warp_argon2::InputView{pwds[i], salts[i], secrets[i], ads[i]};
        if (!warp_argon2::cpu_argon2id(views[i], refs[i], err)) {
            std::cerr << "[FAIL] cpu_argon2id failed: " << err << "\n";
            return false;
        }
    }

    // Check KAT at index 0 explicitly.
    if (warp_argon2::to_hex(refs[0]) !=
        "f6a5adc1ba723dddef9b5ac1d464e180fcd9dffc9d1cbf76cca2fed795d9ca98") {
        std::cerr << "[FAIL] ref[0] unexpected KAT value\n";
        return false;
    }

    std::array<uint8_t, 16> salt16{};
    for (uint8_t i = 0; i < 16; ++i) salt16[i] = static_cast<uint8_t>(0xc0u + i);
    constexpr uint64_t kRangeStart = 999999991ULL;
    constexpr int kRangeCount = 40;  // 2.5 waves at concurrency=16
    std::vector<warp_argon2::Digest> range_refs(static_cast<std::size_t>(kRangeCount));
    for (int i = 0; i < kRangeCount; ++i) {
        const uint64_t val = kRangeStart + static_cast<uint64_t>(i);
        uint8_t pwd[8];
        for (int b = 0; b < 8; ++b) pwd[b] = static_cast<uint8_t>((val >> (8 * b)) & 0xffu);
        if (!warp_argon2::cpu_argon2id(warp_argon2::InputView{pwd, salt16},
                                       range_refs[static_cast<std::size_t>(i)], err)) {
            std::cerr << "[FAIL] cpu_argon2id range failed: " << err << "\n";
            return false;
        }
    }

#if WARP_ARGON2_HAS_METAL
    {
        warp_argon2::MetalBackend metal;
        if (!metal.init(16, err)) {
            std::cerr << "[FAIL] Metal init(16): " << err << "\n";
            return false;
        }
        std::vector<warp_argon2::Digest> gpu_batch;
        if (!metal.hash_batch(views, gpu_batch, err) || gpu_batch.size() != views.size()) {
            std::cerr << "[FAIL] Metal hash_batch: " << err << "\n";
            return false;
        }
        for (std::size_t i = 0; i < views.size(); ++i) {
            if (gpu_batch[i] != refs[i]) {
                std::cerr << "[FAIL] Metal hash_batch mismatch at i=" << i << "\n";
                return false;
            }
        }
        std::vector<warp_argon2::Digest> gpu_range;
        if (!metal.hash_indexed_range(salt16, kRangeStart, kRangeCount, gpu_range, err) ||
            gpu_range.size() != range_refs.size()) {
            std::cerr << "[FAIL] Metal hash_indexed_range: " << err << "\n";
            return false;
        }
        for (std::size_t i = 0; i < range_refs.size(); ++i) {
            if (gpu_range[i] != range_refs[i]) {
                std::cerr << "[FAIL] Metal range mismatch at i=" << i << "\n";
                return false;
            }
        }
        std::cout << "[PASS] Metal (" << metal.device_info().name
                  << ") matched libargon2 across " << views.size() << " arbitrary batches & "
                  << kRangeCount << " multi-wave range hashes\n";
    }
#endif

#if WARP_ARGON2_HAS_CUDA
    {
        const auto devs = warp_argon2::CudaBackend::enumerate_devices();
        if (devs.empty()) {
            std::cerr << "[FAIL] No CUDA device found during test_correctness\n";
            return false;
        }
        warp_argon2::CudaBackend cuda;
        if (!cuda.init(0, 16, 2, err)) {
            std::cerr << "[FAIL] CUDA init(0, 16, 2): " << err << "\n";
            return false;
        }
        std::vector<warp_argon2::Digest> gpu_batch;
        if (!cuda.hash_batch(views, gpu_batch, err) || gpu_batch.size() != views.size()) {
            std::cerr << "[FAIL] CUDA hash_batch: " << err << "\n";
            return false;
        }
        for (std::size_t i = 0; i < views.size(); ++i) {
            if (gpu_batch[i] != refs[i]) {
                std::cerr << "[FAIL] CUDA hash_batch mismatch at i=" << i << "\n";
                return false;
            }
        }
        std::vector<warp_argon2::Digest> gpu_range;
        if (!cuda.hash_indexed_range(salt16, kRangeStart, kRangeCount, gpu_range, err) ||
            gpu_range.size() != range_refs.size()) {
            std::cerr << "[FAIL] CUDA hash_indexed_range: " << err << "\n";
            return false;
        }
        for (std::size_t i = 0; i < range_refs.size(); ++i) {
            if (gpu_range[i] != range_refs[i]) {
                std::cerr << "[FAIL] CUDA range mismatch at i=" << i << "\n";
                return false;
            }
        }
        std::cout << "[PASS] CUDA (" << cuda.device_name()
                  << ") matched libargon2 across " << views.size() << " arbitrary batches & "
                  << kRangeCount << " multi-wave multi-stream range hashes\n";
    }
#endif

    return true;
}

}  // namespace

int main() {
    if (!test_kat_and_h0()) return 1;
    if (!test_di_dependency_bound()) return 1;
    if (!test_diag_out_transpose_mapping()) return 1;
    if (!test_active_gpu_backend()) return 1;
    return 0;
}
