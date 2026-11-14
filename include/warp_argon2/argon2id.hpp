#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace warp_argon2 {

// Fixed high-throughput Argon2id profile: 64 MiB, 1 pass, 1 lane, 32-byte tag (RFC 9106 v0x13).
inline constexpr uint32_t kMemCostKiB = 65536;
inline constexpr uint32_t kMemBlocks = 65536;
inline constexpr uint32_t kPasses = 1;
inline constexpr uint32_t kLanes = 1;
inline constexpr uint32_t kSyncPoints = 4;
inline constexpr uint32_t kSegmentLength = kMemBlocks / (kLanes * kSyncPoints);
inline constexpr uint32_t kBlockQwords = 128;
inline constexpr uint32_t kBlockBytes = 1024;
inline constexpr uint32_t kTagLen = 32;
inline constexpr uint32_t kVersion = 0x13;
inline constexpr uint32_t kTypeArgon2id = 2;
inline constexpr std::size_t kH0Bytes = 64;

// For t=1, lanes=1, Argon2id uses data-independent addressing in slices 0 and 1 (32,768 blocks).
inline constexpr std::size_t kDiTableBlocks = kMemBlocks / 2;

using Digest = std::array<uint8_t, kTagLen>;
using H0Block = std::array<uint8_t, kH0Bytes>;

struct InputView {
    std::span<const uint8_t> password{};
    std::span<const uint8_t> salt{};
    std::span<const uint8_t> secret{};
    std::span<const uint8_t> ad{};
};

std::string to_hex(std::span<const uint8_t> bytes);

// RFC 9106 Section 3.2 pre-hashing seed H0 for the fixed (m=65536, t=1, p=1, outlen=32, v=0x13, Argon2id) profile.
H0Block compute_h0(const InputView& in);

// Compute packed H0 blocks (64 bytes per item) for a batch of inputs.
void compute_h0_batch(std::span<const InputView> inputs, uint8_t* out_h0_flat);

// Host CPU Argon2id reference via vendored phc-winner-argon2.
bool cpu_argon2id(const InputView& in, Digest& out, std::string& err);

// General host CPU Argon2id helper (used by self-tests for arbitrary RFC 9106 parameters).
bool cpu_argon2id_custom(uint32_t t_cost, uint32_t m_cost, uint32_t parallelism,
                         const InputView& in, std::vector<uint8_t>& out, std::string& err);

// Precompute the 32,768 data-independent reference indices for slices 0..1 (matches RFC 9106 Section 3.4).
void build_di_ref_table(uint32_t* out_table_32k);

// Host CPU benchmark over sequential 8-byte little-endian inputs with a fixed 16-byte salt.
double cpu_hash_bench(double seconds, int threads = 1);

}  // namespace warp_argon2
