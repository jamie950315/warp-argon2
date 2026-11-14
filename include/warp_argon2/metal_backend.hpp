#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "warp_argon2/argon2id.hpp"

namespace warp_argon2 {

struct MetalDeviceInfo {
    std::string name;
    uint64_t recommended_max_working_set = 0;
    uint64_t max_buffer_length = 0;
    bool has_unified_memory = false;
};

struct MetalBenchResult {
    bool ok = false;
    double elapsed_sec = 0.0;
    uint64_t hashes = 0;
    double hashes_per_sec = 0.0;
    std::string error;
};

class MetalBackend {
public:
    MetalBackend() = default;
    ~MetalBackend();

    MetalBackend(const MetalBackend&) = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    // Compile/load warp_argon2.metallib and allocate up to max_concurrency * 64 MiB workspace chunks.
    // Pass max_concurrency <= 0 to select the recommended default.
    bool init(int max_concurrency, std::string& err);
    void shutdown();

    bool ready() const { return ready_; }
    int concurrency() const { return concurrency_; }
    MetalDeviceInfo device_info() const { return dev_info_; }

    // Hash a batch of arbitrary inputs represented by precomputed RFC 9106 H0 blocks.
    bool hash_batch_h0(std::span<const H0Block> h0_batch,
                       std::vector<Digest>& out,
                       std::string& err);

    // Hash a batch of arbitrary inputs (computes H0 on host, runs Argon2id memory fill & finalize on Metal).
    bool hash_batch(std::span<const InputView> inputs,
                    std::vector<Digest>& out,
                    std::string& err);

    // Hash `count` inputs where password = LE64(start_index + i) and salt = salt16.
    // H0 is evaluated directly on-GPU in a single Blake2b-512 block compression.
    bool hash_indexed_range(std::span<const uint8_t, 16> salt16,
                            uint64_t start_index,
                            int count,
                            std::vector<Digest>& out,
                            std::string& err);

    MetalBenchResult bench_throughput(double seconds, int batch_size = 0);

    static int default_concurrency();
    static int max_supported_concurrency();

private:
    bool ensure_capacity(int needed, std::string& err);
    bool ensure_di_table(std::string& err);
    bool dispatch_h0_chunk(const uint8_t* h0_flat, int n, uint8_t* digests_out, std::string& err);

    bool ready_ = false;
    int concurrency_ = 0;
    MetalDeviceInfo dev_info_{};
    void* device_ = nullptr;
    void* queue_ = nullptr;
    void* pipeline_h0_ = nullptr;
    void* pipeline_range_ = nullptr;
    void* h0_buf_ = nullptr;
    void* salt_buf_ = nullptr;
    void* digests_buf_ = nullptr;
    void* params_buf_ = nullptr;
    void* di_table_buf_ = nullptr;
    std::vector<void*> mem_chunks_;
};

}  // namespace warp_argon2
