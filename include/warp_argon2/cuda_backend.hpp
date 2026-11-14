#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "warp_argon2/argon2id.hpp"

namespace warp_argon2 {

struct CudaDeviceInfo {
    int index = 0;
    std::string name;
    int compute_major = 0;
    int compute_minor = 0;
    uint64_t total_mem_bytes = 0;
    bool sm89_native = false;
};

struct WaveStageSample {
    uint64_t hashes = 0;
    double h2d_ms = 0.0;
    double init_ms = 0.0;
    double fill_ms = 0.0;
    double final_ms = 0.0;
    double d2h_ms = 0.0;
    double total_ms = 0.0;
};

struct StageTimingReport {
    bool ok = false;
    double elapsed_sec = 0.0;
    uint64_t hashes = 0;
    double h_per_s = 0.0;
    double h2d_ms_sum = 0.0;
    double init_ms_sum = 0.0;
    double fill_ms_sum = 0.0;
    double final_ms_sum = 0.0;
    double d2h_ms_sum = 0.0;
    double wave_total_ms_sum = 0.0;
    std::vector<WaveStageSample> waves;
    std::string error;
};

struct AutotuneEntry {
    int concurrency = 0;
    int streams = 0;
    uint64_t hashes = 0;
    double sec = 0.0;
    double h_per_s = 0.0;
    double mem_mb = 0.0;
    bool ok = false;
    std::string err;
};

struct AutotuneResult {
    bool ok = false;
    int best_concurrency = 0;
    int best_streams = 0;
    double best_h_per_s = 0.0;
    uint64_t gpu_total_bytes = 0;
    std::vector<AutotuneEntry> trials;
    std::string error;
};

struct CudaBenchResult {
    bool ok = false;
    double elapsed_sec = 0.0;
    uint64_t hashes = 0;
    double hashes_per_sec = 0.0;
    std::string error;
};

class CudaBackend {
public:
    CudaBackend() = default;
    ~CudaBackend();

    CudaBackend(const CudaBackend&) = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    static std::vector<CudaDeviceInfo> enumerate_devices();

    bool init(int device_index, int concurrency, int n_streams, std::string& err);
    bool init_auto(int device_index, std::string& err);
    void free_buffers();

    bool is_ready() const { return ready_; }
    int concurrency() const;
    int streams() const { return n_streams_; }
    int device_index() const { return device_index_; }
    std::string device_name() const { return device_name_; }

    // Hash arbitrary inputs via RFC 9106 H0 blocks.
    bool hash_batch_h0(std::span<const H0Block> h0_batch,
                       std::vector<Digest>& out,
                       std::string& err);

    // Hash arbitrary inputs (computes H0 on host, dispatches to CUDA).
    bool hash_batch(std::span<const InputView> inputs,
                    std::vector<Digest>& out,
                    std::string& err);

    // Hash `count` inputs where password = LE64(start_index + i) and salt = salt16.
    // H0 is evaluated directly on-GPU inside argon2_init_kernel_range.
    bool hash_indexed_range(std::span<const uint8_t, 16> salt16,
                            uint64_t start_index,
                            int count,
                            std::vector<Digest>& out,
                            std::string& err);

    CudaBenchResult bench_throughput(double seconds);

    AutotuneResult autotune(
        int device_index,
        double trial_sec = 2.0,
        std::function<void(const AutotuneEntry&)> on_trial = {});

    StageTimingReport stage_timing_run(double seconds);

private:
    bool ensure_di_table(std::string& err);
    bool allocate_buffers(int concurrency, int n_streams, std::string& err);
    int probe_max_concurrency(int device_index, std::string& err);

    bool ready_ = false;
    int device_index_ = 0;
    int n_streams_ = 2;
    std::string device_name_;
    void* impl_ = nullptr;
};

}  // namespace warp_argon2
