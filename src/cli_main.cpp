#include "warp_argon2/argon2id.hpp"

#if WARP_ARGON2_HAS_METAL
#include "warp_argon2/metal_backend.hpp"
#endif

#if WARP_ARGON2_HAS_CUDA
#include "warp_argon2/cuda_backend.hpp"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

enum class BackendKind { Auto, Metal, Cuda, Cpu };

void print_usage() {
    std::cout << "warp-argon2 (RFC 9106 Argon2id: m=65536 KiB, t=1, p=1, outlen=32)\n\n"
              << "Usage:\n"
              << "  warp-argon2 --self-test [--backend auto|metal|cuda|cpu]\n"
              << "  warp-argon2 --bench [--backend auto|metal|cuda|cpu] [--seconds 5] [--concurrency N]\n"
              << "  warp-argon2 --password STR --salt STR [--backend auto|metal|cuda|cpu]\n"
              << "  warp-argon2 --autotune [--gpu 0] [--seconds 2]\n"
              << "  warp-argon2 --stage-timing [--gpu 0] [--concurrency N] [--seconds 3]\n\n"
              << "Options:\n"
              << "  --backend NAME      Backend: auto (default), metal, cuda, or cpu\n"
              << "  --self-test         Verify RFC 9106 KAT, H0 batch, and range kernels vs libargon2\n"
              << "  --bench             Run throughput benchmark\n"
              << "  --autotune          Sweep CUDA concurrency values for peak H/s\n"
              << "  --stage-timing      Measure CUDA H2D / Init / Fill / Final / D2H stage breakdown\n"
              << "  --password STR      Hash a password string (UTF-8 bytes)\n"
              << "  --salt STR          Salt string (UTF-8 bytes)\n"
              << "  --concurrency N     GPU worker concurrency (64 MiB per worker)\n"
              << "  --streams N         CUDA stream count (1..4, default 2)\n"
              << "  --seconds S         Benchmark / autotune trial duration (default 5.0)\n"
              << "  --cpu-threads N     Host CPU benchmark threads (default 1)\n"
              << "  --gpu IDX           CUDA device index (default 0)\n"
              << "  --help              Show this help\n";
}

BackendKind resolve_backend(BackendKind req) {
    if (req != BackendKind::Auto) return req;
#if WARP_ARGON2_HAS_METAL
    return BackendKind::Metal;
#elif WARP_ARGON2_HAS_CUDA
    return BackendKind::Cuda;
#else
    return BackendKind::Cpu;
#endif
}

bool verify_backend_against_cpu(const std::string& label,
                                auto&& run_batch,
                                auto&& run_range) {
    std::vector<std::vector<uint8_t>> pwds;
    std::vector<std::vector<uint8_t>> salts;
    std::vector<std::vector<uint8_t>> secrets;
    std::vector<std::vector<uint8_t>> ads;

    auto push_case = [&](std::string p, std::string s, std::string k, std::string x) {
        pwds.emplace_back(p.begin(), p.end());
        salts.emplace_back(s.begin(), s.end());
        secrets.emplace_back(k.begin(), k.end());
        ads.emplace_back(x.begin(), x.end());
    };

    push_case("", "somesalt1234", "", "");
    push_case("correct horse battery staple", "warp-argon2-salt", "", "");
    push_case("with-secret-and-ad", "salt-value-16byt", "pepper-key", "associated-meta");
    push_case(std::string(240, 'A'), std::string(48, 'S'), "k1234567", "ad1234567890");
    for (int i = 0; i < 12; ++i) {
        std::vector<uint8_t> p(static_cast<std::size_t>(12 + i));
        std::vector<uint8_t> s(16);
        for (std::size_t j = 0; j < p.size(); ++j) p[j] = static_cast<uint8_t>(i * 19 + j * 7);
        for (std::size_t j = 0; j < s.size(); ++j) s[j] = static_cast<uint8_t>(0xa0u + i + j);
        pwds.push_back(std::move(p));
        salts.push_back(std::move(s));
        secrets.emplace_back();
        ads.emplace_back();
    }

    std::vector<warp_argon2::InputView> views(pwds.size());
    for (std::size_t i = 0; i < pwds.size(); ++i) {
        views[i] = warp_argon2::InputView{pwds[i], salts[i], secrets[i], ads[i]};
    }

    std::vector<warp_argon2::Digest> gpu_batch;
    std::string err;
    if (!run_batch(views, gpu_batch, err)) {
        std::cerr << "[FAIL] " << label << " hash_batch error: " << err << "\n";
        return false;
    }
    if (gpu_batch.size() != views.size()) {
        std::cerr << "[FAIL] " << label << " batch digest count mismatch\n";
        return false;
    }

    for (std::size_t i = 0; i < views.size(); ++i) {
        warp_argon2::Digest ref{};
        if (!warp_argon2::cpu_argon2id(views[i], ref, err)) {
            std::cerr << "[FAIL] cpu_argon2id error: " << err << "\n";
            return false;
        }
        if (gpu_batch[i] != ref) {
            std::cerr << "[FAIL] " << label << " hash_batch mismatch at index " << i
                      << "\n  gpu=" << warp_argon2::to_hex(gpu_batch[i])
                      << "\n  cpu=" << warp_argon2::to_hex(ref) << "\n";
            return false;
        }
    }
    std::cout << "[PASS] " << label << " hash_batch matched libargon2 (" << views.size() << "/"
              << views.size() << " arbitrary pwd/salt/secret/AD vectors)\n";

    std::array<uint8_t, 16> salt16{};
    for (uint8_t i = 0; i < 16; ++i) salt16[i] = static_cast<uint8_t>(0x40u + i * 3u);
    constexpr uint64_t kStartIdx = 0x0102030405060700ULL;
    constexpr int kRangeCount = 24;

    std::vector<warp_argon2::Digest> gpu_range;
    if (!run_range(salt16, kStartIdx, kRangeCount, gpu_range, err)) {
        std::cerr << "[FAIL] " << label << " hash_indexed_range error: " << err << "\n";
        return false;
    }
    if (gpu_range.size() != static_cast<std::size_t>(kRangeCount)) {
        std::cerr << "[FAIL] " << label << " range digest count mismatch\n";
        return false;
    }

    for (int i = 0; i < kRangeCount; ++i) {
        const uint64_t v = kStartIdx + static_cast<uint64_t>(i);
        uint8_t pwd[8];
        for (int b = 0; b < 8; ++b) pwd[b] = static_cast<uint8_t>((v >> (8 * b)) & 0xffu);
        warp_argon2::Digest ref{};
        if (!warp_argon2::cpu_argon2id(warp_argon2::InputView{pwd, salt16}, ref, err)) {
            std::cerr << "[FAIL] cpu_argon2id range error: " << err << "\n";
            return false;
        }
        if (gpu_range[static_cast<std::size_t>(i)] != ref) {
            std::cerr << "[FAIL] " << label << " range mismatch at index " << i
                      << "\n  gpu=" << warp_argon2::to_hex(gpu_range[static_cast<std::size_t>(i)])
                      << "\n  cpu=" << warp_argon2::to_hex(ref) << "\n";
            return false;
        }
    }
    std::cout << "[PASS] " << label << " hash_indexed_range matched libargon2 (" << kRangeCount
              << "/" << kRangeCount << " multi-wave outputs)\n";
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    BackendKind req_backend = BackendKind::Auto;
    bool self_test = false;
    bool bench = false;
    bool autotune = false;
    bool stage_timing = false;
    bool has_password = false;
    std::string password;
    std::string salt = "warp-argon2-default-salt";
    int concurrency = 0;
    int streams = 2;
    double seconds = 5.0;
    int cpu_threads = 1;
    int gpu_index = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_val = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << flag << "\n";
                std::exit(2);
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {
            print_usage();
            return 0;
        } else if (arg == "--self-test") {
            self_test = true;
        } else if (arg == "--bench") {
            bench = true;
        } else if (arg == "--autotune") {
            autotune = true;
        } else if (arg == "--stage-timing") {
            stage_timing = true;
        } else if (arg == "--password") {
            password = need_val("--password");
            has_password = true;
        } else if (arg == "--salt") {
            salt = need_val("--salt");
        } else if (arg == "--concurrency") {
            concurrency = std::atoi(need_val("--concurrency"));
        } else if (arg == "--streams") {
            streams = std::atoi(need_val("--streams"));
        } else if (arg == "--seconds") {
            seconds = std::atof(need_val("--seconds"));
        } else if (arg == "--cpu-threads") {
            cpu_threads = std::atoi(need_val("--cpu-threads"));
        } else if (arg == "--gpu") {
            gpu_index = std::atoi(need_val("--gpu"));
        } else if (arg == "--backend") {
            const std::string val = need_val("--backend");
            if (val == "auto") req_backend = BackendKind::Auto;
            else if (val == "metal") req_backend = BackendKind::Metal;
            else if (val == "cuda") req_backend = BackendKind::Cuda;
            else if (val == "cpu") req_backend = BackendKind::Cpu;
            else {
                std::cerr << "Unknown backend: " << val << "\n";
                return 2;
            }
        } else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage();
            return 2;
        }
    }

    if (!self_test && !bench && !autotune && !stage_timing && !has_password) {
        print_usage();
        return 0;
    }

    const BackendKind backend = resolve_backend(req_backend);

    if (autotune || stage_timing) {
#if WARP_ARGON2_HAS_CUDA
        warp_argon2::CudaBackend cuda;
        const auto devs = warp_argon2::CudaBackend::enumerate_devices();
        if (devs.empty()) {
            std::cerr << "No CUDA GPU found\n";
            return 1;
        }
        const auto& dev = devs[static_cast<std::size_t>(std::clamp(gpu_index, 0, static_cast<int>(devs.size()) - 1))];
        std::cout << "CUDA device [" << dev.index << "]: " << dev.name
                  << " (SM " << dev.compute_major << "." << dev.compute_minor
                  << ", " << (dev.total_mem_bytes / (1024 * 1024)) << " MiB)\n";
        if (autotune) {
            const double trial_s = (seconds == 5.0) ? 2.0 : std::max(0.5, seconds);
            auto res = cuda.autotune(dev.index, trial_s, [](const warp_argon2::AutotuneEntry& e) {
                if (e.ok) {
                    std::cout << "  c=" << std::setw(3) << e.concurrency
                              << " (" << std::fixed << std::setprecision(0) << e.mem_mb << " MiB) -> "
                              << std::setprecision(2) << e.h_per_s << " H/s\n";
                } else {
                    std::cout << "  c=" << std::setw(3) << e.concurrency
                              << " FAIL: " << e.err << "\n";
                }
            });
            if (!res.ok) {
                std::cerr << "Autotune failed: " << res.error << "\n";
                return 1;
            }
            std::cout << "Best CUDA concurrency: c=" << res.best_concurrency
                      << " streams=" << res.best_streams
                      << " -> " << std::fixed << std::setprecision(2) << res.best_h_per_s << " H/s\n";
        }
        if (stage_timing) {
            std::string err;
            if (!cuda.init(dev.index, concurrency, streams, err)) {
                std::cerr << "CUDA init failed: " << err << "\n";
                return 1;
            }
            const double run_s = (seconds == 5.0) ? 3.0 : std::max(0.5, seconds);
            auto rpt = cuda.stage_timing_run(run_s);
            if (!rpt.ok) {
                std::cerr << "Stage timing failed: " << rpt.error << "\n";
                return 1;
            }
            const double nw = static_cast<double>(rpt.waves.size());
            std::cout << std::fixed << std::setprecision(2)
                      << "CUDA stage timing (c=" << cuda.concurrency()
                      << ", waves=" << rpt.waves.size()
                      << ", " << rpt.h_per_s << " H/s):\n"
                      << "  avg H2D  : " << (rpt.h2d_ms_sum / nw) << " ms\n"
                      << "  avg Init : " << (rpt.init_ms_sum / nw) << " ms\n"
                      << "  avg Fill : " << (rpt.fill_ms_sum / nw) << " ms\n"
                      << "  avg Final: " << (rpt.final_ms_sum / nw) << " ms\n"
                      << "  avg D2H  : " << (rpt.d2h_ms_sum / nw) << " ms\n"
                      << "  avg Total: " << (rpt.wave_total_ms_sum / nw) << " ms\n";
        }
        return 0;
#else
        std::cerr << "--autotune and --stage-timing require a CUDA-enabled build\n";
        return 1;
#endif
    }

    if (backend == BackendKind::Metal) {
#if WARP_ARGON2_HAS_METAL
        warp_argon2::MetalBackend metal;
        std::string err;
        const int test_c = (concurrency > 0) ? concurrency : (self_test && !bench ? 16 : 0);
        if (!metal.init(test_c, err)) {
            std::cerr << "Metal init failed: " << err << "\n";
            return 1;
        }
        const auto info = metal.device_info();
        std::cout << "Metal device: " << info.name
                  << " (concurrency=" << metal.concurrency()
                  << ", UMA=" << (info.has_unified_memory ? "yes" : "no") << ")\n";

        if (has_password) {
            std::vector<uint8_t> p(password.begin(), password.end());
            std::vector<uint8_t> s(salt.begin(), salt.end());
            std::vector<warp_argon2::InputView> one{warp_argon2::InputView{p, s}};
            std::vector<warp_argon2::Digest> out;
            if (!metal.hash_batch(one, out, err)) {
                std::cerr << "Metal hash_batch failed: " << err << "\n";
                return 1;
            }
            std::cout << warp_argon2::to_hex(out[0]) << "\n";
        }

        if (self_test) {
            const bool ok = verify_backend_against_cpu(
                "Metal",
                [&](std::span<const warp_argon2::InputView> in,
                    std::vector<warp_argon2::Digest>& out, std::string& e) {
                    return metal.hash_batch(in, out, e);
                },
                [&](std::span<const uint8_t, 16> s16, uint64_t start, int count,
                    std::vector<warp_argon2::Digest>& out, std::string& e) {
                    return metal.hash_indexed_range(s16, start, count, out, e);
                });
            if (!ok) return 1;
        }

        if (bench) {
            if (concurrency <= 0 && self_test) {
                if (!metal.init(0, err)) {
                    std::cerr << "Metal re-init for bench failed: " << err << "\n";
                    return 1;
                }
            }
            const auto br = metal.bench_throughput(seconds);
            if (!br.ok) {
                std::cerr << "Metal benchmark failed: " << br.error << "\n";
                return 1;
            }
            std::cout << std::fixed << std::setprecision(2)
                      << "Metal throughput: " << br.hashes_per_sec << " H/s ("
                      << br.hashes << " hashes in " << br.elapsed_sec << " s, concurrency="
                      << metal.concurrency() << ")\n";
        }
        return 0;
#else
        std::cerr << "Metal backend not compiled on this platform\n";
        return 1;
#endif
    }

    if (backend == BackendKind::Cuda) {
#if WARP_ARGON2_HAS_CUDA
        warp_argon2::CudaBackend cuda;
        const auto devs = warp_argon2::CudaBackend::enumerate_devices();
        if (devs.empty()) {
            std::cerr << "No CUDA GPU found\n";
            return 1;
        }
        std::string err;
        const int init_c = (concurrency > 0) ? concurrency : (self_test && !bench ? 16 : 0);
        if (!cuda.init(gpu_index, init_c, streams, err)) {
            std::cerr << "CUDA init failed: " << err << "\n";
            return 1;
        }
        std::cout << "CUDA device [" << cuda.device_index() << "]: " << cuda.device_name()
                  << " (concurrency=" << cuda.concurrency()
                  << ", streams=" << cuda.streams() << ")\n";

        if (has_password) {
            std::vector<uint8_t> p(password.begin(), password.end());
            std::vector<uint8_t> s(salt.begin(), salt.end());
            std::vector<warp_argon2::InputView> one{warp_argon2::InputView{p, s}};
            std::vector<warp_argon2::Digest> out;
            if (!cuda.hash_batch(one, out, err)) {
                std::cerr << "CUDA hash_batch failed: " << err << "\n";
                return 1;
            }
            std::cout << warp_argon2::to_hex(out[0]) << "\n";
        }

        if (self_test) {
            const bool ok = verify_backend_against_cpu(
                "CUDA",
                [&](std::span<const warp_argon2::InputView> in,
                    std::vector<warp_argon2::Digest>& out, std::string& e) {
                    return cuda.hash_batch(in, out, e);
                },
                [&](std::span<const uint8_t, 16> s16, uint64_t start, int count,
                    std::vector<warp_argon2::Digest>& out, std::string& e) {
                    return cuda.hash_indexed_range(s16, start, count, out, e);
                });
            if (!ok) return 1;
        }

        if (bench) {
            if (concurrency <= 0 && self_test) {
                if (!cuda.init(gpu_index, 0, streams, err)) {
                    std::cerr << "CUDA re-init for bench failed: " << err << "\n";
                    return 1;
                }
            }
            const auto br = cuda.bench_throughput(seconds);
            if (!br.ok) {
                std::cerr << "CUDA benchmark failed: " << br.error << "\n";
                return 1;
            }
            std::cout << std::fixed << std::setprecision(2)
                      << "CUDA throughput: " << br.hashes_per_sec << " H/s ("
                      << br.hashes << " hashes in " << br.elapsed_sec << " s, concurrency="
                      << cuda.concurrency() << ")\n";
        }
        return 0;
#else
        std::cerr << "CUDA backend not compiled on this platform\n";
        return 1;
#endif
    }

    // CPU backend
    if (has_password) {
        std::vector<uint8_t> p(password.begin(), password.end());
        std::vector<uint8_t> s(salt.begin(), salt.end());
        warp_argon2::Digest out{};
        std::string err;
        if (!warp_argon2::cpu_argon2id(warp_argon2::InputView{p, s}, out, err)) {
            std::cerr << "CPU hash failed: " << err << "\n";
            return 1;
        }
        std::cout << warp_argon2::to_hex(out) << "\n";
    }
    if (self_test) {
        std::vector<uint8_t> p{'t', 'e', 's', 't', '-', 'p', 'w', 'd'};
        std::vector<uint8_t> s{'t', 'e', 's', 't', '-', 's', 'a', 'l', 't'};
        warp_argon2::Digest out{};
        std::string err;
        if (!warp_argon2::cpu_argon2id(warp_argon2::InputView{p, s}, out, err)) {
            std::cerr << "CPU self-test failed: " << err << "\n";
            return 1;
        }
        std::cout << "[PASS] CPU libargon2 reference (" << warp_argon2::to_hex(out) << ")\n";
    }
    if (bench) {
        const double hps = warp_argon2::cpu_hash_bench(seconds, cpu_threads);
        std::cout << std::fixed << std::setprecision(2)
                  << "CPU throughput (" << std::max(1, cpu_threads) << " thread"
                  << (cpu_threads > 1 ? "s" : "") << "): " << hps << " H/s\n";
    }
    return 0;
}
