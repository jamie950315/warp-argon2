#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "warp_argon2/metal_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace warp_argon2 {

namespace {

constexpr int kMaxMemChunks = 12;
constexpr int kChunkHashes = 32;  // 32 * 64 MiB = 2 GiB per MTLBuffer chunk
constexpr std::size_t kWorkspaceBytes =
    static_cast<std::size_t>(kMemBlocks) * kBlockQwords * sizeof(uint64_t);

NSString* find_metallib() {
    NSFileManager* fm = [NSFileManager defaultManager];

    if (const char* env_path = std::getenv("WARP_ARGON2_METALLIB")) {
        NSString* p = [NSString stringWithUTF8String:env_path];
        if (p && [fm fileExistsAtPath:p]) return p;
    }

    NSString* cwd = [fm currentDirectoryPath];
    NSArray<NSString*>* rels = @[
        @"warp_argon2.metallib",
        @"build/warp_argon2.metallib",
        @"src/metal/warp_argon2.metallib",
        @"../warp_argon2.metallib",
    ];
    for (NSString* rel in rels) {
        NSString* path = [cwd stringByAppendingPathComponent:rel];
        if ([fm fileExistsAtPath:path]) return path;
    }

    NSString* exe = [[NSProcessInfo processInfo] arguments].firstObject;
    if (exe) {
        NSString* dir = [exe stringByDeletingLastPathComponent];
        NSArray<NSString*>* exe_rels = @[
            @"warp_argon2.metallib",
            @"../warp_argon2.metallib",
            @"metal/warp_argon2.metallib",
        ];
        for (NSString* rel in exe_rels) {
            NSString* p = [dir stringByAppendingPathComponent:rel];
            if ([fm fileExistsAtPath:p]) return p;
        }
    }
    return nil;
}

}  // namespace

int MetalBackend::max_supported_concurrency() {
    double total_gib = 16.0;
#if defined(__APPLE__)
    int64_t memsize = 0;
    std::size_t len = sizeof(memsize);
    if (sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0) == 0 && memsize > 0) {
        total_gib = static_cast<double>(memsize) / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    const double usable = std::max(0.5, total_gib - 4.0);
    int n = static_cast<int>((usable - 0.5) * 1024.0 / 64.0);
    n = std::clamp(n, 1, kMaxMemChunks * kChunkHashes);
    return n;
}

int MetalBackend::default_concurrency() {
    const int mem_max = max_supported_concurrency();
    return std::clamp(192, 1, mem_max);
}

MetalBackend::~MetalBackend() { shutdown(); }

void MetalBackend::shutdown() {
    if (h0_buf_) { CFRelease(h0_buf_); h0_buf_ = nullptr; }
    if (salt_buf_) { CFRelease(salt_buf_); salt_buf_ = nullptr; }
    if (digests_buf_) { CFRelease(digests_buf_); digests_buf_ = nullptr; }
    if (params_buf_) { CFRelease(params_buf_); params_buf_ = nullptr; }
    if (di_table_buf_) { CFRelease(di_table_buf_); di_table_buf_ = nullptr; }
    for (void*& ch : mem_chunks_) {
        if (ch) {
            CFRelease(ch);
            ch = nullptr;
        }
    }
    mem_chunks_.clear();
    if (pipeline_h0_) { CFRelease(pipeline_h0_); pipeline_h0_ = nullptr; }
    if (pipeline_range_) { CFRelease(pipeline_range_); pipeline_range_ = nullptr; }
    if (queue_) { CFRelease(queue_); queue_ = nullptr; }
    if (device_) { CFRelease(device_); device_ = nullptr; }
    dev_info_ = {};
    ready_ = false;
    concurrency_ = 0;
}

bool MetalBackend::init(int max_concurrency, std::string& err) {
    shutdown();
    if (max_concurrency <= 0) {
        max_concurrency = default_concurrency();
    }

    @autoreleasepool {
        static id activity = nil;
        if (!activity) {
            activity = [[NSProcessInfo processInfo]
                beginActivityWithOptions:NSActivityUserInitiated | NSActivityLatencyCritical
                                  reason:@"warp-argon2-metal"];
            (void)activity;
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) {
            err = "no Metal GPU device available";
            return false;
        }
        device_ = (void*)CFBridgingRetain(dev);

        dev_info_.name = [[dev name] UTF8String];
        dev_info_.recommended_max_working_set = static_cast<uint64_t>([dev recommendedMaxWorkingSetSize]);
        dev_info_.max_buffer_length = static_cast<uint64_t>([dev maxBufferLength]);
        dev_info_.has_unified_memory = ([dev hasUnifiedMemory] == YES);

        id<MTLCommandQueue> q = [dev newCommandQueue];
        if (!q) {
            err = "failed to create Metal command queue";
            shutdown();
            return false;
        }
        queue_ = (void*)CFBridgingRetain(q);

        NSString* lib_path = find_metallib();
        if (!lib_path) {
            err = "warp_argon2.metallib not found (set WARP_ARGON2_METALLIB or run from build dir)";
            shutdown();
            return false;
        }

        NSError* nserr = nil;
        id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&nserr];
        if (!lib) {
            err = std::string("failed to load warp_argon2.metallib: ") +
                  (nserr ? [[nserr localizedDescription] UTF8String] : "unknown error");
            shutdown();
            return false;
        }

        auto make_pso = [&](NSString* fn_name, void*& slot) -> bool {
            id<MTLFunction> fn = [lib newFunctionWithName:fn_name];
            if (!fn) {
                err = std::string("Metal kernel not found: ") + [fn_name UTF8String];
                return false;
            }
            NSError* perr = nil;
            id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
            if (!pso) {
                err = std::string("pipeline creation failed for ") + [fn_name UTF8String] + ": " +
                      (perr ? [[perr localizedDescription] UTF8String] : "unknown error");
                return false;
            }
            slot = (void*)CFBridgingRetain(pso);
            return true;
        };

        if (!make_pso(@"argon2id_batch_h0", pipeline_h0_) ||
            !make_pso(@"argon2id_indexed_range", pipeline_range_)) {
            shutdown();
            return false;
        }

        concurrency_ = max_concurrency;
        mem_chunks_.assign(kMaxMemChunks, nullptr);
        if (!ensure_capacity(max_concurrency, err) || !ensure_di_table(err)) {
            shutdown();
            return false;
        }

        ready_ = true;
        return true;
    }
}

bool MetalBackend::ensure_di_table(std::string& err) {
    if (di_table_buf_) return true;
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    if (!dev) {
        err = "Metal device missing";
        return false;
    }
    const NSUInteger bytes = static_cast<NSUInteger>(kDiTableBlocks * sizeof(uint32_t));
    id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!buf) {
        err = "failed to allocate Metal DI index table buffer";
        return false;
    }
    build_di_ref_table(reinterpret_cast<uint32_t*>([buf contents]));
    buf.label = @"di_ref_table";
    di_table_buf_ = (void*)CFBridgingRetain(buf);
    return true;
}

bool MetalBackend::ensure_capacity(int needed, std::string& err) {
    if (needed < 1) {
        err = "Metal batch size must be >= 1";
        return false;
    }
    id<MTLDevice> dev = (__bridge id<MTLDevice>)device_;
    if (!dev) {
        err = "Metal device not initialized";
        return false;
    }
    if (mem_chunks_.size() != kMaxMemChunks) {
        mem_chunks_.assign(kMaxMemChunks, nullptr);
    }

    auto ensure_shared_buf = [&](NSUInteger bytes, void*& slot, NSString* label) -> bool {
        if (slot) {
            id<MTLBuffer> existing = (__bridge id<MTLBuffer>)slot;
            if ([existing length] >= bytes) return true;
            CFRelease(slot);
            slot = nullptr;
        }
        id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) {
            err = std::string("Metal buffer allocation failed: ") + [label UTF8String];
            return false;
        }
        buf.label = label;
        slot = (void*)CFBridgingRetain(buf);
        return true;
    };

    if (!ensure_shared_buf(static_cast<NSUInteger>(needed) * kH0Bytes, h0_buf_, @"h0_batch")) {
        return false;
    }
    if (!ensure_shared_buf(16u, salt_buf_, @"salt16")) {
        return false;
    }
    if (!ensure_shared_buf(static_cast<NSUInteger>(needed) * kTagLen, digests_buf_, @"digests")) {
        return false;
    }
    if (!ensure_shared_buf(sizeof(uint32_t) * 2u, params_buf_, @"params")) {
        return false;
    }

    const int need_chunks = (needed + kChunkHashes - 1) / kChunkHashes;
    if (need_chunks > kMaxMemChunks) {
        err = "concurrency exceeds multi-chunk Metal limit (max 384)";
        return false;
    }

    const MTLResourceOptions mem_opts =
        MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;
    for (int c = 0; c < need_chunks; ++c) {
        if (mem_chunks_[static_cast<std::size_t>(c)]) continue;
        const std::size_t chunk_bytes = static_cast<std::size_t>(kChunkHashes) * kWorkspaceBytes;
        id<MTLBuffer> mb = [dev newBufferWithLength:chunk_bytes options:mem_opts];
        if (!mb) {
            err = "Metal Argon2 workspace allocation failed; reduce concurrency";
            return false;
        }
        mb.label = [NSString stringWithFormat:@"argon2_mem_chunk_%d", c];
        mem_chunks_[static_cast<std::size_t>(c)] = (void*)CFBridgingRetain(mb);
    }

    if (needed > concurrency_) {
        concurrency_ = needed;
    }
    return true;
}

bool MetalBackend::dispatch_h0_chunk(const uint8_t* h0_flat, int n, uint8_t* digests_out, std::string& err) {
    if (!ready_) {
        err = "Metal backend not initialized";
        return false;
    }
    if (n < 1) {
        err = "Metal chunk count must be >= 1";
        return false;
    }
    if (!ensure_capacity(n, err)) return false;

    @autoreleasepool {
        id<MTLBuffer> h0b = (__bridge id<MTLBuffer>)h0_buf_;
        id<MTLBuffer> db = (__bridge id<MTLBuffer>)digests_buf_;
        id<MTLBuffer> pb = (__bridge id<MTLBuffer>)params_buf_;
        id<MTLBuffer> dib = (__bridge id<MTLBuffer>)di_table_buf_;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline_h0_;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;

        std::memcpy([h0b contents], h0_flat, static_cast<std::size_t>(n) * kH0Bytes);
        uint32_t* params = reinterpret_cast<uint32_t*>([pb contents]);
        params[0] = static_cast<uint32_t>(n);
        params[1] = static_cast<uint32_t>(kChunkHashes);

        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:h0b offset:0 atIndex:0];
        [enc setBuffer:db offset:0 atIndex:1];
        id<MTLBuffer> c0 = (__bridge id<MTLBuffer>)mem_chunks_[0];
        for (int i = 0; i < kMaxMemChunks; ++i) {
            id<MTLBuffer> ch = mem_chunks_[static_cast<std::size_t>(i)]
                                   ? (__bridge id<MTLBuffer>)mem_chunks_[static_cast<std::size_t>(i)]
                                   : c0;
            [enc setBuffer:ch offset:0 atIndex:static_cast<NSUInteger>(2 + i)];
        }
        [enc setBuffer:pb offset:0 atIndex:14];
        [enc setBuffer:pb offset:sizeof(uint32_t) atIndex:15];
        [enc setBuffer:dib offset:0 atIndex:17];

        [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
            threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            err = std::string("Metal batch command error: ") +
                  [[cb.error localizedDescription] UTF8String];
            return false;
        }
        std::memcpy(digests_out, [db contents], static_cast<std::size_t>(n) * kTagLen);
        return true;
    }
}

bool MetalBackend::hash_batch_h0(std::span<const H0Block> h0_batch,
                                 std::vector<Digest>& out,
                                 std::string& err) {
    if (h0_batch.empty()) {
        out.clear();
        return true;
    }
    if (!ready_) {
        err = "Metal backend not initialized";
        return false;
    }

    out.resize(h0_batch.size());
    const int chunk_max = std::max(1, concurrency_);
    std::vector<uint8_t> tmp_h0;
    std::vector<uint8_t> tmp_dig;

    std::size_t done = 0;
    while (done < h0_batch.size()) {
        const int n = static_cast<int>(
            std::min<std::size_t>(h0_batch.size() - done, static_cast<std::size_t>(chunk_max)));
        tmp_h0.resize(static_cast<std::size_t>(n) * kH0Bytes);
        tmp_dig.resize(static_cast<std::size_t>(n) * kTagLen);
        for (int i = 0; i < n; ++i) {
            std::memcpy(tmp_h0.data() + static_cast<std::size_t>(i) * kH0Bytes,
                        h0_batch[done + static_cast<std::size_t>(i)].data(),
                        kH0Bytes);
        }
        if (!dispatch_h0_chunk(tmp_h0.data(), n, tmp_dig.data(), err)) {
            return false;
        }
        for (int i = 0; i < n; ++i) {
            std::memcpy(out[done + static_cast<std::size_t>(i)].data(),
                        tmp_dig.data() + static_cast<std::size_t>(i) * kTagLen,
                        kTagLen);
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

bool MetalBackend::hash_batch(std::span<const InputView> inputs,
                              std::vector<Digest>& out,
                              std::string& err) {
    if (inputs.empty()) {
        out.clear();
        return true;
    }
    std::vector<H0Block> h0s(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        h0s[i] = compute_h0(inputs[i]);
    }
    return hash_batch_h0(h0s, out, err);
}

bool MetalBackend::hash_indexed_range(std::span<const uint8_t, 16> salt16,
                                      uint64_t start_index,
                                      int count,
                                      std::vector<Digest>& out,
                                      std::string& err) {
    if (count < 1) {
        out.clear();
        return true;
    }
    if (!ready_) {
        err = "Metal backend not initialized";
        return false;
    }

    out.resize(static_cast<std::size_t>(count));
    const int chunk_max = std::max(1, concurrency_);
    int done = 0;

    @autoreleasepool {
        id<MTLBuffer> sbuf = (__bridge id<MTLBuffer>)salt_buf_;
        id<MTLBuffer> db = (__bridge id<MTLBuffer>)digests_buf_;
        id<MTLBuffer> pb = (__bridge id<MTLBuffer>)params_buf_;
        id<MTLBuffer> dib = (__bridge id<MTLBuffer>)di_table_buf_;
        id<MTLComputePipelineState> pso = (__bridge id<MTLComputePipelineState>)pipeline_range_;
        id<MTLCommandQueue> q = (__bridge id<MTLCommandQueue>)queue_;

        std::memcpy([sbuf contents], salt16.data(), 16);

        while (done < count) {
            const int n = std::min(count - done, chunk_max);
            if (!ensure_capacity(n, err)) return false;

            uint32_t* params = reinterpret_cast<uint32_t*>([pb contents]);
            params[0] = static_cast<uint32_t>(n);
            params[1] = static_cast<uint32_t>(kChunkHashes);
            const uint64_t cur_start = start_index + static_cast<uint64_t>(done);

            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:sbuf offset:0 atIndex:0];
            [enc setBuffer:db offset:0 atIndex:1];
            id<MTLBuffer> c0 = (__bridge id<MTLBuffer>)mem_chunks_[0];
            for (int i = 0; i < kMaxMemChunks; ++i) {
                id<MTLBuffer> ch = mem_chunks_[static_cast<std::size_t>(i)]
                                       ? (__bridge id<MTLBuffer>)mem_chunks_[static_cast<std::size_t>(i)]
                                       : c0;
                [enc setBuffer:ch offset:0 atIndex:static_cast<NSUInteger>(2 + i)];
            }
            [enc setBuffer:pb offset:0 atIndex:14];
            [enc setBuffer:pb offset:sizeof(uint32_t) atIndex:15];
            [enc setBytes:&cur_start length:sizeof(cur_start) atIndex:16];
            [enc setBuffer:dib offset:0 atIndex:17];

            [enc dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n), 1, 1)
                threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.error) {
                err = std::string("Metal range command error: ") +
                      [[cb.error localizedDescription] UTF8String];
                return false;
            }
            std::memcpy(out.data() + done, [db contents], static_cast<std::size_t>(n) * kTagLen);
            done += n;
        }
    }
    return true;
}

MetalBenchResult MetalBackend::bench_throughput(double seconds, int batch_size) {
    MetalBenchResult res{};
    if (!ready_) {
        res.error = "Metal backend not initialized";
        return res;
    }
    if (batch_size <= 0) {
        batch_size = std::max(1, concurrency_);
    }

    std::array<uint8_t, 16> salt{};
    for (uint8_t i = 0; i < 16; ++i) salt[i] = static_cast<uint8_t>(0x11u + i);

    std::vector<Digest> digests;
    std::string err;

    // Warmup one full wave.
    if (!hash_indexed_range(salt, 0, batch_size, digests, err)) {
        res.error = err;
        return res;
    }

    const auto t0 = std::chrono::steady_clock::now();
    uint64_t idx = static_cast<uint64_t>(batch_size);
    uint64_t total = 0;
    while (true) {
        if (!hash_indexed_range(salt, idx, batch_size, digests, err)) {
            res.error = err;
            return res;
        }
        idx += static_cast<uint64_t>(batch_size);
        total += static_cast<uint64_t>(batch_size);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (elapsed >= seconds) {
            res.ok = true;
            res.elapsed_sec = elapsed;
            res.hashes = total;
            res.hashes_per_sec = elapsed > 0.0 ? static_cast<double>(total) / elapsed : 0.0;
            return res;
        }
    }
}

}  // namespace warp_argon2
