/**
 * GPU RAID Drive Benchmarking Tool
 *
 * Benchmarks actual GPU RAID performance with real drives
 * Tests various configurations to find optimal settings
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include "../include/gpu_raid.h"

struct BenchmarkResult {
    std::string test_name;
    size_t block_size_kb;
    uint32_t num_drives;
    uint32_t stripe_size_kb;
    double encode_throughput_gbs;
    double decode_throughput_gbs;
    double encode_latency_ms;
    double decode_latency_ms;
    uint32_t gpu_utilization;
    float gpu_temperature;
};

class DriveBenchmark {
public:
    DriveBenchmark() : handle_(nullptr) {}

    ~DriveBenchmark() {
        if (handle_) {
            gpu_raid_destroy(handle_);
        }
    }

    bool initialize(int raid_level, int num_drives, int stripe_size_kb, int gpu_device = 0) {
        gpu_raid_config_t config = {
            .raid_level = (raid_level == 5) ? GPU_RAID_LEVEL_5 : GPU_RAID_LEVEL_6,
            .num_data_drives = static_cast<uint32_t>(num_drives),
            .stripe_size_kb = static_cast<uint32_t>(stripe_size_kb),
            .gpu_device_id = gpu_device,
            .device_type = GPU_RAID_DEVICE_AUTO,
            .memory_pool_size_mb = 4096,
            .num_streams = 4,
            .enable_profiling = true
        };

        gpu_raid_error_t err = gpu_raid_init(&config, &handle_);
        if (err != GPU_RAID_SUCCESS) {
            std::cerr << "GPU RAID initialization failed: "
                     << gpu_raid_get_error_string(err) << "\n";
            return false;
        }

        num_drives_ = num_drives;
        raid_level_ = raid_level;
        return true;
    }

    BenchmarkResult run_throughput_test(size_t block_size_kb, int iterations = 10) {
        BenchmarkResult result;
        result.test_name = "Throughput Test";
        result.block_size_kb = block_size_kb;
        result.num_drives = num_drives_;

        size_t block_size = block_size_kb * 1024;

        // Allocate buffers
        std::vector<uint8_t*> data(num_drives_);
        std::vector<const uint8_t*> data_const(num_drives_);

        for (int i = 0; i < num_drives_; i++) {
            data[i] = new uint8_t[block_size];
            data_const[i] = data[i];

            // Fill with test pattern
            memset(data[i], 0xAA ^ i, block_size);
        }

        int num_parity = (raid_level_ == 5) ? 1 : 2;
        std::vector<uint8_t*> parities(num_parity);
        for (int i = 0; i < num_parity; i++) {
            parities[i] = new uint8_t[block_size];
        }

        // Warmup
        for (int i = 0; i < 3; i++) {
            gpu_raid_encode(handle_, data_const.data(), parities.data(),
                          num_drives_, block_size);
        }

        // Benchmark encode
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            gpu_raid_error_t err = gpu_raid_encode(handle_, data_const.data(),
                                                   parities.data(), num_drives_, block_size);
            if (err != GPU_RAID_SUCCESS) {
                std::cerr << "Encode failed: " << gpu_raid_get_error_string(err) << "\n";
                break;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();

        double encode_time_s = std::chrono::duration<double>(end - start).count();
        size_t total_bytes = static_cast<size_t>(iterations) * num_drives_ * block_size;
        result.encode_throughput_gbs = (total_bytes / encode_time_s) / (1024.0 * 1024.0 * 1024.0);
        result.encode_latency_ms = (encode_time_s / iterations) * 1000.0;

        // Benchmark reconstruct (decode)
        uint32_t failed_idx = num_drives_ / 2;
        uint8_t* recovered = new uint8_t[block_size];

        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; i++) {
            std::vector<const uint8_t*> partial_data = data_const;
            partial_data[failed_idx] = nullptr;

            std::vector<const uint8_t*> parity_const(num_parity);
            for (int p = 0; p < num_parity; p++) {
                parity_const[p] = parities[p];
            }

            gpu_raid_error_t err = gpu_raid_reconstruct(
                handle_, partial_data.data(), parity_const.data(),
                &failed_idx, 1, &recovered, num_drives_, block_size
            );

            if (err != GPU_RAID_SUCCESS) {
                std::cerr << "Reconstruct failed: " << gpu_raid_get_error_string(err) << "\n";
                break;
            }
        }
        end = std::chrono::high_resolution_clock::now();

        double decode_time_s = std::chrono::duration<double>(end - start).count();
        result.decode_throughput_gbs = (total_bytes / decode_time_s) / (1024.0 * 1024.0 * 1024.0);
        result.decode_latency_ms = (decode_time_s / iterations) * 1000.0;

        // Get GPU stats
        gpu_raid_stats_t stats;
        gpu_raid_get_stats(handle_, &stats);
        result.gpu_utilization = stats.gpu_utilization_percent;
        result.gpu_temperature = stats.gpu_temperature_c;

        // Cleanup
        for (auto ptr : data) delete[] ptr;
        for (auto ptr : parities) delete[] ptr;
        delete[] recovered;

        return result;
    }

    std::vector<BenchmarkResult> run_comprehensive_benchmark() {
        std::vector<BenchmarkResult> results;

        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          GPU RAID Comprehensive Benchmark                  ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "Configuration:\n";
        std::cout << "  RAID Level: RAID " << raid_level_ << "\n";
        std::cout << "  Number of Drives: " << num_drives_ << "\n\n";

        std::cout << "Running benchmarks (this may take a few minutes)...\n\n";

        // Test different block sizes
        std::vector<size_t> block_sizes = {64, 128, 256, 512, 1024, 2048};

        std::cout << "╔════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════╗\n";
        std::cout << "║ Block Size ║  Encode   ║  Decode   ║   Encode  ║   Decode  ║  GPU  ║\n";
        std::cout << "║    (KB)    ║   (GB/s)  ║   (GB/s)  ║    (ms)   ║    (ms)   ║ Temp  ║\n";
        std::cout << "╠════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════╣\n";

        for (size_t block_size : block_sizes) {
            auto result = run_throughput_test(block_size, 20);
            results.push_back(result);

            printf("║ %10zu ║ %9.2f ║ %9.2f ║ %9.3f ║ %9.3f ║ %5.1f ║\n",
                   block_size,
                   result.encode_throughput_gbs,
                   result.decode_throughput_gbs,
                   result.encode_latency_ms,
                   result.decode_latency_ms,
                   result.gpu_temperature);
        }

        std::cout << "╚════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════╝\n\n";

        // Find optimal block size
        auto best = std::max_element(results.begin(), results.end(),
            [](const BenchmarkResult& a, const BenchmarkResult& b) {
                return a.encode_throughput_gbs < b.encode_throughput_gbs;
            });

        if (best != results.end()) {
            std::cout << "Optimal Block Size: " << best->block_size_kb << " KB\n";
            std::cout << "  Peak Encode: " << best->encode_throughput_gbs << " GB/s\n";
            std::cout << "  Peak Decode: " << best->decode_throughput_gbs << " GB/s\n\n";
        }

        return results;
    }

    void run_real_drive_test(const std::vector<std::string>& device_paths) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          Real Drive I/O Test                               ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "⚠ WARNING: This test will WRITE to the specified drives!\n";
        std::cout << "           All data on these drives may be LOST!\n\n";
        std::cout << "Drives to test:\n";
        for (const auto& dev : device_paths) {
            std::cout << "  - " << dev << "\n";
        }

        std::cout << "\nType 'YES' to confirm: ";
        std::string confirm;
        std::cin >> confirm;

        if (confirm != "YES") {
            std::cout << "Test cancelled.\n";
            return;
        }

        // Open devices
        std::vector<int> fds;
        for (const auto& dev : device_paths) {
            int fd = open(dev.c_str(), O_RDWR | O_DIRECT | O_SYNC);
            if (fd < 0) {
                std::cerr << "Failed to open " << dev << ": " << strerror(errno) << "\n";
                for (int f : fds) close(f);
                return;
            }
            fds.push_back(fd);
        }

        std::cout << "\nRunning real drive I/O test...\n";

        const size_t test_size = 1024 * 1024 * 1024;  // 1 GB per drive
        const size_t block_size = 1024 * 1024;         // 1 MB blocks

        // Allocate aligned buffers
        std::vector<uint8_t*> buffers(fds.size());
        for (size_t i = 0; i < fds.size(); i++) {
            posix_memalign((void**)&buffers[i], 4096, block_size);
            memset(buffers[i], 0xAA ^ i, block_size);
        }

        // Write test
        auto start = std::chrono::high_resolution_clock::now();

        for (size_t offset = 0; offset < test_size; offset += block_size) {
            // Write to all drives in parallel (simplified - real implementation would use threads)
            for (size_t i = 0; i < fds.size(); i++) {
                ssize_t written = write(fds[i], buffers[i], block_size);
                if (written != static_cast<ssize_t>(block_size)) {
                    std::cerr << "Write error on drive " << i << "\n";
                }
            }
        }

        // Sync all drives
        for (int fd : fds) {
            fsync(fd);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_s = std::chrono::duration<double>(end - start).count();
        double throughput_gbs = (test_size * fds.size() / elapsed_s) / (1024.0 * 1024.0 * 1024.0);

        std::cout << "\nWrite Performance:\n";
        std::cout << "  Total Data: " << (test_size * fds.size()) / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "  Time: " << elapsed_s << " seconds\n";
        std::cout << "  Throughput: " << throughput_gbs << " GB/s\n";

        // Cleanup
        for (auto buf : buffers) free(buf);
        for (int fd : fds) close(fd);

        std::cout << "\n⚠ Remember to recreate filesystems on these drives!\n\n";
    }

private:
    gpu_raid_handle_t handle_;
    int num_drives_;
    int raid_level_;
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n";
    std::cout << "GPU RAID Drive Benchmarking Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  -r, --raid LEVEL       RAID level (5 or 6, default: 5)\n";
    std::cout << "  -n, --drives COUNT     Number of drives (default: 4)\n";
    std::cout << "  -s, --stripe SIZE      Stripe size in KB (default: 256)\n";
    std::cout << "  -b, --block SIZE       Block size in KB for single test\n";
    std::cout << "  -c, --comprehensive    Run comprehensive benchmark\n";
    std::cout << "  -d, --devices DEV...   Test real drives (DESTRUCTIVE!)\n";
    std::cout << "  -g, --gpu ID           GPU device ID (default: 0)\n";
    std::cout << "  -h, --help             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Comprehensive benchmark:\n";
    std::cout << "  " << program << " --comprehensive --raid=5 --drives=6\n\n";
    std::cout << "  # Single block size test:\n";
    std::cout << "  " << program << " --block=512 --raid=6 --drives=4\n\n";
    std::cout << "  # Real drive test (DANGEROUS):\n";
    std::cout << "  " << program << " --devices /dev/sdb /dev/sdc /dev/sdd /dev/sde\n\n";
}

int main(int argc, char** argv) {
    int raid_level = 5;
    int num_drives = 4;
    int stripe_size = 256;
    int block_size = -1;
    int gpu_device = 0;
    bool comprehensive = false;
    std::vector<std::string> real_devices;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-r" || arg == "--raid") {
            if (i + 1 < argc) raid_level = atoi(argv[++i]);
        } else if (arg == "-n" || arg == "--drives") {
            if (i + 1 < argc) num_drives = atoi(argv[++i]);
        } else if (arg == "-s" || arg == "--stripe") {
            if (i + 1 < argc) stripe_size = atoi(argv[++i]);
        } else if (arg == "-b" || arg == "--block") {
            if (i + 1 < argc) block_size = atoi(argv[++i]);
        } else if (arg == "-g" || arg == "--gpu") {
            if (i + 1 < argc) gpu_device = atoi(argv[++i]);
        } else if (arg == "-c" || arg == "--comprehensive") {
            comprehensive = true;
        } else if (arg == "-d" || arg == "--devices") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                real_devices.push_back(argv[++i]);
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Validate
    if (raid_level != 5 && raid_level != 6) {
        std::cerr << "Error: RAID level must be 5 or 6\n";
        return 1;
    }

    if (num_drives < 3) {
        std::cerr << "Error: At least 3 drives required\n";
        return 1;
    }

    std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║          GPU RAID Drive Benchmark Tool                     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

    DriveBenchmark benchmark;

    if (!benchmark.initialize(raid_level, num_drives, stripe_size, gpu_device)) {
        return 1;
    }

    if (!real_devices.empty()) {
        benchmark.run_real_drive_test(real_devices);
    } else if (comprehensive) {
        benchmark.run_comprehensive_benchmark();
    } else if (block_size > 0) {
        auto result = benchmark.run_throughput_test(block_size, 20);

        std::cout << "\nBenchmark Results:\n";
        std::cout << "  Block Size: " << result.block_size_kb << " KB\n";
        std::cout << "  Encode Throughput: " << result.encode_throughput_gbs << " GB/s\n";
        std::cout << "  Decode Throughput: " << result.decode_throughput_gbs << " GB/s\n";
        std::cout << "  Encode Latency: " << result.encode_latency_ms << " ms\n";
        std::cout << "  Decode Latency: " << result.decode_latency_ms << " ms\n";
        std::cout << "  GPU Utilization: " << result.gpu_utilization << "%\n";
        std::cout << "  GPU Temperature: " << result.gpu_temperature << "°C\n\n";
    } else {
        std::cout << "No test specified. Use --comprehensive, --block, or --devices\n";
        std::cout << "Run with --help for usage information\n\n";
    }

    return 0;
}
