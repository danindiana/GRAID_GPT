/**
 * @file 02_memory_operations.cpp
 * @brief Example: Memory allocation and transfer operations
 * @version 0.1.0
 *
 * This example demonstrates:
 * - Host memory allocation
 * - Device DRAM allocation
 * - L1 SRAM allocation
 * - Memory transfers between host and device
 * - Memory bandwidth measurement
 *
 * Build:
 *   g++ -std=c++17 02_memory_operations.cpp -I../include -o memory_ops
 *
 * Run:
 *   ./memory_ops [size_mb]
 */

#include "tt_graid_device.hpp"
#include "tt_graid_memory.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstring>

using namespace tt_graid;

// Timing utilities
class Timer {
public:
    void Start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double ElapsedMs() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::time_resolution_clock::time_point start_;
};

void PrintBandwidth(const std::string& operation, size_t bytes, double time_ms) {
    double bandwidth_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << operation << ": "
              << bandwidth_gbps << " GB/s ("
              << time_ms << " ms for "
              << (bytes / (1024.0 * 1024.0)) << " MB)"
              << std::endl;
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    size_t test_size_mb = 64;
    if (argc > 1) {
        test_size_mb = std::atoi(argv[1]);
    }
    size_t test_size = test_size_mb * 1024 * 1024;

    std::cout << "\n=== Tenstorrent Grayskull Memory Operations ===" << std::endl;
    std::cout << "Test size: " << test_size_mb << " MB\n" << std::endl;

    try {
        // Initialize device
        std::cout << "Initializing device..." << std::endl;
        GraidDevice device(0);
        GraidMemory memory(device);

        const auto& props = device.GetProperties();
        std::cout << "Device: " << props.name << std::endl;
        std::cout << "DRAM: " << (props.dram_size / (1024 * 1024 * 1024))
                  << " GB @ " << props.dram_bandwidth_gbps << " GB/s" << std::endl;
        std::cout << std::endl;

        Timer timer;

        // 1. Allocate host memory
        std::cout << "--- Host Memory Allocation ---" << std::endl;
        timer.Start();
        auto host_buffer = memory.AllocateHost(test_size);
        std::cout << "Allocated " << (test_size / (1024 * 1024))
                  << " MB host memory in " << timer.ElapsedMs() << " ms"
                  << std::endl;

        // Initialize with pattern
        uint8_t* host_ptr = static_cast<uint8_t*>(host_buffer.ptr);
        for (size_t i = 0; i < test_size; ++i) {
            host_ptr[i] = static_cast<uint8_t>(i % 256);
        }
        std::cout << "Initialized with test pattern" << std::endl;
        std::cout << std::endl;

        // 2. Allocate device DRAM
        std::cout << "--- Device DRAM Allocation ---" << std::endl;
        timer.Start();
        auto dram_buffer = memory.AllocateDRAM(test_size, true);
        std::cout << "Allocated " << (test_size / (1024 * 1024))
                  << " MB device DRAM in " << timer.ElapsedMs() << " ms"
                  << std::endl;
        std::cout << std::endl;

        // 3. Host to DRAM transfer
        std::cout << "--- Host to DRAM Transfer ---" << std::endl;
        timer.Start();
        bool success = memory.CopyHostToDRAM(dram_buffer, host_buffer, test_size);
        double h2d_time = timer.ElapsedMs();

        if (success) {
            PrintBandwidth("Host → DRAM", test_size, h2d_time);
        } else {
            std::cerr << "✗ Host to DRAM transfer failed" << std::endl;
        }
        std::cout << std::endl;

        // 4. Allocate another host buffer for readback
        std::cout << "--- DRAM to Host Transfer ---" << std::endl;
        auto host_readback = memory.AllocateHost(test_size);

        timer.Start();
        success = memory.CopyDRAMToHost(host_readback, dram_buffer, test_size);
        double d2h_time = timer.ElapsedMs();

        if (success) {
            PrintBandwidth("DRAM → Host", test_size, d2h_time);

            // Verify data
            uint8_t* readback_ptr = static_cast<uint8_t*>(host_readback.ptr);
            bool data_match = (std::memcmp(host_ptr, readback_ptr, test_size) == 0);

            if (data_match) {
                std::cout << "✓ Data verification passed" << std::endl;
            } else {
                std::cerr << "✗ Data verification failed" << std::endl;
            }
        } else {
            std::cerr << "✗ DRAM to Host transfer failed" << std::endl;
        }
        std::cout << std::endl;

        // 5. L1 SRAM operations (smaller size)
        size_t l1_size = 512 * 1024;  // 512 KB (< 1 MB per core)
        std::cout << "--- L1 SRAM Operations ---" << std::endl;
        std::cout << "L1 test size: " << (l1_size / 1024) << " KB" << std::endl;

        auto worker_cores = device.GetWorkerCores();
        if (worker_cores.empty()) {
            std::cerr << "✗ No worker cores available" << std::endl;
        } else {
            // Allocate L1 on first core
            auto l1_buffer = memory.AllocateL1(l1_size, worker_cores[0]);
            std::cout << "Allocated L1 SRAM on core ("
                      << worker_cores[0].x << ", " << worker_cores[0].y << ")"
                      << std::endl;

            // DRAM to L1 transfer
            timer.Start();
            success = memory.CopyDRAMToL1(l1_buffer, dram_buffer, l1_size);
            double dram2l1_time = timer.ElapsedMs();

            if (success) {
                PrintBandwidth("DRAM → L1", l1_size, dram2l1_time);
            }

            // L1 back to DRAM
            timer.Start();
            success = memory.CopyL1ToDRAM(dram_buffer, l1_buffer, l1_size);
            double l12dram_time = timer.ElapsedMs();

            if (success) {
                PrintBandwidth("L1 → DRAM", l1_size, l12dram_time);
            }

            memory.Free(l1_buffer);
        }
        std::cout << std::endl;

        // 6. Sharded L1 allocation
        if (worker_cores.size() >= 4) {
            std::cout << "--- Sharded L1 Allocation ---" << std::endl;
            std::vector<CoreCoord> shard_cores(worker_cores.begin(),
                                               worker_cores.begin() + 4);

            size_t total_shard_size = 2 * 1024 * 1024;  // 2 MB total
            auto sharded_buffers = memory.AllocateL1Sharded(
                total_shard_size,
                shard_cores
            );

            std::cout << "Allocated " << (total_shard_size / (1024 * 1024))
                      << " MB across " << sharded_buffers.size() << " cores"
                      << std::endl;

            for (size_t i = 0; i < sharded_buffers.size(); ++i) {
                std::cout << "  Shard " << i << ": "
                          << (sharded_buffers[i].size / 1024) << " KB on core ("
                          << sharded_buffers[i].core.x << ", "
                          << sharded_buffers[i].core.y << ")" << std::endl;
            }

            memory.Free(sharded_buffers);
            std::cout << std::endl;
        }

        // 7. Memory usage statistics
        std::cout << "--- Memory Usage Statistics ---" << std::endl;
        auto host_usage = memory.GetMemoryUsage(MemoryLocation::HOST);
        auto dram_usage = memory.GetMemoryUsage(MemoryLocation::DEVICE_DRAM);
        auto l1_usage = memory.GetMemoryUsage(MemoryLocation::DEVICE_L1);

        std::cout << "Host Memory:  "
                  << (host_usage.first / (1024 * 1024)) << " / "
                  << (host_usage.second / (1024 * 1024)) << " MB allocated"
                  << std::endl;

        std::cout << "Device DRAM:  "
                  << (dram_usage.first / (1024 * 1024)) << " / "
                  << (dram_usage.second / (1024 * 1024)) << " MB allocated"
                  << std::endl;

        std::cout << "L1 SRAM:      "
                  << (l1_usage.first / (1024 * 1024)) << " / "
                  << (l1_usage.second / (1024 * 1024)) << " MB allocated"
                  << std::endl;
        std::cout << std::endl;

        // Cleanup
        memory.Free(host_buffer);
        memory.Free(dram_buffer);
        memory.Free(host_readback);

        std::cout << "=== Memory operations completed successfully ===" << std::endl;
        std::cout << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
