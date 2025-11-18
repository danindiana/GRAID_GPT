/**
 * @file 03_reed_solomon_raid.cpp
 * @brief Example: Reed-Solomon RAID encoding and recovery
 * @version 0.1.0
 *
 * This example demonstrates:
 * - RAID-5 parity generation
 * - RAID-6 dual parity generation
 * - Single disk failure recovery
 * - Dual disk failure recovery
 * - Performance benchmarking
 *
 * Build:
 *   g++ -std=c++17 03_reed_solomon_raid.cpp -I../include -o reed_solomon_raid
 *
 * Run:
 *   ./reed_solomon_raid [k] [m] [block_size_kb]
 */

#include "tt_graid_device.hpp"
#include "tt_graid_memory.hpp"
#include "tt_graid_reed_solomon.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <cstring>

using namespace tt_graid;

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
    std::chrono::high_resolution_clock::time_point start_;
};

void PrintSeparator() {
    std::cout << std::string(80, '=') << std::endl;
}

void FillRandomData(void* ptr, size_t size) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<uint8_t> dis(0, 255);

    uint8_t* data = static_cast<uint8_t*>(ptr);
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    uint32_t k = 4;              // Number of data blocks
    uint32_t m = 2;              // Number of parity blocks (RAID-6)
    size_t block_size_kb = 1024; // 1 MB per block

    if (argc > 1) k = std::atoi(argv[1]);
    if (argc > 2) m = std::atoi(argv[2]);
    if (argc > 3) block_size_kb = std::atoi(argv[3]);

    size_t block_size = block_size_kb * 1024;
    size_t total_data_size = k * block_size;

    PrintSeparator();
    std::cout << "Tenstorrent Grayskull Reed-Solomon RAID" << std::endl;
    PrintSeparator();
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Data blocks (k):      " << k << std::endl;
    std::cout << "  Parity blocks (m):    " << m << std::endl;
    std::cout << "  Block size:           " << block_size_kb << " KB" << std::endl;
    std::cout << "  Total data size:      " << (total_data_size / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "  RAID level:           RAID-" << (m == 1 ? "5" : "6") << std::endl;
    PrintSeparator();
    std::cout << std::endl;

    try {
        // Initialize device
        std::cout << "Initializing Grayskull device..." << std::endl;
        GraidDevice device(0);
        GraidMemory memory(device);

        const auto& props = device.GetProperties();
        std::cout << "Device: " << props.name << std::endl;
        std::cout << "Tensix cores: " << props.num_tensix_cores << std::endl;
        std::cout << std::endl;

        Timer timer;

        // ===================================================================
        // RAID-5: Single Parity
        // ===================================================================
        if (m >= 1) {
            std::cout << "--- RAID-5: Single Parity Generation ---" << std::endl;

            // Allocate host buffers for data
            std::vector<BufferDescriptor> host_data_blocks;
            for (uint32_t i = 0; i < k; ++i) {
                auto buffer = memory.AllocateHost(block_size);
                FillRandomData(buffer.ptr, block_size);
                host_data_blocks.push_back(buffer);
                std::cout << "Data block " << i << ": " << (block_size / 1024)
                          << " KB allocated and filled" << std::endl;
            }

            // Allocate host buffer for parity
            auto host_parity = memory.AllocateHost(block_size);
            std::cout << "Parity block:   " << (block_size / 1024)
                      << " KB allocated" << std::endl;

            // Transfer to device
            std::vector<BufferDescriptor> device_data_blocks;
            for (auto& host_buf : host_data_blocks) {
                auto dev_buf = memory.AllocateDRAM(block_size);
                memory.CopyHostToDRAM(dev_buf, host_buf, block_size);
                device_data_blocks.push_back(dev_buf);
            }

            auto device_parity = memory.AllocateDRAM(block_size);

            // Create RAID-5 stripe
            std::cout << "\nGenerating RAID-5 parity..." << std::endl;
            timer.Start();

            RaidOperations raid_ops(device, memory);
            bool success = raid_ops.CreateRaid5Stripe(
                device_data_blocks,
                device_parity
            );

            double encode_time = timer.ElapsedMs();

            if (success) {
                double throughput = (total_data_size / (1024.0 * 1024.0 * 1024.0))
                                    / (encode_time / 1000.0);
                std::cout << "✓ RAID-5 encoding successful" << std::endl;
                std::cout << "  Time: " << std::fixed << std::setprecision(2)
                          << encode_time << " ms" << std::endl;
                std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
            } else {
                std::cerr << "✗ RAID-5 encoding failed" << std::endl;
            }

            // Test recovery from single failure
            std::cout << "\nSimulating single disk failure (block 2)..."
                      << std::endl;

            std::vector<BufferDescriptor> available_blocks;
            for (uint32_t i = 0; i < k; ++i) {
                if (i != 2) {  // Skip block 2 (failed)
                    available_blocks.push_back(device_data_blocks[i]);
                }
            }
            available_blocks.push_back(device_parity);

            auto recovered_block = memory.AllocateDRAM(block_size);

            timer.Start();
            success = raid_ops.RecoverSingleFailure(
                available_blocks,
                2,  // Failed block index
                recovered_block
            );
            double recovery_time = timer.ElapsedMs();

            if (success) {
                std::cout << "✓ Single block recovery successful" << std::endl;
                std::cout << "  Recovery time: " << recovery_time << " ms" << std::endl;

                // Verify recovered data
                auto recovered_host = memory.AllocateHost(block_size);
                memory.CopyDRAMToHost(recovered_host, recovered_block, block_size);

                bool data_match = (std::memcmp(
                    recovered_host.ptr,
                    host_data_blocks[2].ptr,
                    block_size
                ) == 0);

                if (data_match) {
                    std::cout << "  ✓ Recovered data verified" << std::endl;
                } else {
                    std::cerr << "  ✗ Recovered data mismatch" << std::endl;
                }

                memory.Free(recovered_host);
            } else {
                std::cerr << "✗ Recovery failed" << std::endl;
            }

            // Cleanup
            for (auto& buf : host_data_blocks) memory.Free(buf);
            for (auto& buf : device_data_blocks) memory.Free(buf);
            memory.Free(host_parity);
            memory.Free(device_parity);
            memory.Free(recovered_block);

            std::cout << std::endl;
        }

        // ===================================================================
        // RAID-6: Dual Parity
        // ===================================================================
        if (m >= 2) {
            std::cout << "--- RAID-6: Dual Parity Generation ---" << std::endl;

            // Allocate host buffers
            std::vector<BufferDescriptor> host_data_blocks;
            for (uint32_t i = 0; i < k; ++i) {
                auto buffer = memory.AllocateHost(block_size);
                FillRandomData(buffer.ptr, block_size);
                host_data_blocks.push_back(buffer);
            }

            auto host_p_parity = memory.AllocateHost(block_size);
            auto host_q_parity = memory.AllocateHost(block_size);

            std::cout << "Allocated " << k << " data blocks + 2 parity blocks"
                      << std::endl;

            // Transfer to device
            std::vector<BufferDescriptor> device_data_blocks;
            for (auto& host_buf : host_data_blocks) {
                auto dev_buf = memory.AllocateDRAM(block_size);
                memory.CopyHostToDRAM(dev_buf, host_buf, block_size);
                device_data_blocks.push_back(dev_buf);
            }

            auto device_p_parity = memory.AllocateDRAM(block_size);
            auto device_q_parity = memory.AllocateDRAM(block_size);

            // Create RAID-6 stripe
            std::cout << "\nGenerating RAID-6 dual parity..." << std::endl;
            timer.Start();

            RaidOperations raid_ops(device, memory);
            bool success = raid_ops.CreateRaid6Stripe(
                device_data_blocks,
                device_p_parity,
                device_q_parity
            );

            double encode_time = timer.ElapsedMs();

            if (success) {
                double throughput = (total_data_size / (1024.0 * 1024.0 * 1024.0))
                                    / (encode_time / 1000.0);
                std::cout << "✓ RAID-6 encoding successful" << std::endl;
                std::cout << "  Time: " << std::fixed << std::setprecision(2)
                          << encode_time << " ms" << std::endl;
                std::cout << "  Throughput: " << throughput << " GB/s" << std::endl;
            } else {
                std::cerr << "✗ RAID-6 encoding failed" << std::endl;
            }

            // Test recovery from dual failure
            std::cout << "\nSimulating dual disk failure (blocks 1 and 3)..."
                      << std::endl;

            std::vector<BufferDescriptor> available_blocks;
            std::vector<uint32_t> failed_indices = {1, 3};

            for (uint32_t i = 0; i < k; ++i) {
                if (i != 1 && i != 3) {  // Skip failed blocks
                    available_blocks.push_back(device_data_blocks[i]);
                }
            }
            available_blocks.push_back(device_p_parity);
            available_blocks.push_back(device_q_parity);

            std::vector<BufferDescriptor> recovered_blocks;
            recovered_blocks.push_back(memory.AllocateDRAM(block_size));
            recovered_blocks.push_back(memory.AllocateDRAM(block_size));

            timer.Start();
            success = raid_ops.RecoverDualFailure(
                available_blocks,
                failed_indices,
                recovered_blocks
            );
            double recovery_time = timer.ElapsedMs();

            if (success) {
                std::cout << "✓ Dual block recovery successful" << std::endl;
                std::cout << "  Recovery time: " << recovery_time << " ms" << std::endl;

                // Verify recovered data
                bool all_match = true;
                for (size_t i = 0; i < failed_indices.size(); ++i) {
                    auto recovered_host = memory.AllocateHost(block_size);
                    memory.CopyDRAMToHost(
                        recovered_host,
                        recovered_blocks[i],
                        block_size
                    );

                    bool match = (std::memcmp(
                        recovered_host.ptr,
                        host_data_blocks[failed_indices[i]].ptr,
                        block_size
                    ) == 0);

                    if (!match) all_match = false;
                    memory.Free(recovered_host);
                }

                if (all_match) {
                    std::cout << "  ✓ All recovered data verified" << std::endl;
                } else {
                    std::cerr << "  ✗ Some recovered data mismatched" << std::endl;
                }
            } else {
                std::cerr << "✗ Dual recovery failed" << std::endl;
            }

            // Cleanup
            for (auto& buf : host_data_blocks) memory.Free(buf);
            for (auto& buf : device_data_blocks) memory.Free(buf);
            memory.Free(host_p_parity);
            memory.Free(host_q_parity);
            memory.Free(device_p_parity);
            memory.Free(device_q_parity);
            for (auto& buf : recovered_blocks) memory.Free(buf);

            std::cout << std::endl;
        }

        // ===================================================================
        // Benchmark
        // ===================================================================
        std::cout << "--- Performance Benchmark ---" << std::endl;
        std::cout << "Running 100 iterations..." << std::endl;

        RaidOperations raid_ops(device, memory);
        float avg_throughput = raid_ops.BenchmarkEncode(k, m, block_size, 100);

        std::cout << "Average encoding throughput: " << std::fixed
                  << std::setprecision(2) << avg_throughput << " GB/s" << std::endl;
        std::cout << std::endl;

        PrintSeparator();
        std::cout << "Reed-Solomon RAID operations completed successfully!" << std::endl;
        PrintSeparator();
        std::cout << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
