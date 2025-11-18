/**
 * @file 01_device_enumeration.cpp
 * @brief Example: Enumerate and query Tenstorrent Grayskull devices
 * @version 0.1.0
 *
 * This example demonstrates:
 * - Detecting available Grayskull devices
 * - Querying device properties
 * - Displaying hardware capabilities
 *
 * Build:
 *   g++ -std=c++17 01_device_enumeration.cpp -I../include -o device_enum
 *
 * Run:
 *   ./device_enum
 */

#include "tt_graid_device.hpp"
#include <iostream>
#include <iomanip>

using namespace tt_graid;

void PrintSeparator() {
    std::cout << std::string(80, '=') << std::endl;
}

void PrintDeviceProperties(const DeviceProperties& props) {
    PrintSeparator();
    std::cout << "Device " << props.device_id << ": " << props.name << std::endl;
    PrintSeparator();

    // Device type
    std::cout << "Hardware Type:        ";
    switch (props.type) {
        case DeviceType::GRAYSKULL_E75:
            std::cout << "Grayskull e75" << std::endl;
            break;
        case DeviceType::GRAYSKULL_E150:
            std::cout << "Grayskull e150" << std::endl;
            break;
        default:
            std::cout << "Unknown" << std::endl;
            break;
    }

    // Core configuration
    std::cout << "\n--- Compute Configuration ---" << std::endl;
    std::cout << "Tensix Cores:         " << props.num_tensix_cores << std::endl;
    std::cout << "Core Clock:           " << props.core_clock_mhz << " MHz ("
              << (props.core_clock_mhz / 1000.0) << " GHz)" << std::endl;
    std::cout << "Core Grid Size:       " << props.grid_size.x << " x "
              << props.grid_size.y << std::endl;

    // Memory configuration
    std::cout << "\n--- Memory Configuration ---" << std::endl;
    std::cout << "L1 SRAM per Core:     " << (props.sram_size_per_core / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Total L1 SRAM:        " << (props.total_sram_size / (1024 * 1024))
              << " MB" << std::endl;
    std::cout << "Device DRAM:          " << (props.dram_size / (1024 * 1024 * 1024))
              << " GB LPDDR4" << std::endl;
    std::cout << "Memory Bandwidth:     " << props.dram_bandwidth_gbps
              << " GB/s" << std::endl;

    // Compute capabilities
    std::cout << "\n--- Compute Capabilities ---" << std::endl;
    std::cout << "Peak FP8 Performance: " << props.fp8_tflops
              << " TeraFLOPS" << std::endl;
    std::cout << "Peak FP16 Performance:" << props.fp16_tflops
              << " TeraFLOPS" << std::endl;

    // Power and interface
    std::cout << "\n--- Power and Interface ---" << std::endl;
    std::cout << "TDP:                  " << props.power_tdp_watts
              << " W" << std::endl;
    std::cout << "PCIe Interface:       Gen " << props.pcie_generation
              << " x" << props.pcie_lanes << std::endl;

    // Derived metrics
    std::cout << "\n--- Efficiency Metrics ---" << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "FP8 TFLOPS/Watt:      "
              << (props.fp8_tflops / props.power_tdp_watts) << std::endl;
    std::cout << "FP16 TFLOPS/Watt:     "
              << (props.fp16_tflops / props.power_tdp_watts) << std::endl;
    std::cout << "Memory BW/Watt:       "
              << (props.dram_bandwidth_gbps / props.power_tdp_watts)
              << " GB/s/W" << std::endl;

    PrintSeparator();
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "\n";
    PrintSeparator();
    std::cout << "Tenstorrent Grayskull Device Enumeration" << std::endl;
    std::cout << "GRAID_GPT - GPU-Accelerated RAID" << std::endl;
    PrintSeparator();
    std::cout << "\n";

    try {
        // Enumerate devices
        uint32_t num_devices = GraidDevice::GetNumDevices();

        std::cout << "Found " << num_devices << " Grayskull device(s)\n" << std::endl;

        if (num_devices == 0) {
            std::cout << "No Grayskull devices detected." << std::endl;
            std::cout << "\nPossible reasons:" << std::endl;
            std::cout << "  1. No Grayskull card installed" << std::endl;
            std::cout << "  2. Driver not loaded (check dmesg)" << std::endl;
            std::cout << "  3. Insufficient permissions (try sudo)" << std::endl;
            std::cout << "  4. Software stack not installed (TT-Metalium v0.55)" << std::endl;
            return 1;
        }

        // Query each device
        for (uint32_t i = 0; i < num_devices; ++i) {
            DeviceProperties props;
            if (GraidDevice::GetDeviceProperties(i, props)) {
                PrintDeviceProperties(props);
            } else {
                std::cerr << "Failed to query device " << i << std::endl;
            }
        }

        // Initialize first device
        std::cout << "Initializing device 0..." << std::endl;
        GraidDevice device(0);

        if (device.IsInitialized()) {
            std::cout << "✓ Device initialized successfully" << std::endl;

            // Get worker cores
            auto worker_cores = device.GetWorkerCores();
            std::cout << "✓ Worker cores available: " << worker_cores.size() << std::endl;

            // Get DRAM cores
            auto dram_cores = device.GetDramCores();
            std::cout << "✓ DRAM channels: " << dram_cores.size() << std::endl;

            // Test synchronization
            std::cout << "\nTesting device synchronization..." << std::endl;
            if (device.Synchronize()) {
                std::cout << "✓ Device synchronized successfully" << std::endl;
            }

            // Display worker core grid
            std::cout << "\nWorker Core Grid (first 10 cores):" << std::endl;
            for (size_t i = 0; i < std::min(size_t(10), worker_cores.size()); ++i) {
                const auto& core = worker_cores[i];
                std::cout << "  Core " << std::setw(2) << i << ": ("
                          << std::setw(2) << core.x << ", "
                          << std::setw(2) << core.y << ")" << std::endl;
            }

            if (worker_cores.size() > 10) {
                std::cout << "  ... and " << (worker_cores.size() - 10)
                          << " more cores" << std::endl;
            }

        } else {
            std::cerr << "✗ Device initialization failed" << std::endl;
            return 1;
        }

        std::cout << "\n";
        PrintSeparator();
        std::cout << "Device enumeration completed successfully!" << std::endl;
        PrintSeparator();
        std::cout << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
}
