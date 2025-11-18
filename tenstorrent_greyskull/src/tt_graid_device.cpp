/**
 * @file tt_graid_device.cpp
 * @brief Implementation of device management for Tenstorrent Grayskull
 *
 * Note: This is a stub implementation for when TT-Metal is not available.
 * Full implementation requires TT-Metalium v0.55 SDK.
 */

#include "tt_graid_device.hpp"
#include <stdexcept>
#include <sstream>

namespace tt_graid {

// Static device enumeration
uint32_t GraidDevice::GetNumDevices() {
    // TODO: Call TT-Metal API when available
    // For now, return 0 in stub mode
    return 0;
}

bool GraidDevice::GetDeviceProperties(uint32_t device_id, DeviceProperties& properties) {
    // TODO: Query actual device properties
    // Stub implementation returns false
    return false;
}

// Constructor
GraidDevice::GraidDevice(uint32_t device_id)
    : device_id_(device_id)
    , is_initialized_(false)
    , device_handle_(nullptr)
{
    Initialize();
}

// Destructor
GraidDevice::~GraidDevice() {
    Cleanup();
}

// Move constructor
GraidDevice::GraidDevice(GraidDevice&& other) noexcept
    : device_id_(other.device_id_)
    , is_initialized_(other.is_initialized_)
    , properties_(std::move(other.properties_))
    , device_handle_(other.device_handle_)
{
    other.device_handle_ = nullptr;
    other.is_initialized_ = false;
}

// Move assignment
GraidDevice& GraidDevice::operator=(GraidDevice&& other) noexcept {
    if (this != &other) {
        Cleanup();

        device_id_ = other.device_id_;
        is_initialized_ = other.is_initialized_;
        properties_ = std::move(other.properties_);
        device_handle_ = other.device_handle_;

        other.device_handle_ = nullptr;
        other.is_initialized_ = false;
    }
    return *this;
}

DeviceStatus GraidDevice::GetStatus() const {
    if (!is_initialized_) {
        return DeviceStatus::NOT_INITIALIZED;
    }
    // TODO: Query actual device status
    return DeviceStatus::READY;
}

bool GraidDevice::Synchronize() {
    if (!is_initialized_) {
        return false;
    }
    // TODO: Call TT-Metal synchronization
    return true;
}

bool GraidDevice::Reset() {
    if (!is_initialized_) {
        return false;
    }
    // TODO: Reset device
    return true;
}

uint32_t GraidDevice::GetNoCLatency(const CoreCoord& src_core, const CoreCoord& dst_core) const {
    // Manhattan distance approximation for 2D mesh NoC
    uint32_t dx = (src_core.x > dst_core.x) ? (src_core.x - dst_core.x) : (dst_core.x - src_core.x);
    uint32_t dy = (src_core.y > dst_core.y) ? (src_core.y - dst_core.y) : (dst_core.y - src_core.y);

    // Approximate latency: base latency + hop latency * distance
    const uint32_t base_latency = 10;   // cycles
    const uint32_t hop_latency = 5;     // cycles per hop

    return base_latency + hop_latency * (dx + dy);
}

std::vector<CoreCoord> GraidDevice::GetWorkerCores() const {
    std::vector<CoreCoord> cores;

    if (!is_initialized_) {
        return cores;
    }

    // Generate worker core coordinates based on device type
    uint32_t grid_x = properties_.grid_size.x;
    uint32_t grid_y = properties_.grid_size.y;

    for (uint32_t y = 0; y < grid_y; ++y) {
        for (uint32_t x = 0; x < grid_x; ++x) {
            // Skip reserved cores (e.g., control cores, DRAM controllers)
            // In Grayskull, some cores in the 10x10 grid are reserved
            if (properties_.type == DeviceType::GRAYSKULL_E75 ||
                properties_.type == DeviceType::GRAYSKULL_E150) {
                // Example: Skip corners or specific positions
                // This would match actual hardware layout
                cores.push_back(CoreCoord(x, y));
            }
        }
    }

    return cores;
}

std::vector<CoreCoord> GraidDevice::GetDramCores() const {
    std::vector<CoreCoord> cores;

    if (!is_initialized_) {
        return cores;
    }

    // DRAM controllers are typically at specific positions
    // Grayskull has DRAM channels - position depends on actual hardware
    // TODO: Get from TT-Metal device descriptor

    return cores;
}

void GraidDevice::SetProfiling(bool enable) {
    // TODO: Enable/disable profiling in TT-Metal
}

std::string GraidDevice::GetProfilingResults() const {
    // TODO: Return profiling data
    return "{}";
}

void GraidDevice::Initialize() {
    // TODO: Initialize with TT-Metal API
    // For stub: simulate device based on ID

    properties_.device_id = device_id_;
    DetectDeviceType();
    PopulateProperties();

    // In stub mode, mark as not initialized (no real hardware)
    is_initialized_ = false;

    // When TT-Metal is available, this would be:
    // device_handle_ = tt::tt_metal::CreateDevice(device_id_);
    // is_initialized_ = (device_handle_ != nullptr);
}

void GraidDevice::Cleanup() {
    if (device_handle_ != nullptr) {
        // TODO: Close device with TT-Metal API
        // tt::tt_metal::CloseDevice(device_handle_);
        device_handle_ = nullptr;
    }
    is_initialized_ = false;
}

DeviceType GraidDevice::DetectDeviceType() {
    // TODO: Detect from actual hardware
    // For stub: assume e75 for device 0, e150 for device 1
    if (device_id_ == 0) {
        properties_.type = DeviceType::GRAYSKULL_E75;
    } else {
        properties_.type = DeviceType::GRAYSKULL_E150;
    }
    return properties_.type;
}

void GraidDevice::PopulateProperties() {
    if (properties_.type == DeviceType::GRAYSKULL_E75) {
        properties_.name = "Grayskull e75";
        properties_.num_tensix_cores = 96;
        properties_.core_clock_mhz = 1000;
        properties_.grid_size = CoreCoord(10, 10);
        properties_.sram_size_per_core = 1024 * 1024;  // 1 MB
        properties_.total_sram_size = 96 * 1024 * 1024;  // 96 MB
        properties_.dram_size = 8ULL * 1024 * 1024 * 1024;  // 8 GB
        properties_.dram_bandwidth_gbps = 102;
        properties_.fp8_tflops = 221.0f;
        properties_.fp16_tflops = 55.0f;
        properties_.power_tdp_watts = 75;
    } else if (properties_.type == DeviceType::GRAYSKULL_E150) {
        properties_.name = "Grayskull e150";
        properties_.num_tensix_cores = 120;
        properties_.core_clock_mhz = 1200;
        properties_.grid_size = CoreCoord(12, 10);
        properties_.sram_size_per_core = 1024 * 1024;  // 1 MB
        properties_.total_sram_size = 120 * 1024 * 1024;  // 120 MB
        properties_.dram_size = 8ULL * 1024 * 1024 * 1024;  // 8 GB
        properties_.dram_bandwidth_gbps = 118;
        properties_.fp8_tflops = 332.0f;
        properties_.fp16_tflops = 83.0f;
        properties_.power_tdp_watts = 200;
    }

    properties_.pcie_generation = 4;
    properties_.pcie_lanes = 16;
}

} // namespace tt_graid
