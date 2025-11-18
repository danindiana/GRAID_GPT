#ifndef TT_GRAID_DEVICE_HPP
#define TT_GRAID_DEVICE_HPP

/**
 * @file tt_graid_device.hpp
 * @brief Device management interface for Tenstorrent Grayskull e75/e150
 * @version 0.1.0
 * @date 2024-11
 *
 * This header provides device enumeration, initialization, and management
 * functionality for Tenstorrent Grayskull RISC-V AI accelerators.
 *
 * Compatible with TT-Metalium v0.55 (last supported version for Grayskull)
 */

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace tt_graid {

/**
 * @brief Grayskull hardware variants
 */
enum class DeviceType {
    GRAYSKULL_E75,   ///< e75: 96 cores @ 1.0 GHz, 75W
    GRAYSKULL_E150,  ///< e150: 120 cores @ 1.2 GHz, 200W
    UNKNOWN
};

/**
 * @brief Device status enumeration
 */
enum class DeviceStatus {
    READY,           ///< Device initialized and ready
    BUSY,            ///< Device executing operations
    ERROR,           ///< Device in error state
    NOT_INITIALIZED, ///< Device not initialized
    OFFLINE          ///< Device not detected
};

/**
 * @brief Tensix core grid coordinates
 */
struct CoreCoord {
    uint32_t x;  ///< X coordinate in grid
    uint32_t y;  ///< Y coordinate in grid

    CoreCoord(uint32_t x_ = 0, uint32_t y_ = 0) : x(x_), y(y_) {}

    bool operator==(const CoreCoord& other) const {
        return x == other.x && y == other.y;
    }
};

/**
 * @brief Device properties and capabilities
 */
struct DeviceProperties {
    DeviceType type;              ///< Hardware variant
    std::string name;             ///< Device name string
    uint32_t device_id;           ///< System device ID

    // Core configuration
    uint32_t num_tensix_cores;    ///< Total Tensix cores (96 or 120)
    uint32_t core_clock_mhz;      ///< Core clock frequency in MHz
    CoreCoord grid_size;          ///< Core grid dimensions

    // Memory configuration
    uint64_t sram_size_per_core;  ///< L1 SRAM per core (1 MB)
    uint64_t total_sram_size;     ///< Total on-chip SRAM
    uint64_t dram_size;           ///< External LPDDR4 size (8 GB)
    uint64_t dram_bandwidth_gbps; ///< Memory bandwidth (102 or 118 GB/s)

    // Compute capabilities
    float fp8_tflops;             ///< Peak FP8 performance
    float fp16_tflops;            ///< Peak FP16 performance
    uint32_t power_tdp_watts;     ///< Thermal design power

    // PCIe configuration
    uint32_t pcie_generation;     ///< PCIe generation (4)
    uint32_t pcie_lanes;          ///< PCIe lanes (16)

    DeviceProperties()
        : type(DeviceType::UNKNOWN)
        , device_id(0)
        , num_tensix_cores(0)
        , core_clock_mhz(0)
        , sram_size_per_core(0)
        , total_sram_size(0)
        , dram_size(0)
        , dram_bandwidth_gbps(0)
        , fp8_tflops(0.0f)
        , fp16_tflops(0.0f)
        , power_tdp_watts(0)
        , pcie_generation(4)
        , pcie_lanes(16)
    {}
};

/**
 * @brief Main device class for Grayskull hardware management
 */
class GraidDevice {
public:
    /**
     * @brief Enumerate all available Grayskull devices
     * @return Number of detected devices
     */
    static uint32_t GetNumDevices();

    /**
     * @brief Get properties for a specific device
     * @param device_id Device index (0-based)
     * @param properties Output properties structure
     * @return true if successful, false if device_id invalid
     */
    static bool GetDeviceProperties(uint32_t device_id, DeviceProperties& properties);

    /**
     * @brief Constructor - initializes device by ID
     * @param device_id Device index to initialize
     * @throws std::runtime_error if device initialization fails
     */
    explicit GraidDevice(uint32_t device_id = 0);

    /**
     * @brief Destructor - releases device resources
     */
    ~GraidDevice();

    // Prevent copying
    GraidDevice(const GraidDevice&) = delete;
    GraidDevice& operator=(const GraidDevice&) = delete;

    // Allow moving
    GraidDevice(GraidDevice&& other) noexcept;
    GraidDevice& operator=(GraidDevice&& other) noexcept;

    /**
     * @brief Get device properties
     * @return Reference to device properties
     */
    const DeviceProperties& GetProperties() const { return properties_; }

    /**
     * @brief Get current device status
     * @return Device status enumeration
     */
    DeviceStatus GetStatus() const;

    /**
     * @brief Synchronize device - wait for all operations to complete
     * @return true if successful
     */
    bool Synchronize();

    /**
     * @brief Reset device to initial state
     * @return true if successful
     */
    bool Reset();

    /**
     * @brief Get device ID
     * @return Device index
     */
    uint32_t GetDeviceId() const { return device_id_; }

    /**
     * @brief Check if device is initialized
     * @return true if initialized
     */
    bool IsInitialized() const { return is_initialized_; }

    /**
     * @brief Get NoC (Network-on-Chip) router configuration
     * @param src_core Source core coordinates
     * @param dst_core Destination core coordinates
     * @return Estimated latency in cycles
     */
    uint32_t GetNoCLatency(const CoreCoord& src_core, const CoreCoord& dst_core) const;

    /**
     * @brief Get worker core coordinates (excludes control cores)
     * @return Vector of worker core coordinates
     */
    std::vector<CoreCoord> GetWorkerCores() const;

    /**
     * @brief Get DRAM bank coordinates
     * @return Vector of DRAM controller core coordinates
     */
    std::vector<CoreCoord> GetDramCores() const;

    /**
     * @brief Enable/disable profiling
     * @param enable Enable profiling if true
     */
    void SetProfiling(bool enable);

    /**
     * @brief Get profiling results
     * @return Profiling data as JSON string
     */
    std::string GetProfilingResults() const;

private:
    uint32_t device_id_;              ///< Device index
    bool is_initialized_;             ///< Initialization status
    DeviceProperties properties_;     ///< Cached device properties
    void* device_handle_;             ///< Opaque device handle (TT-Metal device*)

    /**
     * @brief Initialize device hardware
     */
    void Initialize();

    /**
     * @brief Cleanup device resources
     */
    void Cleanup();

    /**
     * @brief Detect device type from hardware
     */
    DeviceType DetectDeviceType();

    /**
     * @brief Populate device properties based on type
     */
    void PopulateProperties();
};

/**
 * @brief RAII wrapper for device initialization
 */
class ScopedDevice {
public:
    explicit ScopedDevice(uint32_t device_id = 0)
        : device_(std::make_unique<GraidDevice>(device_id)) {}

    GraidDevice* get() { return device_.get(); }
    const GraidDevice* get() const { return device_.get(); }

    GraidDevice* operator->() { return device_.get(); }
    const GraidDevice* operator->() const { return device_.get(); }

private:
    std::unique_ptr<GraidDevice> device_;
};

} // namespace tt_graid

#endif // TT_GRAID_DEVICE_HPP
