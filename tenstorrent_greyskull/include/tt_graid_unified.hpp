/**
 * @file tt_graid_unified.hpp
 * @brief Unified interface for both Tenstorrent Grayskull and NVIDIA GPU backends
 * @version 0.1.0
 * @date 2024-11
 *
 * Provides a unified API that automatically selects the best available hardware:
 * - Tenstorrent Grayskull e75/e150 (via TT-Metal)
 * - NVIDIA GPUs (via cuSPARSELt)
 * - CPU fallback
 */

#ifndef TT_GRAID_UNIFIED_HPP
#define TT_GRAID_UNIFIED_HPP

#include "tt_graid_device.hpp"
#include "tt_graid_memory.hpp"
#include "tt_graid_reed_solomon.hpp"
#include "tt_graid_config.hpp"

#include <memory>
#include <vector>
#include <string>

namespace tt_graid {

/**
 * @brief Hardware backend type
 */
enum class BackendType {
    TENSTORRENT_GRAYSKULL,   ///< Tenstorrent Grayskull e75/e150
    NVIDIA_GPU,              ///< NVIDIA GPU with cuSPARSELt
    AMD_GPU,                 ///< AMD GPU with ROCm (future)
    CPU_FALLBACK,            ///< CPU-only implementation
    AUTO                     ///< Automatically select best available
};

/**
 * @brief Backend capabilities
 */
struct BackendCapabilities {
    BackendType type;
    std::string name;
    bool available;              ///< Backend is available
    uint32_t num_devices;        ///< Number of devices

    // Performance characteristics
    float peak_tflops;           ///< Peak TFLOPS
    float memory_bandwidth_gbps; ///< Memory bandwidth
    uint64_t memory_size;        ///< Total memory
    uint32_t power_tdp_watts;    ///< Power consumption

    // Supported features
    bool supports_fp8;
    bool supports_fp16;
    bool supports_int8;
    bool supports_multicast;
    bool supports_rdma;

    BackendCapabilities()
        : type(BackendType::CPU_FALLBACK)
        , name("CPU")
        , available(false)
        , num_devices(0)
        , peak_tflops(0)
        , memory_bandwidth_gbps(0)
        , memory_size(0)
        , power_tdp_watts(0)
        , supports_fp8(false)
        , supports_fp16(false)
        , supports_int8(false)
        , supports_multicast(false)
        , supports_rdma(false)
    {}
};

/**
 * @brief Unified device abstraction
 */
class UnifiedDevice {
public:
    /**
     * @brief Enumerate all available backends
     */
    static std::vector<BackendCapabilities> EnumerateBackends();

    /**
     * @brief Select best backend automatically
     */
    static BackendType SelectBestBackend();

    /**
     * @brief Constructor with backend selection
     * @param backend Desired backend (AUTO for automatic)
     * @param device_id Device index
     */
    UnifiedDevice(BackendType backend = BackendType::AUTO, uint32_t device_id = 0);

    /**
     * @brief Get active backend type
     */
    BackendType GetBackendType() const { return backend_type_; }

    /**
     * @brief Get backend capabilities
     */
    const BackendCapabilities& GetCapabilities() const { return capabilities_; }

    /**
     * @brief Check if hardware is available
     */
    bool IsHardwareAvailable() const { return capabilities_.available; }

    /**
     * @brief Get backend-agnostic device properties
     */
    struct UnifiedProperties {
        std::string name;
        uint32_t num_compute_units;
        uint64_t memory_size;
        float memory_bandwidth_gbps;
        float peak_tflops;
        uint32_t power_tdp_watts;
    };

    UnifiedProperties GetProperties() const;

    /**
     * @brief Synchronize device
     */
    bool Synchronize();

private:
    BackendType backend_type_;
    BackendCapabilities capabilities_;

    // Backend-specific device handles (only one active)
    std::unique_ptr<GraidDevice> tt_device_;
    void* cuda_device_;  // Opaque CUDA device handle
    void* rocm_device_;  // Opaque ROCm device handle

    void InitializeBackend(BackendType backend, uint32_t device_id);
};

/**
 * @brief Unified memory manager
 */
class UnifiedMemory {
public:
    UnifiedMemory(UnifiedDevice& device);

    /**
     * @brief Allocate memory (automatically uses best location)
     */
    BufferDescriptor Allocate(size_t size);

    /**
     * @brief Free memory
     */
    void Free(BufferDescriptor& buffer);

    /**
     * @brief Copy data (automatically handles transfers)
     */
    bool Copy(
        BufferDescriptor& dst,
        const BufferDescriptor& src,
        size_t size
    );

    /**
     * @brief Get memory usage
     */
    std::pair<size_t, size_t> GetUsage() const;

private:
    UnifiedDevice& device_;
    std::unique_ptr<GraidMemory> tt_memory_;
    void* cuda_memory_;
};

/**
 * @brief Unified RAID operations (works with any backend)
 */
class UnifiedRaidOperations {
public:
    UnifiedRaidOperations(
        UnifiedDevice& device,
        UnifiedMemory& memory,
        const ExtendedRaidConfig& config
    );

    /**
     * @brief Encode stripe (backend-agnostic)
     */
    bool Encode(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<BufferDescriptor>& parity_blocks
    );

    /**
     * @brief Decode and recover missing blocks
     */
    bool Decode(
        const std::vector<BufferDescriptor>& available_blocks,
        const std::vector<uint32_t>& available_indices,
        std::vector<BufferDescriptor>& recovered_blocks,
        const std::vector<uint32_t>& missing_indices
    );

    /**
     * @brief Benchmark encoding performance
     */
    float BenchmarkEncode(
        uint32_t k,
        uint32_t m,
        size_t block_size,
        uint32_t iterations = 100
    );

    /**
     * @brief Auto-select optimal configuration for current backend
     */
    ExtendedRaidConfig AutoConfigure(size_t typical_block_size);

    /**
     * @brief Get backend-specific recommendations
     */
    std::string GetOptimizationRecommendations() const;

private:
    UnifiedDevice& device_;
    UnifiedMemory& memory_;
    ExtendedRaidConfig config_;

    // Backend-specific implementations
    std::unique_ptr<OptimizedRaidOperations> tt_raid_ops_;
    void* cuda_raid_ops_;  // Opaque CUDA implementation

    void SelectImplementation();
};

/**
 * @brief Multi-backend RAID system (use multiple accelerators)
 */
class MultiBackendRaid {
public:
    /**
     * @brief Add a backend to the system
     */
    void AddBackend(
        BackendType type,
        uint32_t device_id,
        float weight = 1.0f  ///< Workload distribution weight
    );

    /**
     * @brief Encode using load balancing across backends
     */
    bool EncodeDistributed(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<BufferDescriptor>& parity_blocks
    );

    /**
     * @brief Get combined throughput
     */
    float GetCombinedThroughput() const;

    /**
     * @brief Get per-backend statistics
     */
    struct BackendStats {
        BackendType type;
        std::string name;
        float throughput_gbps;
        float utilization;
        uint64_t bytes_processed;
    };

    std::vector<BackendStats> GetBackendStatistics() const;

private:
    struct BackendInstance {
        std::unique_ptr<UnifiedDevice> device;
        std::unique_ptr<UnifiedMemory> memory;
        std::unique_ptr<UnifiedRaidOperations> raid_ops;
        float weight;
    };

    std::vector<BackendInstance> backends_;

    void DistributeWorkload(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<std::vector<BufferDescriptor>>& backend_assignments
    );
};

/**
 * @brief Backend comparison and recommendation system
 */
class BackendSelector {
public:
    /**
     * @brief Compare backends for specific workload
     */
    struct ComparisonResult {
        BackendType recommended_backend;
        std::string reason;

        struct BackendScore {
            BackendType backend;
            float score;
            std::string pros;
            std::string cons;
        };

        std::vector<BackendScore> all_scores;
    };

    static ComparisonResult CompareBackends(
        const ExtendedRaidConfig& config,
        size_t expected_throughput_gbps = 0
    );

    /**
     * @brief Get recommendation for workload
     */
    static BackendType RecommendBackend(
        RaidLevel raid_level,
        uint32_t num_data_blocks,
        size_t block_size,
        bool prioritize_latency = false
    );

    /**
     * @brief Generate detailed comparison report
     */
    static std::string GenerateComparisonReport(
        const ExtendedRaidConfig& config
    );
};

/**
 * @brief Migration helper (move RAID operations between backends)
 */
class BackendMigration {
public:
    /**
     * @brief Migrate RAID operations from one backend to another
     */
    static bool MigrateBackend(
        UnifiedRaidOperations& current,
        BackendType target_backend,
        UnifiedRaidOperations& migrated
    );

    /**
     * @brief Test migration (dry run)
     */
    static bool TestMigration(
        BackendType source,
        BackendType target,
        const ExtendedRaidConfig& config
    );

    /**
     * @brief Estimate migration overhead
     */
    static float EstimateMigrationTime(
        BackendType source,
        BackendType target,
        size_t data_size
    );
};

} // namespace tt_graid

#endif // TT_GRAID_UNIFIED_HPP
