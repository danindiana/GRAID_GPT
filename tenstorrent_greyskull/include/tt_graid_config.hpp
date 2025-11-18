/**
 * @file tt_graid_config.hpp
 * @brief Custom RAID configurations and optimizations for Tenstorrent Grayskull
 * @version 0.1.0
 * @date 2024-11
 *
 * Provides predefined RAID configurations, optimization presets, and
 * performance tuning options for Grayskull e75 and e150 cards.
 */

#ifndef TT_GRAID_CONFIG_HPP
#define TT_GRAID_CONFIG_HPP

#include "tt_graid_device.hpp"
#include "tt_graid_reed_solomon.hpp"
#include <vector>
#include <string>

namespace tt_graid {

/**
 * @brief RAID configuration presets
 */
enum class RaidPreset {
    RAID5_STANDARD,      ///< RAID-5: 4 data + 1 parity
    RAID5_LARGE,         ///< RAID-5: 8 data + 1 parity
    RAID5_ENTERPRISE,    ///< RAID-5: 12 data + 1 parity
    RAID6_STANDARD,      ///< RAID-6: 4 data + 2 parity
    RAID6_LARGE,         ///< RAID-6: 8 data + 2 parity
    RAID6_ENTERPRISE,    ///< RAID-6: 12 data + 2 parity
    RAID10_MIRROR,       ///< RAID-10: Mirrored stripes
    CUSTOM               ///< Custom configuration
};

/**
 * @brief Performance optimization level
 */
enum class OptimizationLevel {
    LATENCY,             ///< Optimize for lowest latency
    THROUGHPUT,          ///< Optimize for maximum throughput
    BALANCED,            ///< Balance latency and throughput
    POWER_EFFICIENT      ///< Optimize for power efficiency
};

/**
 * @brief NoC routing strategy
 */
enum class NoCRoutingStrategy {
    NEAREST_NEIGHBOR,    ///< Use nearby cores (minimize NoC hops)
    ROW_MAJOR,           ///< Allocate cores row-by-row
    COLUMN_MAJOR,        ///< Allocate cores column-by-column
    CHECKERBOARD,        ///< Alternate pattern for better bandwidth
    CUSTOM               ///< User-defined core allocation
};

/**
 * @brief Core allocation policy
 */
struct CoreAllocationPolicy {
    NoCRoutingStrategy routing_strategy;
    uint32_t cores_per_block;      ///< Cores allocated per data block
    uint32_t max_cores;             ///< Maximum cores to use
    bool allow_core_sharing;        ///< Allow multiple blocks on same core
    bool prioritize_l1_locality;    ///< Keep data in L1 when possible

    CoreAllocationPolicy()
        : routing_strategy(NoCRoutingStrategy::NEAREST_NEIGHBOR)
        , cores_per_block(1)
        , max_cores(0)  // 0 = use all available
        , allow_core_sharing(false)
        , prioritize_l1_locality(true)
    {}
};

/**
 * @brief Memory optimization settings
 */
struct MemoryOptimization {
    bool use_interleaved_dram;      ///< Interleave DRAM access
    bool use_l1_caching;            ///< Cache frequently accessed data in L1
    bool use_multicast;             ///< Use NoC multicast for broadcast ops
    size_t dma_chunk_size;          ///< DMA transfer chunk size
    uint32_t prefetch_distance;     ///< Prefetch N blocks ahead

    MemoryOptimization()
        : use_interleaved_dram(true)
        , use_l1_caching(true)
        , use_multicast(true)
        , dma_chunk_size(256 * 1024)  // 256 KB
        , prefetch_distance(2)
    {}
};

/**
 * @brief Comprehensive RAID configuration
 */
struct ExtendedRaidConfig {
    ReedSolomonConfig rs_config;
    RaidPreset preset;
    OptimizationLevel opt_level;
    CoreAllocationPolicy core_policy;
    MemoryOptimization mem_opt;

    std::string name;
    std::string description;

    ExtendedRaidConfig()
        : preset(RaidPreset::CUSTOM)
        , opt_level(OptimizationLevel::BALANCED)
        , name("Custom")
        , description("")
    {}

    /**
     * @brief Create configuration from preset
     */
    static ExtendedRaidConfig FromPreset(RaidPreset preset, DeviceType device_type);

    /**
     * @brief Optimize for specific device
     */
    void OptimizeForDevice(const DeviceProperties& props);

    /**
     * @brief Validate configuration
     */
    bool Validate() const;

    /**
     * @brief Get estimated throughput (GB/s)
     */
    float EstimateThroughput(const DeviceProperties& props) const;

    /**
     * @brief Get estimated latency (ms)
     */
    float EstimateLatency(const DeviceProperties& props, size_t block_size) const;
};

/**
 * @brief RAID configuration builder with fluent interface
 */
class RaidConfigBuilder {
public:
    RaidConfigBuilder() = default;

    RaidConfigBuilder& WithPreset(RaidPreset preset) {
        config_.preset = preset;
        return *this;
    }

    RaidConfigBuilder& WithDataBlocks(uint32_t k) {
        config_.rs_config.k = k;
        return *this;
    }

    RaidConfigBuilder& WithParityBlocks(uint32_t m) {
        config_.rs_config.m = m;
        return *this;
    }

    RaidConfigBuilder& WithBlockSize(size_t size) {
        config_.rs_config.block_size = size;
        return *this;
    }

    RaidConfigBuilder& WithOptimization(OptimizationLevel level) {
        config_.opt_level = level;
        return *this;
    }

    RaidConfigBuilder& WithCoresPerBlock(uint32_t cores) {
        config_.core_policy.cores_per_block = cores;
        return *this;
    }

    RaidConfigBuilder& WithRoutingStrategy(NoCRoutingStrategy strategy) {
        config_.core_policy.routing_strategy = strategy;
        return *this;
    }

    RaidConfigBuilder& EnableL1Caching(bool enable = true) {
        config_.mem_opt.use_l1_caching = enable;
        return *this;
    }

    RaidConfigBuilder& EnableMulticast(bool enable = true) {
        config_.mem_opt.use_multicast = enable;
        return *this;
    }

    RaidConfigBuilder& WithDMAChunkSize(size_t size) {
        config_.mem_opt.dma_chunk_size = size;
        return *this;
    }

    RaidConfigBuilder& WithName(const std::string& name) {
        config_.name = name;
        return *this;
    }

    ExtendedRaidConfig Build() {
        return config_;
    }

private:
    ExtendedRaidConfig config_;
};

/**
 * @brief Core allocator for optimal NoC routing
 */
class CoreAllocator {
public:
    CoreAllocator(GraidDevice& device, const CoreAllocationPolicy& policy);

    /**
     * @brief Allocate cores for RAID operation
     * @param num_blocks Number of data blocks
     * @return Vector of core groups (one group per block)
     */
    std::vector<std::vector<CoreCoord>> AllocateCores(uint32_t num_blocks);

    /**
     * @brief Get optimal core layout for minimum NoC latency
     */
    std::vector<CoreCoord> GetOptimalLayout(uint32_t num_cores) const;

    /**
     * @brief Calculate total NoC communication cost
     */
    uint64_t CalculateNoCCost(const std::vector<CoreCoord>& cores) const;

    /**
     * @brief Visualize core allocation
     */
    std::string VisualizeCoreAllocation(const std::vector<std::vector<CoreCoord>>& allocation) const;

private:
    GraidDevice& device_;
    CoreAllocationPolicy policy_;
    std::vector<CoreCoord> available_cores_;

    /**
     * @brief Allocate cores using nearest neighbor strategy
     */
    std::vector<CoreCoord> AllocateNearestNeighbor(uint32_t num_cores);

    /**
     * @brief Allocate cores in row-major order
     */
    std::vector<CoreCoord> AllocateRowMajor(uint32_t num_cores);

    /**
     * @brief Allocate cores in checkerboard pattern
     */
    std::vector<CoreCoord> AllocateCheckerboard(uint32_t num_cores);
};

/**
 * @brief Performance profiler for RAID operations
 */
struct RaidProfile {
    uint64_t total_bytes;           ///< Total bytes processed
    uint64_t encode_time_us;        ///< Encoding time in microseconds
    uint64_t transfer_time_us;      ///< Data transfer time
    uint64_t compute_time_us;       ///< Computation time
    uint64_t noc_time_us;           ///< NoC communication time

    float throughput_gbps;          ///< Overall throughput
    float compute_efficiency;       ///< Compute utilization %
    float memory_efficiency;        ///< Memory bandwidth utilization %
    float noc_efficiency;           ///< NoC bandwidth utilization %

    uint32_t cores_used;            ///< Number of cores utilized
    uint32_t l1_cache_hits;         ///< L1 cache hits
    uint32_t l1_cache_misses;       ///< L1 cache misses

    /**
     * @brief Generate performance report
     */
    std::string GenerateReport() const;

    /**
     * @brief Compare with another profile
     */
    std::string CompareTo(const RaidProfile& other) const;
};

/**
 * @brief Optimized RAID operations with profiling
 */
class OptimizedRaidOperations {
public:
    OptimizedRaidOperations(
        GraidDevice& device,
        GraidMemory& memory,
        const ExtendedRaidConfig& config
    );

    /**
     * @brief Encode with profiling
     */
    bool EncodeWithProfiling(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<BufferDescriptor>& parity_blocks,
        RaidProfile& profile
    );

    /**
     * @brief Decode with profiling
     */
    bool DecodeWithProfiling(
        const std::vector<BufferDescriptor>& available_blocks,
        const std::vector<uint32_t>& available_indices,
        std::vector<BufferDescriptor>& recovered_blocks,
        const std::vector<uint32_t>& missing_indices,
        RaidProfile& profile
    );

    /**
     * @brief Auto-tune configuration for best performance
     */
    ExtendedRaidConfig AutoTune(
        size_t block_size,
        uint32_t num_iterations = 10
    );

    /**
     * @brief Get current configuration
     */
    const ExtendedRaidConfig& GetConfig() const { return config_; }

    /**
     * @brief Update configuration
     */
    void SetConfig(const ExtendedRaidConfig& config);

private:
    GraidDevice& device_;
    GraidMemory& memory_;
    ExtendedRaidConfig config_;
    std::unique_ptr<CoreAllocator> allocator_;
};

} // namespace tt_graid

#endif // TT_GRAID_CONFIG_HPP
