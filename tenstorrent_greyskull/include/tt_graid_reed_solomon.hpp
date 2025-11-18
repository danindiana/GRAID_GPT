#ifndef TT_GRAID_REED_SOLOMON_HPP
#define TT_GRAID_REED_SOLOMON_HPP

/**
 * @file tt_graid_reed_solomon.hpp
 * @brief Reed-Solomon encoding/decoding for RAID on Tenstorrent Grayskull
 * @version 0.1.0
 * @date 2024-11
 *
 * Implements Reed-Solomon error correction for RAID-5/6 using:
 * - Galois Field GF(2^8) arithmetic
 * - Tensix core parallelization
 * - SIMD vector operations
 * - Network-on-Chip data distribution
 */

#include "tt_graid_device.hpp"
#include "tt_graid_memory.hpp"
#include <vector>
#include <cstdint>

namespace tt_graid {

/**
 * @brief RAID configuration
 */
enum class RaidLevel {
    RAID5,   ///< Single parity (k data + 1 parity)
    RAID6    ///< Dual parity (k data + 2 parity)
};

/**
 * @brief Galois Field GF(2^8) lookup tables
 */
struct GaloisFieldTables {
    uint8_t exp_table[512];    ///< Exponential table
    uint8_t log_table[256];    ///< Logarithm table
    uint8_t inv_table[256];    ///< Multiplicative inverse table

    /**
     * @brief Initialize lookup tables
     */
    void Initialize();

    /**
     * @brief Multiply two GF(2^8) elements
     * @param a First element
     * @param b Second element
     * @return Product in GF(2^8)
     */
    uint8_t Multiply(uint8_t a, uint8_t b) const;

    /**
     * @brief Divide two GF(2^8) elements
     * @param a Dividend
     * @param b Divisor
     * @return Quotient in GF(2^8)
     */
    uint8_t Divide(uint8_t a, uint8_t b) const;

    /**
     * @brief Raise element to power
     * @param a Base
     * @param n Exponent
     * @return a^n in GF(2^8)
     */
    uint8_t Power(uint8_t a, uint8_t n) const;
};

/**
 * @brief Reed-Solomon encoder/decoder configuration
 */
struct ReedSolomonConfig {
    uint32_t k;              ///< Number of data blocks
    uint32_t m;              ///< Number of parity blocks
    size_t block_size;       ///< Size of each block in bytes
    RaidLevel raid_level;    ///< RAID level

    // Tensix core mapping
    uint32_t cores_per_block;     ///< Cores allocated per data block
    std::vector<CoreCoord> cores; ///< Worker cores to use

    ReedSolomonConfig()
        : k(0)
        , m(0)
        , block_size(0)
        , raid_level(RaidLevel::RAID5)
        , cores_per_block(1)
    {}
};

/**
 * @brief Reed-Solomon encoder for RAID parity generation
 */
class ReedSolomonEncoder {
public:
    /**
     * @brief Constructor
     * @param device Reference to Grayskull device
     * @param memory Reference to memory manager
     * @param config Encoder configuration
     */
    ReedSolomonEncoder(
        GraidDevice& device,
        GraidMemory& memory,
        const ReedSolomonConfig& config
    );

    /**
     * @brief Destructor
     */
    ~ReedSolomonEncoder();

    /**
     * @brief Encode data blocks to generate parity blocks
     * @param data_blocks Input data blocks (k blocks)
     * @param parity_blocks Output parity blocks (m blocks)
     * @return true if successful
     */
    bool Encode(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<BufferDescriptor>& parity_blocks
    );

    /**
     * @brief Encode with streaming (for large datasets)
     * @param data_blocks Input data blocks
     * @param parity_blocks Output parity blocks
     * @param stream_size Size to process per iteration
     * @return true if successful
     */
    bool EncodeStreaming(
        const std::vector<BufferDescriptor>& data_blocks,
        std::vector<BufferDescriptor>& parity_blocks,
        size_t stream_size
    );

    /**
     * @brief Get encoding throughput
     * @return Throughput in GB/s
     */
    float GetThroughput() const { return throughput_gbps_; }

    /**
     * @brief Get performance statistics
     * @return Statistics as formatted string
     */
    std::string GetStatistics() const;

private:
    GraidDevice& device_;
    GraidMemory& memory_;
    ReedSolomonConfig config_;
    GaloisFieldTables gf_tables_;

    // Kernel handles
    void* encoding_kernel_;
    void* multicast_kernel_;

    // Performance tracking
    float throughput_gbps_;
    uint64_t total_bytes_encoded_;
    uint64_t total_time_us_;

    /**
     * @brief Compile encoding kernel
     */
    bool CompileKernel();

    /**
     * @brief Distribute GF tables to L1 SRAM
     */
    bool DistributeTables();

    /**
     * @brief Generate RAID-5 single parity
     */
    bool GenerateSingleParity(
        const std::vector<BufferDescriptor>& data_blocks,
        BufferDescriptor& parity_block
    );

    /**
     * @brief Generate RAID-6 dual parity (P and Q)
     */
    bool GenerateDualParity(
        const std::vector<BufferDescriptor>& data_blocks,
        BufferDescriptor& p_parity,
        BufferDescriptor& q_parity
    );
};

/**
 * @brief Reed-Solomon decoder for RAID recovery
 */
class ReedSolomonDecoder {
public:
    /**
     * @brief Constructor
     * @param device Reference to Grayskull device
     * @param memory Reference to memory manager
     * @param config Decoder configuration
     */
    ReedSolomonDecoder(
        GraidDevice& device,
        GraidMemory& memory,
        const ReedSolomonConfig& config
    );

    /**
     * @brief Destructor
     */
    ~ReedSolomonDecoder();

    /**
     * @brief Decode and recover missing data blocks
     * @param available_blocks Available blocks (data + parity)
     * @param available_indices Indices of available blocks
     * @param recovered_blocks Output recovered blocks
     * @param missing_indices Indices of missing blocks
     * @return true if successful
     */
    bool Decode(
        const std::vector<BufferDescriptor>& available_blocks,
        const std::vector<uint32_t>& available_indices,
        std::vector<BufferDescriptor>& recovered_blocks,
        const std::vector<uint32_t>& missing_indices
    );

    /**
     * @brief Verify data integrity
     * @param data_blocks Data blocks to verify
     * @param parity_blocks Parity blocks
     * @return true if integrity check passes
     */
    bool Verify(
        const std::vector<BufferDescriptor>& data_blocks,
        const std::vector<BufferDescriptor>& parity_blocks
    );

private:
    GraidDevice& device_;
    GraidMemory& memory_;
    ReedSolomonConfig config_;
    GaloisFieldTables gf_tables_;

    void* decoding_kernel_;

    /**
     * @brief Compile decoding kernel
     */
    bool CompileKernel();

    /**
     * @brief Build Vandermonde matrix for recovery
     */
    std::vector<std::vector<uint8_t>> BuildVandermondeMatrix(
        const std::vector<uint32_t>& available_indices
    );

    /**
     * @brief Invert matrix in GF(2^8)
     */
    std::vector<std::vector<uint8_t>> InvertMatrix(
        const std::vector<std::vector<uint8_t>>& matrix
    );
};

/**
 * @brief High-level RAID operations
 */
class RaidOperations {
public:
    /**
     * @brief Constructor
     * @param device Reference to Grayskull device
     * @param memory Reference to memory manager
     */
    RaidOperations(GraidDevice& device, GraidMemory& memory);

    /**
     * @brief Create RAID-5 stripe
     * @param data_blocks Input data blocks
     * @param parity_block Output parity block
     * @return true if successful
     */
    bool CreateRaid5Stripe(
        const std::vector<BufferDescriptor>& data_blocks,
        BufferDescriptor& parity_block
    );

    /**
     * @brief Create RAID-6 stripe
     * @param data_blocks Input data blocks
     * @param p_parity Output P parity block
     * @param q_parity Output Q parity block
     * @return true if successful
     */
    bool CreateRaid6Stripe(
        const std::vector<BufferDescriptor>& data_blocks,
        BufferDescriptor& p_parity,
        BufferDescriptor& q_parity
    );

    /**
     * @brief Recover from single disk failure (RAID-5)
     * @param available_blocks Available blocks
     * @param failed_index Index of failed block
     * @param recovered_block Output recovered block
     * @return true if successful
     */
    bool RecoverSingleFailure(
        const std::vector<BufferDescriptor>& available_blocks,
        uint32_t failed_index,
        BufferDescriptor& recovered_block
    );

    /**
     * @brief Recover from dual disk failure (RAID-6)
     * @param available_blocks Available blocks
     * @param failed_indices Indices of failed blocks
     * @param recovered_blocks Output recovered blocks
     * @return true if successful
     */
    bool RecoverDualFailure(
        const std::vector<BufferDescriptor>& available_blocks,
        const std::vector<uint32_t>& failed_indices,
        std::vector<BufferDescriptor>& recovered_blocks
    );

    /**
     * @brief Benchmark encoding performance
     * @param k Number of data blocks
     * @param m Number of parity blocks
     * @param block_size Block size in bytes
     * @param iterations Number of iterations
     * @return Average throughput in GB/s
     */
    float BenchmarkEncode(
        uint32_t k,
        uint32_t m,
        size_t block_size,
        uint32_t iterations = 100
    );

private:
    GraidDevice& device_;
    GraidMemory& memory_;

    std::unique_ptr<ReedSolomonEncoder> encoder_raid5_;
    std::unique_ptr<ReedSolomonEncoder> encoder_raid6_;
    std::unique_ptr<ReedSolomonDecoder> decoder_;
};

} // namespace tt_graid

#endif // TT_GRAID_REED_SOLOMON_HPP
