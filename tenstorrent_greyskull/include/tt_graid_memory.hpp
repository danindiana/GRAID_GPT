#ifndef TT_GRAID_MEMORY_HPP
#define TT_GRAID_MEMORY_HPP

/**
 * @file tt_graid_memory.hpp
 * @brief Memory management for Tenstorrent Grayskull devices
 * @version 0.1.0
 * @date 2024-11
 *
 * Provides memory allocation, transfer, and management for:
 * - Host memory (DDR4/DDR5)
 * - Device LPDDR4 (8 GB)
 * - L1 SRAM (1 MB per Tensix core)
 * - Register files
 */

#include "tt_graid_device.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace tt_graid {

/**
 * @brief Memory location enumeration
 */
enum class MemoryLocation {
    HOST,        ///< Host system memory
    DEVICE_DRAM, ///< Device LPDDR4 memory
    DEVICE_L1,   ///< Tensix core L1 SRAM
    INTERLEAVED, ///< Interleaved across DRAM banks
    SHARDED      ///< Sharded across L1 SRAM cores
};

/**
 * @brief Memory buffer descriptor
 */
struct BufferDescriptor {
    void* ptr;                     ///< Memory pointer
    size_t size;                   ///< Buffer size in bytes
    MemoryLocation location;       ///< Memory location
    uint32_t alignment;            ///< Alignment in bytes
    CoreCoord core;                ///< Core coordinate (for L1)
    bool is_allocated;             ///< Allocation status

    BufferDescriptor()
        : ptr(nullptr)
        , size(0)
        , location(MemoryLocation::HOST)
        , alignment(16)
        , is_allocated(false)
    {}
};

/**
 * @brief Memory allocator for Grayskull devices
 */
class GraidMemory {
public:
    /**
     * @brief Constructor
     * @param device Reference to initialized device
     */
    explicit GraidMemory(GraidDevice& device);

    /**
     * @brief Destructor - frees all allocated buffers
     */
    ~GraidMemory();

    // Prevent copying
    GraidMemory(const GraidMemory&) = delete;
    GraidMemory& operator=(const GraidMemory&) = delete;

    /**
     * @brief Allocate host memory (page-locked for DMA)
     * @param size Size in bytes
     * @param alignment Alignment requirement (default 16)
     * @return Buffer descriptor
     */
    BufferDescriptor AllocateHost(size_t size, uint32_t alignment = 16);

    /**
     * @brief Allocate device DRAM memory
     * @param size Size in bytes
     * @param interleaved Use interleaved mode across DRAM banks
     * @return Buffer descriptor
     */
    BufferDescriptor AllocateDRAM(size_t size, bool interleaved = true);

    /**
     * @brief Allocate L1 SRAM on specific core
     * @param size Size in bytes (max 1 MB per core)
     * @param core Target core coordinates
     * @return Buffer descriptor
     */
    BufferDescriptor AllocateL1(size_t size, const CoreCoord& core);

    /**
     * @brief Allocate sharded L1 memory across multiple cores
     * @param size Total size in bytes
     * @param cores Vector of core coordinates
     * @param shard_size Size per shard (default: equal distribution)
     * @return Vector of buffer descriptors (one per core)
     */
    std::vector<BufferDescriptor> AllocateL1Sharded(
        size_t size,
        const std::vector<CoreCoord>& cores,
        size_t shard_size = 0
    );

    /**
     * @brief Free a buffer
     * @param buffer Buffer descriptor to free
     */
    void Free(BufferDescriptor& buffer);

    /**
     * @brief Free multiple buffers
     * @param buffers Vector of buffer descriptors
     */
    void Free(std::vector<BufferDescriptor>& buffers);

    /**
     * @brief Copy data from host to device DRAM
     * @param dst Destination buffer (DRAM)
     * @param src Source buffer (host)
     * @param size Size in bytes
     * @param async Asynchronous transfer
     * @return true if successful
     */
    bool CopyHostToDRAM(
        BufferDescriptor& dst,
        const BufferDescriptor& src,
        size_t size,
        bool async = false
    );

    /**
     * @brief Copy data from device DRAM to host
     * @param dst Destination buffer (host)
     * @param src Source buffer (DRAM)
     * @param size Size in bytes
     * @param async Asynchronous transfer
     * @return true if successful
     */
    bool CopyDRAMToHost(
        BufferDescriptor& dst,
        const BufferDescriptor& src,
        size_t size,
        bool async = false
    );

    /**
     * @brief Copy data from DRAM to L1
     * @param dst Destination buffer (L1)
     * @param src Source buffer (DRAM)
     * @param size Size in bytes
     * @return true if successful
     */
    bool CopyDRAMToL1(
        BufferDescriptor& dst,
        const BufferDescriptor& src,
        size_t size
    );

    /**
     * @brief Copy data from L1 to DRAM
     * @param dst Destination buffer (DRAM)
     * @param src Source buffer (L1)
     * @param size Size in bytes
     * @return true if successful
     */
    bool CopyL1ToDRAM(
        BufferDescriptor& dst,
        const BufferDescriptor& src,
        size_t size
    );

    /**
     * @brief Multicast data from DRAM to multiple L1 cores
     * @param dst Vector of L1 destinations
     * @param src Source DRAM buffer
     * @param size Size in bytes
     * @return true if successful
     */
    bool MulticastDRAMToL1(
        std::vector<BufferDescriptor>& dst,
        const BufferDescriptor& src,
        size_t size
    );

    /**
     * @brief Set memory to a value
     * @param buffer Target buffer
     * @param value Value to set (byte)
     * @param size Size in bytes
     * @return true if successful
     */
    bool Memset(BufferDescriptor& buffer, uint8_t value, size_t size);

    /**
     * @brief Synchronize all memory operations
     * @return true if successful
     */
    bool Synchronize();

    /**
     * @brief Get memory usage statistics
     * @param location Memory location to query
     * @return Pair of (used_bytes, total_bytes)
     */
    std::pair<size_t, size_t> GetMemoryUsage(MemoryLocation location) const;

    /**
     * @brief Get memory bandwidth statistics
     * @return Bandwidth in GB/s
     */
    float GetMemoryBandwidth() const;

private:
    GraidDevice& device_;              ///< Reference to device
    void* memory_manager_;             ///< Opaque memory manager handle

    size_t host_allocated_;            ///< Host memory allocated
    size_t dram_allocated_;            ///< DRAM allocated
    size_t l1_allocated_;              ///< L1 SRAM allocated

    /**
     * @brief Validate buffer parameters
     */
    bool ValidateBuffer(const BufferDescriptor& buffer, size_t required_size) const;

    /**
     * @brief Align size to required boundary
     */
    static size_t AlignSize(size_t size, uint32_t alignment);
};

/**
 * @brief RAII wrapper for memory buffers
 */
class ScopedBuffer {
public:
    ScopedBuffer(GraidMemory& memory, const BufferDescriptor& buffer)
        : memory_(memory), buffer_(buffer) {}

    ~ScopedBuffer() {
        memory_.Free(buffer_);
    }

    // Prevent copying
    ScopedBuffer(const ScopedBuffer&) = delete;
    ScopedBuffer& operator=(const ScopedBuffer&) = delete;

    const BufferDescriptor& get() const { return buffer_; }
    BufferDescriptor& get() { return buffer_; }

private:
    GraidMemory& memory_;
    BufferDescriptor buffer_;
};

} // namespace tt_graid

#endif // TT_GRAID_MEMORY_HPP
