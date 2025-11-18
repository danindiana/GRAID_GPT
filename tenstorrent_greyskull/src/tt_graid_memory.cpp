/**
 * @file tt_graid_memory.cpp
 * @brief Implementation of memory management for Tenstorrent Grayskull
 *
 * Note: This is a stub implementation for when TT-Metal is not available.
 */

#include "tt_graid_memory.hpp"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace tt_graid {

GraidMemory::GraidMemory(GraidDevice& device)
    : device_(device)
    , memory_manager_(nullptr)
    , host_allocated_(0)
    , dram_allocated_(0)
    , l1_allocated_(0)
{
}

GraidMemory::~GraidMemory() {
    // TODO: Free all allocated buffers
}

BufferDescriptor GraidMemory::AllocateHost(size_t size, uint32_t alignment) {
    BufferDescriptor buffer;
    buffer.size = AlignSize(size, alignment);
    buffer.alignment = alignment;
    buffer.location = MemoryLocation::HOST;

    // Allocate aligned host memory
    #ifdef _WIN32
        buffer.ptr = _aligned_malloc(buffer.size, alignment);
    #else
        if (posix_memalign(&buffer.ptr, alignment, buffer.size) != 0) {
            buffer.ptr = nullptr;
        }
    #endif

    if (buffer.ptr != nullptr) {
        buffer.is_allocated = true;
        host_allocated_ += buffer.size;
    }

    return buffer;
}

BufferDescriptor GraidMemory::AllocateDRAM(size_t size, bool interleaved) {
    BufferDescriptor buffer;
    buffer.size = size;
    buffer.location = MemoryLocation::DEVICE_DRAM;
    buffer.is_allocated = false;

    // TODO: Allocate device DRAM with TT-Metal API
    // For stub: just track the size
    dram_allocated_ += size;

    return buffer;
}

BufferDescriptor GraidMemory::AllocateL1(size_t size, const CoreCoord& core) {
    BufferDescriptor buffer;
    buffer.size = size;
    buffer.location = MemoryLocation::DEVICE_L1;
    buffer.core = core;
    buffer.is_allocated = false;

    // Validate size (max 1 MB per core)
    const size_t max_l1_size = 1024 * 1024;
    if (size > max_l1_size) {
        throw std::runtime_error("L1 allocation exceeds 1 MB per core limit");
    }

    // TODO: Allocate L1 SRAM with TT-Metal API
    l1_allocated_ += size;

    return buffer;
}

std::vector<BufferDescriptor> GraidMemory::AllocateL1Sharded(
    size_t size,
    const std::vector<CoreCoord>& cores,
    size_t shard_size)
{
    std::vector<BufferDescriptor> buffers;

    if (cores.empty()) {
        return buffers;
    }

    // Calculate shard size if not specified
    if (shard_size == 0) {
        shard_size = (size + cores.size() - 1) / cores.size();
    }

    // Allocate shards
    for (const auto& core : cores) {
        auto buffer = AllocateL1(shard_size, core);
        buffers.push_back(buffer);
    }

    return buffers;
}

void GraidMemory::Free(BufferDescriptor& buffer) {
    if (!buffer.is_allocated && buffer.ptr == nullptr) {
        return;
    }

    if (buffer.location == MemoryLocation::HOST && buffer.ptr != nullptr) {
        #ifdef _WIN32
            _aligned_free(buffer.ptr);
        #else
            free(buffer.ptr);
        #endif
        host_allocated_ -= buffer.size;
    } else if (buffer.location == MemoryLocation::DEVICE_DRAM) {
        // TODO: Free DRAM with TT-Metal API
        dram_allocated_ -= buffer.size;
    } else if (buffer.location == MemoryLocation::DEVICE_L1) {
        // TODO: Free L1 SRAM with TT-Metal API
        l1_allocated_ -= buffer.size;
    }

    buffer.ptr = nullptr;
    buffer.is_allocated = false;
    buffer.size = 0;
}

void GraidMemory::Free(std::vector<BufferDescriptor>& buffers) {
    for (auto& buffer : buffers) {
        Free(buffer);
    }
    buffers.clear();
}

bool GraidMemory::CopyHostToDRAM(
    BufferDescriptor& dst,
    const BufferDescriptor& src,
    size_t size,
    bool async)
{
    if (!ValidateBuffer(src, size) || !ValidateBuffer(dst, size)) {
        return false;
    }

    // TODO: Use TT-Metal DMA API
    // For stub: just return success
    return true;
}

bool GraidMemory::CopyDRAMToHost(
    BufferDescriptor& dst,
    const BufferDescriptor& src,
    size_t size,
    bool async)
{
    if (!ValidateBuffer(src, size) || !ValidateBuffer(dst, size)) {
        return false;
    }

    // TODO: Use TT-Metal DMA API
    return true;
}

bool GraidMemory::CopyDRAMToL1(
    BufferDescriptor& dst,
    const BufferDescriptor& src,
    size_t size)
{
    if (!ValidateBuffer(src, size) || !ValidateBuffer(dst, size)) {
        return false;
    }

    // TODO: Use TT-Metal NoC transfer
    return true;
}

bool GraidMemory::CopyL1ToDRAM(
    BufferDescriptor& dst,
    const BufferDescriptor& src,
    size_t size)
{
    if (!ValidateBuffer(src, size) || !ValidateBuffer(dst, size)) {
        return false;
    }

    // TODO: Use TT-Metal NoC transfer
    return true;
}

bool GraidMemory::MulticastDRAMToL1(
    std::vector<BufferDescriptor>& dst,
    const BufferDescriptor& src,
    size_t size)
{
    if (!ValidateBuffer(src, size)) {
        return false;
    }

    for (const auto& dest_buf : dst) {
        if (!ValidateBuffer(dest_buf, size)) {
            return false;
        }
    }

    // TODO: Use TT-Metal multicast operation
    return true;
}

bool GraidMemory::Memset(BufferDescriptor& buffer, uint8_t value, size_t size) {
    if (!ValidateBuffer(buffer, size)) {
        return false;
    }

    if (buffer.location == MemoryLocation::HOST && buffer.ptr != nullptr) {
        std::memset(buffer.ptr, value, size);
        return true;
    }

    // TODO: Device memset with TT-Metal
    return true;
}

bool GraidMemory::Synchronize() {
    // TODO: Synchronize all pending transfers
    return device_.Synchronize();
}

std::pair<size_t, size_t> GraidMemory::GetMemoryUsage(MemoryLocation location) const {
    const auto& props = device_.GetProperties();

    switch (location) {
        case MemoryLocation::HOST:
            // Return allocated and total (assume system has enough)
            return {host_allocated_, host_allocated_ * 10};

        case MemoryLocation::DEVICE_DRAM:
            return {dram_allocated_, props.dram_size};

        case MemoryLocation::DEVICE_L1:
            return {l1_allocated_, props.total_sram_size};

        default:
            return {0, 0};
    }
}

float GraidMemory::GetMemoryBandwidth() const {
    // TODO: Measure actual bandwidth
    const auto& props = device_.GetProperties();
    return static_cast<float>(props.dram_bandwidth_gbps);
}

bool GraidMemory::ValidateBuffer(const BufferDescriptor& buffer, size_t required_size) const {
    if (buffer.size < required_size) {
        return false;
    }

    if (buffer.location == MemoryLocation::HOST && buffer.ptr == nullptr) {
        return false;
    }

    return true;
}

size_t GraidMemory::AlignSize(size_t size, uint32_t alignment) {
    return ((size + alignment - 1) / alignment) * alignment;
}

} // namespace tt_graid
