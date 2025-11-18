/**
 * DMA Buffer Management for GPU RAID
 *
 * Implements zero-copy DMA buffers for efficient kernel-GPU data transfer
 * Uses CUDA managed memory and pinned host memory
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <map>
#include <mutex>

#include "../include/cuda_utils.h"

/**
 * DMA Buffer descriptor
 */
struct DMABuffer {
    void *host_ptr;         // Pinned host memory
    void *device_ptr;       // GPU device memory
    size_t size;            // Buffer size
    bool is_mapped;         // Using zero-copy mapped memory
    cudaStream_t stream;    // Associated CUDA stream
    uint64_t handle;        // Unique handle
};

static std::map<uint64_t, DMABuffer*> g_dma_buffers;
static std::mutex g_dma_mutex;
static uint64_t g_next_handle = 1;

/**
 * Allocate DMA buffer with zero-copy capability
 */
extern "C" uint64_t dma_buffer_alloc(size_t size, bool zero_copy) {
    DMABuffer *buf = new DMABuffer;
    if (!buf) {
        return 0;
    }

    memset(buf, 0, sizeof(DMABuffer));
    buf->size = size;
    buf->is_mapped = zero_copy;

    cudaError_t err;

    if (zero_copy) {
        // Allocate zero-copy mapped memory
        // This allows CPU and GPU to access the same physical memory
        err = cudaHostAlloc(&buf->host_ptr, size, cudaHostAllocMapped);
        if (err != cudaSuccess) {
            fprintf(stderr, "DMA: cudaHostAlloc failed: %s\n", cudaGetErrorString(err));
            delete buf;
            return 0;
        }

        // Get device pointer to the same memory
        err = cudaHostGetDevicePointer(&buf->device_ptr, buf->host_ptr, 0);
        if (err != cudaSuccess) {
            fprintf(stderr, "DMA: cudaHostGetDevicePointer failed: %s\n",
                    cudaGetErrorString(err));
            cudaFreeHost(buf->host_ptr);
            delete buf;
            return 0;
        }

        printf("DMA: Allocated zero-copy buffer: %zu bytes (host=%p, device=%p)\n",
               size, buf->host_ptr, buf->device_ptr);
    } else {
        // Allocate pinned host memory + separate device memory
        err = cudaMallocHost(&buf->host_ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "DMA: cudaMallocHost failed: %s\n", cudaGetErrorString(err));
            delete buf;
            return 0;
        }

        err = cudaMalloc(&buf->device_ptr, size);
        if (err != cudaSuccess) {
            fprintf(stderr, "DMA: cudaMalloc failed: %s\n", cudaGetErrorString(err));
            cudaFreeHost(buf->host_ptr);
            delete buf;
            return 0;
        }

        printf("DMA: Allocated pinned+device buffer: %zu bytes\n", size);
    }

    // Create dedicated stream for async operations
    err = cudaStreamCreate(&buf->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "DMA: cudaStreamCreate failed: %s\n", cudaGetErrorString(err));
        if (zero_copy) {
            cudaFreeHost(buf->host_ptr);
        } else {
            cudaFreeHost(buf->host_ptr);
            cudaFree(buf->device_ptr);
        }
        delete buf;
        return 0;
    }

    // Generate handle and register buffer
    std::lock_guard<std::mutex> lock(g_dma_mutex);
    buf->handle = g_next_handle++;
    g_dma_buffers[buf->handle] = buf;

    return buf->handle;
}

/**
 * Free DMA buffer
 */
extern "C" int dma_buffer_free(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    DMABuffer *buf = it->second;

    // Synchronize stream before freeing
    cudaStreamSynchronize(buf->stream);
    cudaStreamDestroy(buf->stream);

    if (buf->is_mapped) {
        cudaFreeHost(buf->host_ptr);
    } else {
        cudaFreeHost(buf->host_ptr);
        cudaFree(buf->device_ptr);
    }

    delete buf;
    g_dma_buffers.erase(it);

    return 0;
}

/**
 * Get host pointer for CPU access
 */
extern "C" void* dma_buffer_get_host_ptr(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return nullptr;
    }

    return it->second->host_ptr;
}

/**
 * Get device pointer for GPU access
 */
extern "C" void* dma_buffer_get_device_ptr(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return nullptr;
    }

    return it->second->device_ptr;
}

/**
 * Async copy from host to device
 * Only needed for non-zero-copy buffers
 */
extern "C" int dma_buffer_h2d_async(uint64_t handle, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    DMABuffer *buf = it->second;

    // Zero-copy buffers don't need explicit transfer
    if (buf->is_mapped) {
        return 0;
    }

    if (offset + size > buf->size) {
        return -EINVAL;
    }

    uint8_t *host_base = (uint8_t*)buf->host_ptr + offset;
    uint8_t *device_base = (uint8_t*)buf->device_ptr + offset;

    cudaError_t err = cudaMemcpyAsync(device_base, host_base, size,
                                      cudaMemcpyHostToDevice, buf->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "DMA: H2D async copy failed: %s\n", cudaGetErrorString(err));
        return -EIO;
    }

    return 0;
}

/**
 * Async copy from device to host
 */
extern "C" int dma_buffer_d2h_async(uint64_t handle, size_t offset, size_t size) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    DMABuffer *buf = it->second;

    if (buf->is_mapped) {
        return 0;  // No copy needed
    }

    if (offset + size > buf->size) {
        return -EINVAL;
    }

    uint8_t *host_base = (uint8_t*)buf->host_ptr + offset;
    uint8_t *device_base = (uint8_t*)buf->device_ptr + offset;

    cudaError_t err = cudaMemcpyAsync(host_base, device_base, size,
                                      cudaMemcpyDeviceToHost, buf->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "DMA: D2H async copy failed: %s\n", cudaGetErrorString(err));
        return -EIO;
    }

    return 0;
}

/**
 * Synchronize DMA buffer operations
 */
extern "C" int dma_buffer_sync(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    cudaError_t err = cudaStreamSynchronize(it->second->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "DMA: Stream sync failed: %s\n", cudaGetErrorString(err));
        return -EIO;
    }

    return 0;
}

/**
 * Get stream for custom operations
 */
extern "C" cudaStream_t dma_buffer_get_stream(uint64_t handle) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return nullptr;
    }

    return it->second->stream;
}

/**
 * Check if zero-copy is supported
 */
extern "C" bool dma_zero_copy_supported(void) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Check for unified addressing and mapped pinned memory support
    return prop.canMapHostMemory && prop.unifiedAddressing;
}

/**
 * Get DMA buffer info
 */
extern "C" int dma_buffer_get_info(uint64_t handle, size_t *size, bool *is_zero_copy) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    DMABuffer *buf = it->second;
    if (size) *size = buf->size;
    if (is_zero_copy) *is_zero_copy = buf->is_mapped;

    return 0;
}

/**
 * Get statistics
 */
extern "C" void dma_buffer_stats(size_t *total_allocated, size_t *num_buffers,
                                 size_t *zero_copy_count) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    size_t total = 0;
    size_t zc_count = 0;

    for (const auto& pair : g_dma_buffers) {
        total += pair.second->size;
        if (pair.second->is_mapped) {
            zc_count++;
        }
    }

    if (total_allocated) *total_allocated = total;
    if (num_buffers) *num_buffers = g_dma_buffers.size();
    if (zero_copy_count) *zero_copy_count = zc_count;
}

/**
 * Batch allocate multiple DMA buffers
 */
extern "C" int dma_buffer_batch_alloc(size_t block_size, int num_blocks,
                                     bool zero_copy, uint64_t *handles) {
    for (int i = 0; i < num_blocks; i++) {
        handles[i] = dma_buffer_alloc(block_size, zero_copy);
        if (handles[i] == 0) {
            // Allocation failed, free previously allocated buffers
            for (int j = 0; j < i; j++) {
                dma_buffer_free(handles[j]);
            }
            return -ENOMEM;
        }
    }

    return 0;
}

/**
 * Batch free multiple DMA buffers
 */
extern "C" void dma_buffer_batch_free(uint64_t *handles, int num_buffers) {
    for (int i = 0; i < num_buffers; i++) {
        dma_buffer_free(handles[i]);
    }
}

/**
 * Memory prefetch for managed memory (CUDA 8.0+)
 */
extern "C" int dma_buffer_prefetch(uint64_t handle, int device_id) {
    std::lock_guard<std::mutex> lock(g_dma_mutex);

    auto it = g_dma_buffers.find(handle);
    if (it == g_dma_buffers.end()) {
        return -EINVAL;
    }

    DMABuffer *buf = it->second;

#if CUDART_VERSION >= 8000
    // Prefetch to GPU for faster first access
    cudaError_t err = cudaMemPrefetchAsync(buf->device_ptr, buf->size,
                                          device_id, buf->stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "DMA: Prefetch failed: %s\n", cudaGetErrorString(err));
        return -EIO;
    }
#endif

    return 0;
}
