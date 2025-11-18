/**
 * GPU Memory Pool Manager
 *
 * Efficient GPU memory allocation and management
 * Pre-allocates memory pool to avoid repeated cudaMalloc calls
 */

#include "../include/gpu_raid.h"
#include "../include/cuda_utils.h"
#include "../include/raid_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_ALLOCATIONS 256

typedef struct {
    void* ptr;
    size_t size;
    size_t offset;
    bool in_use;
    const char* tag;
} allocation_record_t;

typedef struct {
    void* base_ptr;
    size_t total_size;
    size_t used;
    allocation_record_t allocations[MAX_ALLOCATIONS];
    int num_allocations;
} memory_pool_t;

static memory_pool_t g_pool = {0};

/**
 * Initialize GPU memory pool
 */
gpu_raid_error_t memory_pool_init(gpu_raid_context_t* ctx) {
    size_t pool_size = ctx->config.memory_pool_size_mb * 1024ULL * 1024ULL;

    // Check if pool fits in GPU memory
    size_t free_mem, total_mem;
    CUDA_CHECK_RETURN(cudaMemGetInfo(&free_mem, &total_mem),
                      GPU_RAID_ERROR_OUT_OF_MEMORY);

    if (pool_size > free_mem * 0.8) {  // Use max 80% of free memory
        pool_size = free_mem * 0.8;
        printf("Warning: Reducing memory pool to %.2f MB (80%% of free memory)\n",
               pool_size / (1024.0 * 1024.0));
    }

    // Allocate pool
    CUDA_CHECK_RETURN(cudaMalloc(&g_pool.base_ptr, pool_size),
                      GPU_RAID_ERROR_OUT_OF_MEMORY);

    g_pool.total_size = pool_size;
    g_pool.used = 0;
    g_pool.num_allocations = 0;

    ctx->device_memory_pool = g_pool.base_ptr;
    ctx->pool_size = pool_size;
    ctx->pool_used = 0;

    printf("GPU Memory Pool:\n");
    printf("  Allocated: %.2f MB\n", pool_size / (1024.0 * 1024.0));
    printf("  Free GPU Memory: %.2f MB / %.2f MB\n",
           free_mem / (1024.0 * 1024.0),
           total_mem / (1024.0 * 1024.0));
    printf("\n");

    return GPU_RAID_SUCCESS;
}

/**
 * Allocate from pool (aligned)
 */
void* memory_pool_alloc(size_t size, const char* tag) {
    if (g_pool.base_ptr == nullptr) {
        fprintf(stderr, "Memory pool not initialized\n");
        return nullptr;
    }

    // Align to 128 bytes for optimal performance
    size_t aligned_size = (size + GPU_RAID_ALIGNMENT - 1) & ~(GPU_RAID_ALIGNMENT - 1);

    // Check if we have space
    if (g_pool.used + aligned_size > g_pool.total_size) {
        fprintf(stderr, "Memory pool exhausted: requested %zu bytes, only %zu available\n",
                aligned_size, g_pool.total_size - g_pool.used);
        return nullptr;
    }

    // Check allocation limit
    if (g_pool.num_allocations >= MAX_ALLOCATIONS) {
        fprintf(stderr, "Maximum allocations reached\n");
        return nullptr;
    }

    // Allocate
    void* ptr = (char*)g_pool.base_ptr + g_pool.used;

    allocation_record_t* record = &g_pool.allocations[g_pool.num_allocations];
    record->ptr = ptr;
    record->size = aligned_size;
    record->offset = g_pool.used;
    record->in_use = true;
    record->tag = tag;

    g_pool.used += aligned_size;
    g_pool.num_allocations++;

    return ptr;
}

/**
 * Free pool allocation (marks as available for reuse)
 */
void memory_pool_free(void* ptr) {
    if (ptr == nullptr) return;

    for (int i = 0; i < g_pool.num_allocations; i++) {
        if (g_pool.allocations[i].ptr == ptr) {
            g_pool.allocations[i].in_use = false;
            return;
        }
    }

    fprintf(stderr, "Warning: Attempted to free unknown pointer %p\n", ptr);
}

/**
 * Reset pool (marks all allocations as free)
 */
void memory_pool_reset() {
    for (int i = 0; i < g_pool.num_allocations; i++) {
        g_pool.allocations[i].in_use = false;
    }
    // Note: We don't actually free the memory, just mark as available
}

/**
 * Compact pool (not implemented - would require copying data)
 */
void memory_pool_compact() {
    // TODO: Implement compaction if needed
    // This would involve moving allocations to eliminate gaps
}

/**
 * Get pool statistics
 */
void memory_pool_get_stats(size_t* total, size_t* used, size_t* free, int* num_allocs) {
    if (total) *total = g_pool.total_size;
    if (used) *used = g_pool.used;
    if (free) *free = g_pool.total_size - g_pool.used;
    if (num_allocs) *num_allocs = g_pool.num_allocations;
}

/**
 * Print pool status
 */
void memory_pool_print_status() {
    printf("\n=== GPU Memory Pool Status ===\n");
    printf("Total Size: %.2f MB\n", g_pool.total_size / (1024.0 * 1024.0));
    printf("Used: %.2f MB (%.1f%%)\n",
           g_pool.used / (1024.0 * 1024.0),
           100.0 * g_pool.used / g_pool.total_size);
    printf("Free: %.2f MB\n",
           (g_pool.total_size - g_pool.used) / (1024.0 * 1024.0));
    printf("Allocations: %d / %d\n", g_pool.num_allocations, MAX_ALLOCATIONS);

    if (g_pool.num_allocations > 0) {
        printf("\nActive Allocations:\n");
        for (int i = 0; i < g_pool.num_allocations; i++) {
            allocation_record_t* rec = &g_pool.allocations[i];
            if (rec->in_use) {
                printf("  [%2d] %8zu bytes @ offset %8zu (%s)\n",
                       i, rec->size, rec->offset, rec->tag ? rec->tag : "unknown");
            }
        }
    }

    printf("==============================\n\n");
}

/**
 * Destroy memory pool
 */
void memory_pool_destroy() {
    if (g_pool.base_ptr != nullptr) {
        cudaFree(g_pool.base_ptr);
        g_pool.base_ptr = nullptr;
    }

    g_pool.total_size = 0;
    g_pool.used = 0;
    g_pool.num_allocations = 0;
}

/**
 * Allocate pinned host memory (for fast transfers)
 */
void* memory_pool_alloc_pinned(size_t size) {
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);

    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate pinned memory: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

/**
 * Free pinned host memory
 */
void memory_pool_free_pinned(void* ptr) {
    if (ptr != nullptr) {
        cudaFreeHost(ptr);
    }
}

/**
 * Initialize pinned buffer for transfers
 */
gpu_raid_error_t memory_pool_init_pinned_buffer(gpu_raid_context_t* ctx) {
    // Allocate pinned buffer for host-device transfers
    size_t buffer_size = ctx->config.stripe_size_kb * 1024ULL *
                         (ctx->config.num_data_drives + 2);  // +2 for parities

    void* pinned = memory_pool_alloc_pinned(buffer_size);
    if (!pinned) {
        return GPU_RAID_ERROR_OUT_OF_MEMORY;
    }

    ctx->pinned_buffer = pinned;
    ctx->pinned_buffer_size = buffer_size;

    printf("Pinned Host Buffer: %.2f MB\n", buffer_size / (1024.0 * 1024.0));
    return GPU_RAID_SUCCESS;
}

/**
 * Destroy pinned buffer
 */
void memory_pool_destroy_pinned_buffer(gpu_raid_context_t* ctx) {
    if (ctx->pinned_buffer) {
        memory_pool_free_pinned(ctx->pinned_buffer);
        ctx->pinned_buffer = nullptr;
        ctx->pinned_buffer_size = 0;
    }
}

/**
 * Get memory usage from CUDA
 */
void memory_pool_get_cuda_memory_info(size_t* free, size_t* total) {
    cudaMemGetInfo(free, total);
}
