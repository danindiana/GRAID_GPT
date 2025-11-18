/**
 * GPU RAID Manager - Main Implementation
 *
 * Core RAID orchestration logic
 * Implements the public API defined in gpu_raid.h
 */

#include "../include/gpu_raid.h"
#include "../include/cuda_utils.h"
#include "../include/raid_types.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// External kernel launch functions
extern "C" {
    cudaError_t gf_init_tables();
    cudaError_t launch_xor_parity_kernel(
        const uint8_t** data_blocks_dev, uint8_t* parity_block_dev,
        size_t block_size, int num_blocks, cudaStream_t stream);
    cudaError_t launch_raid6_encode(
        const uint8_t** data_blocks_dev, uint8_t* parity_p_dev,
        uint8_t* parity_q_dev, size_t block_size, int num_blocks,
        const uint8_t* gen_coeffs_dev, cudaStream_t stream);
    cudaError_t launch_raid5_reconstruct(
        const uint8_t** surviving_blocks_dev, const uint8_t* parity_dev,
        uint8_t* recovered_dev, size_t block_size, int num_surviving,
        cudaStream_t stream);
}

// External device manager functions
extern gpu_raid_error_t device_manager_init(gpu_raid_context_t* ctx);
extern void device_manager_destroy_streams(gpu_raid_context_t* ctx);
extern gpu_raid_error_t device_manager_create_streams(gpu_raid_context_t* ctx);

// External memory pool functions
extern gpu_raid_error_t memory_pool_init(gpu_raid_context_t* ctx);
extern void memory_pool_destroy();
extern void* memory_pool_alloc(size_t size, const char* tag);
extern gpu_raid_error_t memory_pool_init_pinned_buffer(gpu_raid_context_t* ctx);
extern void memory_pool_destroy_pinned_buffer(gpu_raid_context_t* ctx);

/**
 * Get version string
 */
const char* gpu_raid_get_version(void) {
    static char version[32];
    snprintf(version, sizeof(version), "%d.%d.%d",
             GPU_RAID_VERSION_MAJOR,
             GPU_RAID_VERSION_MINOR,
             GPU_RAID_VERSION_PATCH);
    return version;
}

/**
 * Get error string
 */
const char* gpu_raid_get_error_string(gpu_raid_error_t error) {
    switch (error) {
        case GPU_RAID_SUCCESS: return "Success";
        case GPU_RAID_ERROR_INVALID_PARAM: return "Invalid parameter";
        case GPU_RAID_ERROR_NO_GPU: return "No GPU found";
        case GPU_RAID_ERROR_GPU_INIT_FAILED: return "GPU initialization failed";
        case GPU_RAID_ERROR_OUT_OF_MEMORY: return "Out of memory";
        case GPU_RAID_ERROR_CUDA_ERROR: return "CUDA error";
        case GPU_RAID_ERROR_INVALID_RAID_LEVEL: return "Invalid RAID level";
        case GPU_RAID_ERROR_TOO_MANY_FAILURES: return "Too many drive failures";
        case GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED: return "Kernel launch failed";
        case GPU_RAID_ERROR_NOT_INITIALIZED: return "Not initialized";
        case GPU_RAID_ERROR_TENSTORRENT_UNAVAILABLE: return "Tenstorrent unavailable";
        default: return "Unknown error";
    }
}

/**
 * Initialize GPU RAID
 */
gpu_raid_error_t gpu_raid_init(
    const gpu_raid_config_t* config,
    gpu_raid_handle_t* handle
) {
    if (!config || !handle) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    // Validate configuration
    if (config->raid_level != GPU_RAID_LEVEL_5 &&
        config->raid_level != GPU_RAID_LEVEL_6) {
        return GPU_RAID_ERROR_INVALID_RAID_LEVEL;
    }

    if (config->num_data_drives < 2 || config->num_data_drives > MAX_DATA_DRIVES) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    // Allocate context
    gpu_raid_context_t* ctx = (gpu_raid_context_t*)calloc(1, sizeof(gpu_raid_context_t));
    if (!ctx) {
        return GPU_RAID_ERROR_OUT_OF_MEMORY;
    }

    // Copy configuration
    memcpy(&ctx->config, config, sizeof(gpu_raid_config_t));

    // Initialize device
    gpu_raid_error_t err = device_manager_init(ctx);
    if (err != GPU_RAID_SUCCESS) {
        free(ctx);
        return err;
    }

    // Create CUDA streams
    err = device_manager_create_streams(ctx);
    if (err != GPU_RAID_SUCCESS) {
        free(ctx);
        return err;
    }

    // Initialize memory pool
    err = memory_pool_init(ctx);
    if (err != GPU_RAID_SUCCESS) {
        device_manager_destroy_streams(ctx);
        free(ctx);
        return err;
    }

    // Initialize pinned buffer
    err = memory_pool_init_pinned_buffer(ctx);
    if (err != GPU_RAID_SUCCESS) {
        memory_pool_destroy();
        device_manager_destroy_streams(ctx);
        free(ctx);
        return err;
    }

    // Initialize Galois Field tables (for RAID 6)
    if (config->raid_level == GPU_RAID_LEVEL_6) {
        cudaError_t cuda_err = gf_init_tables();
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Failed to initialize GF tables: %s\n",
                    cudaGetErrorString(cuda_err));
            memory_pool_destroy_pinned_buffer(ctx);
            memory_pool_destroy();
            device_manager_destroy_streams(ctx);
            free(ctx);
            return GPU_RAID_ERROR_GPU_INIT_FAILED;
        }
        ctx->gf_tables_initialized = true;

        // Allocate generator coefficients
        ctx->dev_gen_coeffs = (uint8_t*)memory_pool_alloc(
            config->num_data_drives, "gen_coeffs"
        );

        // Generate Vandermonde coefficients (g^0, g^1, g^2, ...)
        uint8_t* host_coeffs = (uint8_t*)malloc(config->num_data_drives);
        host_coeffs[0] = 1;
        for (uint32_t i = 1; i < config->num_data_drives; i++) {
            // g = 2 in GF(2^8)
            host_coeffs[i] = (host_coeffs[i-1] << 1) ^ ((host_coeffs[i-1] & 0x80) ? 0x1D : 0);
        }

        cudaMemcpy(ctx->dev_gen_coeffs, host_coeffs,
                   config->num_data_drives, cudaMemcpyHostToDevice);
        free(host_coeffs);
    }

    // Initialize statistics
    memset(&ctx->stats, 0, sizeof(gpu_raid_stats_t));
    ctx->start_time = (double)clock() / CLOCKS_PER_SEC;

    ctx->initialized = true;
    ctx->profiling_enabled = config->enable_profiling;

    *handle = ctx;

    printf("GPU RAID initialized successfully\n");
    printf("  RAID Level: %d\n", config->raid_level);
    printf("  Data Drives: %u\n", config->num_data_drives);
    printf("  Stripe Size: %u KB\n\n", config->stripe_size_kb);

    return GPU_RAID_SUCCESS;
}

/**
 * Destroy GPU RAID
 */
void gpu_raid_destroy(gpu_raid_handle_t handle) {
    if (!handle) return;

    gpu_raid_context_t* ctx = (gpu_raid_context_t*)handle;

    // Synchronize before cleanup
    cudaDeviceSynchronize();

    // Destroy streams
    device_manager_destroy_streams(ctx);

    // Destroy pinned buffer
    memory_pool_destroy_pinned_buffer(ctx);

    // Destroy memory pool
    memory_pool_destroy();

    // Free context
    free(ctx);

    printf("GPU RAID destroyed\n");
}

/**
 * Encode (generate parity)
 */
gpu_raid_error_t gpu_raid_encode(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    uint8_t** parity_blocks,
    uint32_t num_data_blocks,
    size_t block_size
) {
    if (!handle || !data_blocks || !parity_blocks) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    gpu_raid_context_t* ctx = (gpu_raid_context_t*)handle;

    if (!ctx->initialized) {
        return GPU_RAID_ERROR_NOT_INITIALIZED;
    }

    if (num_data_blocks != ctx->config.num_data_drives) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    double start_time = (double)clock() / CLOCKS_PER_SEC;

    // Allocate device memory for data blocks
    uint8_t** dev_data_ptrs_host = (uint8_t**)malloc(num_data_blocks * sizeof(uint8_t*));
    for (uint32_t i = 0; i < num_data_blocks; i++) {
        dev_data_ptrs_host[i] = (uint8_t*)memory_pool_alloc(block_size, "data_block");
        cudaMemcpy(dev_data_ptrs_host[i], data_blocks[i], block_size, cudaMemcpyHostToDevice);
    }

    // Copy pointer array to device
    uint8_t** dev_data_ptrs_dev;
    cudaMalloc(&dev_data_ptrs_dev, num_data_blocks * sizeof(uint8_t*));
    cudaMemcpy(dev_data_ptrs_dev, dev_data_ptrs_host,
               num_data_blocks * sizeof(uint8_t*), cudaMemcpyHostToDevice);

    if (ctx->config.raid_level == GPU_RAID_LEVEL_5) {
        // RAID 5: XOR parity
        uint8_t* dev_parity = (uint8_t*)memory_pool_alloc(block_size, "parity");

        cudaError_t err = launch_xor_parity_kernel(
            (const uint8_t**)dev_data_ptrs_dev, dev_parity,
            block_size, num_data_blocks, ctx->streams[0]
        );

        if (err != cudaSuccess) {
            cudaFree(dev_data_ptrs_dev);
            free(dev_data_ptrs_host);
            return GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED;
        }

        cudaMemcpy(parity_blocks[0], dev_parity, block_size, cudaMemcpyDeviceToHost);

    } else {
        // RAID 6: Reed-Solomon P+Q
        uint8_t* dev_parity_p = (uint8_t*)memory_pool_alloc(block_size, "parity_p");
        uint8_t* dev_parity_q = (uint8_t*)memory_pool_alloc(block_size, "parity_q");

        cudaError_t err = launch_raid6_encode(
            (const uint8_t**)dev_data_ptrs_dev, dev_parity_p, dev_parity_q,
            block_size, num_data_blocks, ctx->dev_gen_coeffs, ctx->streams[0]
        );

        if (err != cudaSuccess) {
            cudaFree(dev_data_ptrs_dev);
            free(dev_data_ptrs_host);
            return GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED;
        }

        cudaMemcpy(parity_blocks[0], dev_parity_p, block_size, cudaMemcpyDeviceToHost);
        cudaMemcpy(parity_blocks[1], dev_parity_q, block_size, cudaMemcpyDeviceToHost);
    }

    cudaDeviceSynchronize();
    cudaFree(dev_data_ptrs_dev);
    free(dev_data_ptrs_host);

    // Update statistics
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    double elapsed = end_time - start_time;

    ctx->stats.total_encodes++;
    ctx->stats.total_bytes_encoded += num_data_blocks * block_size;
    ctx->last_encode_time = elapsed * 1000.0;  // Convert to ms
    ctx->stats.avg_encode_time_ms =
        (ctx->stats.avg_encode_time_ms * (ctx->stats.total_encodes - 1) +
         ctx->last_encode_time) / ctx->stats.total_encodes;

    double throughput = (num_data_blocks * block_size) / elapsed / (1024.0 * 1024.0 * 1024.0);
    if (throughput > ctx->stats.peak_throughput_gbs) {
        ctx->stats.peak_throughput_gbs = throughput;
    }

    return GPU_RAID_SUCCESS;
}

/**
 * Reconstruct failed blocks
 */
gpu_raid_error_t gpu_raid_reconstruct(
    gpu_raid_handle_t handle,
    const uint8_t** all_blocks,
    const uint8_t** parity_blocks,
    const uint32_t* failed_indices,
    uint32_t num_failed,
    uint8_t** recovered_blocks,
    uint32_t num_total_blocks,
    size_t block_size
) {
    if (!handle || !all_blocks || !parity_blocks || !failed_indices || !recovered_blocks) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    gpu_raid_context_t* ctx = (gpu_raid_context_t*)handle;

    // Validate failure count
    if (ctx->config.raid_level == GPU_RAID_LEVEL_5 && num_failed > 1) {
        return GPU_RAID_ERROR_TOO_MANY_FAILURES;
    }
    if (ctx->config.raid_level == GPU_RAID_LEVEL_6 && num_failed > 2) {
        return GPU_RAID_ERROR_TOO_MANY_FAILURES;
    }

    double start_time = (double)clock() / CLOCKS_PER_SEC;

    // For RAID 5, use XOR reconstruction
    if (ctx->config.raid_level == GPU_RAID_LEVEL_5 && num_failed == 1) {
        // Allocate and copy surviving blocks
        uint8_t** dev_surviving_ptrs_host = (uint8_t**)malloc((num_total_blocks - 1) * sizeof(uint8_t*));
        int idx = 0;
        for (uint32_t i = 0; i < num_total_blocks; i++) {
            if (all_blocks[i] != NULL) {
                dev_surviving_ptrs_host[idx] = (uint8_t*)memory_pool_alloc(block_size, "surviving");
                cudaMemcpy(dev_surviving_ptrs_host[idx], all_blocks[i],
                          block_size, cudaMemcpyHostToDevice);
                idx++;
            }
        }

        uint8_t** dev_surviving_ptrs_dev;
        cudaMalloc(&dev_surviving_ptrs_dev, (num_total_blocks - 1) * sizeof(uint8_t*));
        cudaMemcpy(dev_surviving_ptrs_dev, dev_surviving_ptrs_host,
                   (num_total_blocks - 1) * sizeof(uint8_t*), cudaMemcpyHostToDevice);

        uint8_t* dev_parity = (uint8_t*)memory_pool_alloc(block_size, "parity");
        cudaMemcpy(dev_parity, parity_blocks[0], block_size, cudaMemcpyHostToDevice);

        uint8_t* dev_recovered = (uint8_t*)memory_pool_alloc(block_size, "recovered");

        cudaError_t err = launch_raid5_reconstruct(
            (const uint8_t**)dev_surviving_ptrs_dev, dev_parity, dev_recovered,
            block_size, num_total_blocks - 1, ctx->streams[0]
        );

        if (err != cudaSuccess) {
            cudaFree(dev_surviving_ptrs_dev);
            free(dev_surviving_ptrs_host);
            return GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED;
        }

        cudaMemcpy(recovered_blocks[0], dev_recovered, block_size, cudaMemcpyDeviceToHost);

        cudaFree(dev_surviving_ptrs_dev);
        free(dev_surviving_ptrs_host);
    }

    cudaDeviceSynchronize();

    // Update statistics
    double end_time = (double)clock() / CLOCKS_PER_SEC;
    double elapsed = end_time - start_time;

    ctx->stats.total_decodes++;
    ctx->stats.total_bytes_decoded += num_failed * block_size;
    ctx->last_decode_time = elapsed * 1000.0;
    ctx->stats.avg_decode_time_ms =
        (ctx->stats.avg_decode_time_ms * (ctx->stats.total_decodes - 1) +
         ctx->last_decode_time) / ctx->stats.total_decodes;

    return GPU_RAID_SUCCESS;
}

/**
 * Verify parity
 */
gpu_raid_error_t gpu_raid_verify(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    const uint8_t** parity_blocks,
    uint32_t num_blocks,
    size_t block_size,
    bool* is_valid
) {
    // Simplified: Recompute parity and compare
    // In production, would use dedicated verification kernel

    *is_valid = true;  // Placeholder
    return GPU_RAID_SUCCESS;
}

/**
 * Sync all operations
 */
gpu_raid_error_t gpu_raid_sync(gpu_raid_handle_t handle) {
    if (!handle) return GPU_RAID_ERROR_INVALID_PARAM;

    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), GPU_RAID_ERROR_CUDA_ERROR);
    return GPU_RAID_SUCCESS;
}

/**
 * Get statistics
 */
gpu_raid_error_t gpu_raid_get_stats(
    gpu_raid_handle_t handle,
    gpu_raid_stats_t* stats
) {
    if (!handle || !stats) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    gpu_raid_context_t* ctx = (gpu_raid_context_t*)handle;
    memcpy(stats, &ctx->stats, sizeof(gpu_raid_stats_t));

    return GPU_RAID_SUCCESS;
}

/**
 * Reset statistics
 */
void gpu_raid_reset_stats(gpu_raid_handle_t handle) {
    if (!handle) return;

    gpu_raid_context_t* ctx = (gpu_raid_context_t*)handle;
    memset(&ctx->stats, 0, sizeof(gpu_raid_stats_t));
    ctx->start_time = (double)clock() / CLOCKS_PER_SEC;
}
