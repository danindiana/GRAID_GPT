/**
 * GPU RAID - Public API Header
 *
 * GPU-accelerated RAID implementation for NVIDIA GPUs
 * Supports RAID 5 (single parity) and RAID 6 (dual parity)
 *
 * Target Hardware: RTX 3080, RTX 3060, Quadro RTX 4000
 *
 * Example Usage:
 * ```c
 * gpu_raid_config_t config = {
 *     .raid_level = GPU_RAID_LEVEL_6,
 *     .num_data_drives = 6,
 *     .stripe_size_kb = 256,
 *     .gpu_device_id = 0
 * };
 *
 * gpu_raid_handle_t handle;
 * gpu_raid_init(&config, &handle);
 *
 * // Encode data
 * gpu_raid_encode(handle, data_blocks, parity_blocks);
 *
 * // Cleanup
 * gpu_raid_destroy(handle);
 * ```
 */

#ifndef GPU_RAID_H
#define GPU_RAID_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define GPU_RAID_VERSION_MAJOR 0
#define GPU_RAID_VERSION_MINOR 1
#define GPU_RAID_VERSION_PATCH 0

// Error codes
typedef enum {
    GPU_RAID_SUCCESS = 0,
    GPU_RAID_ERROR_INVALID_PARAM = -1,
    GPU_RAID_ERROR_NO_GPU = -2,
    GPU_RAID_ERROR_GPU_INIT_FAILED = -3,
    GPU_RAID_ERROR_OUT_OF_MEMORY = -4,
    GPU_RAID_ERROR_CUDA_ERROR = -5,
    GPU_RAID_ERROR_INVALID_RAID_LEVEL = -6,
    GPU_RAID_ERROR_TOO_MANY_FAILURES = -7,
    GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED = -8,
    GPU_RAID_ERROR_NOT_INITIALIZED = -9,
    GPU_RAID_ERROR_TENSTORRENT_UNAVAILABLE = -10
} gpu_raid_error_t;

// RAID levels
typedef enum {
    GPU_RAID_LEVEL_5 = 5,  // Single parity (XOR)
    GPU_RAID_LEVEL_6 = 6   // Dual parity (Reed-Solomon P+Q)
} gpu_raid_level_t;

// GPU device type
typedef enum {
    GPU_RAID_DEVICE_AUTO = 0,       // Auto-detect
    GPU_RAID_DEVICE_RTX_3080 = 1,
    GPU_RAID_DEVICE_RTX_3060 = 2,
    GPU_RAID_DEVICE_QUADRO_4000 = 3,
    GPU_RAID_DEVICE_GENERIC = 99
} gpu_raid_device_type_t;

// Configuration structure
typedef struct {
    gpu_raid_level_t raid_level;        // RAID level (5 or 6)
    uint32_t num_data_drives;            // Number of data drives
    uint32_t stripe_size_kb;             // Stripe size in KB
    int gpu_device_id;                   // CUDA device ID (-1 for auto)
    gpu_raid_device_type_t device_type;  // GPU model (AUTO for detect)
    bool enable_tenstorrent;             // Enable TT integration (experimental)
    size_t memory_pool_size_mb;          // GPU memory pool size
    uint32_t num_streams;                // Number of CUDA streams
    bool enable_profiling;               // Enable performance profiling
} gpu_raid_config_t;

// Opaque handle
typedef struct gpu_raid_context* gpu_raid_handle_t;

// Statistics structure
typedef struct {
    uint64_t total_encodes;              // Total encode operations
    uint64_t total_decodes;              // Total decode/rebuild operations
    uint64_t total_bytes_encoded;        // Total bytes encoded
    uint64_t total_bytes_decoded;        // Total bytes decoded
    double avg_encode_time_ms;           // Average encode time
    double avg_decode_time_ms;           // Average decode time
    double peak_throughput_gbs;          // Peak throughput achieved
    uint32_t gpu_utilization_percent;    // Current GPU utilization
    float gpu_temperature_c;             // GPU temperature
    uint32_t gpu_power_watts;            // GPU power consumption
} gpu_raid_stats_t;

/**
 * Initialize GPU RAID system
 *
 * @param config: Configuration parameters
 * @param handle: Output handle (must call gpu_raid_destroy when done)
 * @return Error code (GPU_RAID_SUCCESS on success)
 */
gpu_raid_error_t gpu_raid_init(
    const gpu_raid_config_t* config,
    gpu_raid_handle_t* handle
);

/**
 * Destroy GPU RAID system and free resources
 *
 * @param handle: Handle from gpu_raid_init
 */
void gpu_raid_destroy(gpu_raid_handle_t handle);

/**
 * Encode data blocks to generate parity
 *
 * RAID 5: Generates 1 parity block
 * RAID 6: Generates 2 parity blocks (P and Q)
 *
 * @param handle: GPU RAID handle
 * @param data_blocks: Array of pointers to data blocks (host memory)
 * @param parity_blocks: Array of pointers to parity blocks (host memory)
 * @param num_data_blocks: Number of data blocks
 * @param block_size: Size of each block in bytes
 * @return Error code
 */
gpu_raid_error_t gpu_raid_encode(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    uint8_t** parity_blocks,
    uint32_t num_data_blocks,
    size_t block_size
);

/**
 * Encode data blocks (async version)
 *
 * Returns immediately; use gpu_raid_sync to wait for completion
 *
 * @param handle: GPU RAID handle
 * @param data_blocks: Array of pointers to data blocks
 * @param parity_blocks: Array of pointers to parity blocks
 * @param num_data_blocks: Number of data blocks
 * @param block_size: Size of each block
 * @param stream_id: Stream ID for async operation
 * @return Error code
 */
gpu_raid_error_t gpu_raid_encode_async(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    uint8_t** parity_blocks,
    uint32_t num_data_blocks,
    size_t block_size,
    uint32_t stream_id
);

/**
 * Reconstruct failed blocks from surviving data and parity
 *
 * RAID 5: Recovers 1 failed block
 * RAID 6: Recovers up to 2 failed blocks
 *
 * @param handle: GPU RAID handle
 * @param all_blocks: Array of all blocks (NULL for failed blocks)
 * @param parity_blocks: Array of parity blocks
 * @param failed_indices: Indices of failed blocks
 * @param num_failed: Number of failed blocks
 * @param recovered_blocks: Output recovered blocks (host memory)
 * @param num_total_blocks: Total number of data blocks
 * @param block_size: Size of each block
 * @return Error code
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
);

/**
 * Update parity after modifying a data block
 *
 * More efficient than full re-encoding when only one block changes
 *
 * @param handle: GPU RAID handle
 * @param old_parity: Current parity blocks
 * @param new_parity: Output updated parity blocks
 * @param old_data: Old data block
 * @param new_data: New data block
 * @param block_index: Index of modified block
 * @param block_size: Size of blocks
 * @return Error code
 */
gpu_raid_error_t gpu_raid_update_parity(
    gpu_raid_handle_t handle,
    const uint8_t** old_parity,
    uint8_t** new_parity,
    const uint8_t* old_data,
    const uint8_t* new_data,
    uint32_t block_index,
    size_t block_size
);

/**
 * Verify parity consistency
 *
 * Checks that current data produces the same parity
 *
 * @param handle: GPU RAID handle
 * @param data_blocks: Data blocks to verify
 * @param parity_blocks: Expected parity blocks
 * @param num_blocks: Number of data blocks
 * @param block_size: Size of each block
 * @param is_valid: Output: true if parity matches
 * @return Error code
 */
gpu_raid_error_t gpu_raid_verify(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    const uint8_t** parity_blocks,
    uint32_t num_blocks,
    size_t block_size,
    bool* is_valid
);

/**
 * Synchronize all pending GPU operations
 *
 * Waits for all async operations to complete
 *
 * @param handle: GPU RAID handle
 * @return Error code
 */
gpu_raid_error_t gpu_raid_sync(gpu_raid_handle_t handle);

/**
 * Get performance statistics
 *
 * @param handle: GPU RAID handle
 * @param stats: Output statistics structure
 * @return Error code
 */
gpu_raid_error_t gpu_raid_get_stats(
    gpu_raid_handle_t handle,
    gpu_raid_stats_t* stats
);

/**
 * Reset statistics counters
 *
 * @param handle: GPU RAID handle
 */
void gpu_raid_reset_stats(gpu_raid_handle_t handle);

/**
 * Get version string
 *
 * @return Version string (e.g., "0.1.0")
 */
const char* gpu_raid_get_version(void);

/**
 * Get error message for error code
 *
 * @param error: Error code
 * @return Human-readable error message
 */
const char* gpu_raid_get_error_string(gpu_raid_error_t error);

/**
 * Query GPU capabilities
 *
 * @param device_id: CUDA device ID
 * @param device_type: Output detected device type
 * @param cuda_cores: Output number of CUDA cores
 * @param memory_gb: Output GPU memory in GB
 * @return Error code
 */
gpu_raid_error_t gpu_raid_query_device(
    int device_id,
    gpu_raid_device_type_t* device_type,
    uint32_t* cuda_cores,
    float* memory_gb
);

/**
 * Load GPU configuration from JSON file
 *
 * @param json_path: Path to configuration JSON
 * @param config: Output configuration structure
 * @return Error code
 */
gpu_raid_error_t gpu_raid_load_config(
    const char* json_path,
    gpu_raid_config_t* config
);

#ifdef __cplusplus
}
#endif

#endif // GPU_RAID_H
