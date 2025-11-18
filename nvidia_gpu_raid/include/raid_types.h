/**
 * RAID Internal Types and Structures
 *
 * Internal data structures for GPU RAID implementation
 */

#ifndef RAID_TYPES_H
#define RAID_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include "gpu_raid.h"

#ifdef __cplusplus
extern "C" {
#endif

// Maximum supported values
#define MAX_DATA_DRIVES 16
#define MAX_PARITY_DRIVES 2
#define MAX_CUDA_STREAMS 8
#define MAX_STRIPE_SIZE_MB 16

// Memory alignment
#define GPU_RAID_ALIGNMENT 128  // 128-byte alignment for best performance

// Internal GPU RAID context
typedef struct gpu_raid_context {
    // Configuration
    gpu_raid_config_t config;

    // GPU device information
    int cuda_device_id;
    gpu_raid_device_type_t device_type;
    cudaDeviceProp device_props;
    uint32_t cuda_cores;
    size_t total_memory;

    // CUDA resources
    cudaStream_t streams[MAX_CUDA_STREAMS];
    uint32_t num_streams;

    // Device memory pools
    void* device_memory_pool;
    size_t pool_size;
    size_t pool_used;

    // Pinned host memory
    void* pinned_buffer;
    size_t pinned_buffer_size;

    // Generator coefficients for RAID 6 (device memory)
    uint8_t* dev_gen_coeffs;

    // Performance statistics
    gpu_raid_stats_t stats;
    double start_time;
    double last_encode_time;
    double last_decode_time;

    // Galois Field tables initialized
    bool gf_tables_initialized;

    // Tenstorrent integration
    bool tenstorrent_available;
    void* tt_context;  // Opaque Tenstorrent context

    // Flags
    bool initialized;
    bool profiling_enabled;
} gpu_raid_context_t;

// Stripe information
typedef struct {
    uint32_t stripe_id;
    uint32_t num_blocks;
    size_t block_size;
    uint8_t** data_blocks_host;
    uint8_t** data_blocks_device;
    uint8_t* parity_p_device;
    uint8_t* parity_q_device;
} stripe_info_t;

// Kernel launch parameters
typedef struct {
    int threads_per_block;
    int blocks_per_grid;
    size_t shared_memory_bytes;
    cudaStream_t stream;
} kernel_params_t;

// Device memory allocation record
typedef struct {
    void* ptr;
    size_t size;
    bool in_use;
    const char* tag;
} device_allocation_t;

#ifdef __cplusplus
}
#endif

#endif // RAID_TYPES_H
