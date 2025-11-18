## API Reference

Complete reference for the GPU RAID C API.

### Table of Contents

- [Initialization](#initialization)
- [Encoding](#encoding)
- [Reconstruction](#reconstruction)
- [Verification](#verification)
- [Statistics](#statistics)
- [Utilities](#utilities)
- [Error Handling](#error-handling)

---

## Initialization

### gpu_raid_init

Initialize GPU RAID system.

```c
gpu_raid_error_t gpu_raid_init(
    const gpu_raid_config_t* config,
    gpu_raid_handle_t* handle
);
```

**Parameters:**
- `config` - Configuration structure
- `handle` - Output handle (must call `gpu_raid_destroy` when done)

**Returns:** Error code

**Example:**
```c
gpu_raid_config_t config = {
    .raid_level = GPU_RAID_LEVEL_5,
    .num_data_drives = 4,
    .stripe_size_kb = 256,
    .gpu_device_id = 0,
    .device_type = GPU_RAID_DEVICE_AUTO,
    .memory_pool_size_mb = 512,
    .num_streams = 1,
    .enable_profiling = true
};

gpu_raid_handle_t handle;
gpu_raid_error_t err = gpu_raid_init(&config, &handle);
```

### gpu_raid_config_t

Configuration structure.

```c
typedef struct {
    gpu_raid_level_t raid_level;        // RAID level (5 or 6)
    uint32_t num_data_drives;            // Number of data drives
    uint32_t stripe_size_kb;             // Stripe size in KB
    int gpu_device_id;                   // CUDA device ID (-1 for auto)
    gpu_raid_device_type_t device_type;  // GPU model (AUTO for detect)
    bool enable_tenstorrent;             // Enable TT integration
    size_t memory_pool_size_mb;          // GPU memory pool size
    uint32_t num_streams;                // Number of CUDA streams
    bool enable_profiling;               // Enable performance profiling
} gpu_raid_config_t;
```

**Fields:**
- `raid_level` - Use `GPU_RAID_LEVEL_5` or `GPU_RAID_LEVEL_6`
- `num_data_drives` - 2 to 16 drives
- `stripe_size_kb` - 64 to 16384 KB
- `gpu_device_id` - 0-based device index, or -1 for auto-select
- `memory_pool_size_mb` - 256 to 4096 MB recommended
- `num_streams` - 1 to 8 for concurrent operations

### gpu_raid_destroy

Destroy GPU RAID system and free resources.

```c
void gpu_raid_destroy(gpu_raid_handle_t handle);
```

**Parameters:**
- `handle` - Handle from `gpu_raid_init`

---

## Encoding

### gpu_raid_encode

Encode data blocks to generate parity.

```c
gpu_raid_error_t gpu_raid_encode(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    uint8_t** parity_blocks,
    uint32_t num_data_blocks,
    size_t block_size
);
```

**Parameters:**
- `handle` - GPU RAID handle
- `data_blocks` - Array of pointers to data blocks (host memory)
- `parity_blocks` - Array of pointers to parity blocks (host memory)
  - RAID 5: 1 parity block
  - RAID 6: 2 parity blocks (P and Q)
- `num_data_blocks` - Number of data blocks
- `block_size` - Size of each block in bytes

**Returns:** Error code

**Example (RAID 5):**
```c
uint8_t* data[4];
uint8_t* parity;

for (int i = 0; i < 4; i++) {
    data[i] = malloc(BLOCK_SIZE);
    // Fill with data
}
parity = malloc(BLOCK_SIZE);

gpu_raid_encode(handle, (const uint8_t**)data, &parity, 4, BLOCK_SIZE);
```

**Example (RAID 6):**
```c
uint8_t* parities[2];
parities[0] = malloc(BLOCK_SIZE);  // P parity
parities[1] = malloc(BLOCK_SIZE);  // Q parity

gpu_raid_encode(handle, (const uint8_t**)data, parities, 4, BLOCK_SIZE);
```

### gpu_raid_encode_async

Asynchronous encoding (non-blocking).

```c
gpu_raid_error_t gpu_raid_encode_async(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    uint8_t** parity_blocks,
    uint32_t num_data_blocks,
    size_t block_size,
    uint32_t stream_id
);
```

**Additional Parameter:**
- `stream_id` - CUDA stream ID (0 to `num_streams-1`)

**Note:** Call `gpu_raid_sync()` to wait for completion.

### gpu_raid_update_parity

Incrementally update parity after modifying a data block.

```c
gpu_raid_error_t gpu_raid_update_parity(
    gpu_raid_handle_t handle,
    const uint8_t** old_parity,
    uint8_t** new_parity,
    const uint8_t* old_data,
    const uint8_t* new_data,
    uint32_t block_index,
    size_t block_size
);
```

**Use Case:** More efficient than full re-encoding when only one block changes.

---

## Reconstruction

### gpu_raid_reconstruct

Reconstruct failed blocks from surviving data and parity.

```c
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
```

**Parameters:**
- `all_blocks` - Array of all blocks (`NULL` for failed blocks)
- `parity_blocks` - Array of parity blocks
- `failed_indices` - Indices of failed blocks
- `num_failed` - Number of failed blocks (1 for RAID 5, 1-2 for RAID 6)
- `recovered_blocks` - Output recovered blocks
- `num_total_blocks` - Total number of data blocks

**Example:**
```c
// Simulate failure of block 2
const uint8_t* all_blocks[4] = {
    data[0],
    data[1],
    NULL,      // Failed
    data[3]
};

uint32_t failed_idx = 2;
uint8_t* recovered = malloc(BLOCK_SIZE);

gpu_raid_reconstruct(
    handle, all_blocks, (const uint8_t**)&parity,
    &failed_idx, 1, &recovered, 4, BLOCK_SIZE
);

// recovered now contains reconstructed block 2
```

---

## Verification

### gpu_raid_verify

Verify parity consistency.

```c
gpu_raid_error_t gpu_raid_verify(
    gpu_raid_handle_t handle,
    const uint8_t** data_blocks,
    const uint8_t** parity_blocks,
    uint32_t num_blocks,
    size_t block_size,
    bool* is_valid
);
```

**Parameters:**
- `is_valid` - Output: `true` if parity matches, `false` otherwise

**Example:**
```c
bool valid;
gpu_raid_verify(handle, (const uint8_t**)data, (const uint8_t**)&parity,
                4, BLOCK_SIZE, &valid);

if (valid) {
    printf("Parity is consistent\n");
} else {
    printf("Parity mismatch detected!\n");
}
```

---

## Statistics

### gpu_raid_get_stats

Get performance statistics.

```c
gpu_raid_error_t gpu_raid_get_stats(
    gpu_raid_handle_t handle,
    gpu_raid_stats_t* stats
);
```

### gpu_raid_stats_t

Statistics structure.

```c
typedef struct {
    uint64_t total_encodes;
    uint64_t total_decodes;
    uint64_t total_bytes_encoded;
    uint64_t total_bytes_decoded;
    double avg_encode_time_ms;
    double avg_decode_time_ms;
    double peak_throughput_gbs;
    uint32_t gpu_utilization_percent;
    float gpu_temperature_c;
    uint32_t gpu_power_watts;
} gpu_raid_stats_t;
```

**Example:**
```c
gpu_raid_stats_t stats;
gpu_raid_get_stats(handle, &stats);

printf("Total encodes: %lu\n", stats.total_encodes);
printf("Throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
printf("GPU temp: %.1f°C\n", stats.gpu_temperature_c);
```

### gpu_raid_reset_stats

Reset statistics counters.

```c
void gpu_raid_reset_stats(gpu_raid_handle_t handle);
```

---

## Utilities

### gpu_raid_sync

Synchronize all pending GPU operations.

```c
gpu_raid_error_t gpu_raid_sync(gpu_raid_handle_t handle);
```

### gpu_raid_query_device

Query GPU capabilities.

```c
gpu_raid_error_t gpu_raid_query_device(
    int device_id,
    gpu_raid_device_type_t* device_type,
    uint32_t* cuda_cores,
    float* memory_gb
);
```

**Example:**
```c
gpu_raid_device_type_t type;
uint32_t cores;
float mem_gb;

gpu_raid_query_device(0, &type, &cores, &mem_gb);

if (type == GPU_RAID_DEVICE_RTX_3080) {
    printf("RTX 3080 detected: %u cores, %.1f GB\n", cores, mem_gb);
}
```

### gpu_raid_load_config

Load configuration from JSON file.

```c
gpu_raid_error_t gpu_raid_load_config(
    const char* json_path,
    gpu_raid_config_t* config
);
```

**Example:**
```c
gpu_raid_config_t config;
gpu_raid_load_config("config/rtx3080_config.json", &config);
```

### gpu_raid_get_version

Get library version.

```c
const char* gpu_raid_get_version(void);
```

**Example:**
```c
printf("GPU RAID version: %s\n", gpu_raid_get_version());
```

---

## Error Handling

### gpu_raid_error_t

Error codes.

```c
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
    GPU_RAID_ERROR_NOT_INITIALIZED = -9
} gpu_raid_error_t;
```

### gpu_raid_get_error_string

Get human-readable error message.

```c
const char* gpu_raid_get_error_string(gpu_raid_error_t error);
```

**Example:**
```c
gpu_raid_error_t err = gpu_raid_init(&config, &handle);
if (err != GPU_RAID_SUCCESS) {
    fprintf(stderr, "Error: %s\n", gpu_raid_get_error_string(err));
}
```

---

## Thread Safety

**Not Thread-Safe:** A single `gpu_raid_handle_t` should not be used concurrently from multiple threads.

**Multi-GPU:** To use multiple GPUs, create separate handles with different `gpu_device_id`.

---

## Memory Management

- All input/output buffers should be in **host memory** (not GPU memory)
- The library handles GPU memory allocation internally
- Buffers passed to API must remain valid until operations complete
- For async operations, buffers must remain valid until `gpu_raid_sync()`

---

## Performance Tips

1. **Use larger block sizes** (256 KB - 1 MB) for better throughput
2. **Enable multiple streams** (`num_streams > 1`) for async operations
3. **Pre-allocate buffers** to avoid malloc overhead
4. **Use pinned memory** if doing many transfers (via `cudaMallocHost`)
5. **Batch operations** when possible

---

## Complete Example

```c
#include <gpu_raid.h>

#define NUM_DRIVES 4
#define BLOCK_SIZE (256 * 1024)

int main() {
    // Initialize
    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = NUM_DRIVES,
        .stripe_size_kb = 256,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .memory_pool_size_mb = 1024,
        .num_streams = 2,
        .enable_profiling = true
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        return 1;
    }

    // Allocate buffers
    uint8_t* data[NUM_DRIVES];
    for (int i = 0; i < NUM_DRIVES; i++) {
        data[i] = malloc(BLOCK_SIZE);
        // Fill with data
    }
    uint8_t* parity = malloc(BLOCK_SIZE);

    // Encode
    gpu_raid_encode(handle, (const uint8_t**)data, &parity,
                    NUM_DRIVES, BLOCK_SIZE);

    // Verify
    bool valid;
    gpu_raid_verify(handle, (const uint8_t**)data, (const uint8_t**)&parity,
                    NUM_DRIVES, BLOCK_SIZE, &valid);

    // Get stats
    gpu_raid_stats_t stats;
    gpu_raid_get_stats(handle, &stats);
    printf("Throughput: %.2f GB/s\n", stats.peak_throughput_gbs);

    // Cleanup
    for (int i = 0; i < NUM_DRIVES; i++) free(data[i]);
    free(parity);
    gpu_raid_destroy(handle);

    return 0;
}
```
