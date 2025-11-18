/**
 * RAID 6 Dual Failure Recovery Example
 *
 * Demonstrates RAID 6 with Reed-Solomon P+Q parity
 * Recovers from simultaneous failure of 2 drives
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/gpu_raid.h"

#define NUM_DATA_DRIVES 6
#define BLOCK_SIZE (512 * 1024)  // 512 KB per block

void fill_random_data(uint8_t* block, size_t size) {
    for (size_t i = 0; i < size; i++) {
        block[i] = rand() & 0xFF;
    }
}

bool compare_blocks(const uint8_t* a, const uint8_t* b, size_t size) {
    return memcmp(a, b, size) == 0;
}

int main(int argc, char** argv) {
    printf("=== GPU RAID 6 Dual Failure Recovery Example ===\n\n");

    srand(time(NULL));

    // Initialize RAID 6
    printf("Initializing GPU RAID 6 with %d data drives + 2 parity drives...\n",
           NUM_DATA_DRIVES);

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_6,
        .num_data_drives = NUM_DATA_DRIVES,
        .stripe_size_kb = 512,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 1024,
        .num_streams = 2,
        .enable_profiling = true
    };

    gpu_raid_handle_t handle;
    gpu_raid_error_t err = gpu_raid_init(&config, &handle);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Initialization failed: %s\n",
                gpu_raid_get_error_string(err));
        return 1;
    }

    printf("Initialized successfully!\n\n");

    // Query GPU device
    gpu_raid_device_type_t device_type;
    uint32_t cuda_cores;
    float memory_gb;

    gpu_raid_query_device(0, &device_type, &cuda_cores, &memory_gb);

    const char* device_names[] = {
        "Auto-detect", "RTX 3080", "RTX 3060", "Quadro RTX 4000", "Generic"
    };
    const char* device_name = (device_type >= 0 && device_type <= 3) ?
                               device_names[device_type] : device_names[4];

    printf("GPU Details:\n");
    printf("  Device: %s\n", device_name);
    printf("  CUDA Cores: %u\n", cuda_cores);
    printf("  Memory: %.1f GB\n\n", memory_gb);

    // Allocate data blocks
    printf("Allocating and filling %d data blocks (%zu KB each)...\n",
           NUM_DATA_DRIVES, BLOCK_SIZE / 1024);

    uint8_t* data_blocks[NUM_DATA_DRIVES];
    const uint8_t* data_blocks_const[NUM_DATA_DRIVES];
    uint8_t* parity_blocks[2];  // P and Q

    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        data_blocks[i] = (uint8_t*)malloc(BLOCK_SIZE);
        fill_random_data(data_blocks[i], BLOCK_SIZE);
        data_blocks_const[i] = data_blocks[i];
    }

    parity_blocks[0] = (uint8_t*)malloc(BLOCK_SIZE);  // P parity
    parity_blocks[1] = (uint8_t*)malloc(BLOCK_SIZE);  // Q parity

    size_t total_data_mb = (NUM_DATA_DRIVES * BLOCK_SIZE) / (1024 * 1024);
    printf("  Total data: %zu MB\n\n", total_data_mb);

    // Encode (generate P and Q parities)
    printf("Generating P and Q parities...\n");

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    err = gpu_raid_encode(
        handle,
        data_blocks_const,
        parity_blocks,
        NUM_DATA_DRIVES,
        BLOCK_SIZE
    );

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Encoding failed: %s\n", gpu_raid_get_error_string(err));
        return 1;
    }

    double encode_time_s = (ts_end.tv_sec - ts_start.tv_sec) +
                           (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("  Encoding complete\n");
    printf("  Time: %.3f ms\n", encode_time_s * 1000.0);
    printf("  Throughput: %.2f GB/s\n\n",
           (NUM_DATA_DRIVES * BLOCK_SIZE) / encode_time_s / (1024.0 * 1024.0 * 1024.0));

    // Verify parities
    printf("Verifying parity consistency...\n");

    bool is_valid;
    err = gpu_raid_verify(handle, data_blocks_const, (const uint8_t**)parity_blocks,
                          NUM_DATA_DRIVES, BLOCK_SIZE, &is_valid);

    printf("  Parity check: %s\n\n", is_valid ? "VALID ✓" : "INVALID ✗");

    // Save copies of blocks 1 and 4 for verification
    printf("Simulating dual drive failure (blocks 1 and 4)...\n");

    uint8_t* original_block1 = (uint8_t*)malloc(BLOCK_SIZE);
    uint8_t* original_block4 = (uint8_t*)malloc(BLOCK_SIZE);
    memcpy(original_block1, data_blocks[1], BLOCK_SIZE);
    memcpy(original_block4, data_blocks[4], BLOCK_SIZE);

    // Simulate failures
    memset(data_blocks[1], 0xAA, BLOCK_SIZE);
    memset(data_blocks[4], 0xBB, BLOCK_SIZE);

    printf("  Blocks 1 and 4 corrupted\n");
    printf("  Attempting recovery using P+Q parities...\n\n");

    // Prepare for reconstruction
    const uint8_t* all_blocks[NUM_DATA_DRIVES];
    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        all_blocks[i] = (i == 1 || i == 4) ? NULL : data_blocks[i];
    }

    uint32_t failed_indices[2] = {1, 4};
    uint8_t* recovered_blocks[2];
    recovered_blocks[0] = (uint8_t*)malloc(BLOCK_SIZE);
    recovered_blocks[1] = (uint8_t*)malloc(BLOCK_SIZE);

    // Reconstruct both blocks
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    err = gpu_raid_reconstruct(
        handle,
        all_blocks,
        (const uint8_t**)parity_blocks,
        failed_indices,
        2,  // 2 failures
        recovered_blocks,
        NUM_DATA_DRIVES,
        BLOCK_SIZE
    );

    clock_gettime(CLOCK_MONOTONIC, &ts_end);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Reconstruction failed: %s\n",
                gpu_raid_get_error_string(err));
        return 1;
    }

    double rebuild_time_s = (ts_end.tv_sec - ts_start.tv_sec) +
                            (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    printf("Reconstruction Results:\n");
    printf("  Time: %.3f ms\n", rebuild_time_s * 1000.0);
    printf("  Rebuild speed: %.2f GB/s\n",
           (2 * BLOCK_SIZE) / rebuild_time_s / (1024.0 * 1024.0 * 1024.0));

    // Verify recovered data
    bool block1_ok = compare_blocks(original_block1, recovered_blocks[0], BLOCK_SIZE);
    bool block4_ok = compare_blocks(original_block4, recovered_blocks[1], BLOCK_SIZE);

    printf("  Block 1 recovery: %s\n", block1_ok ? "SUCCESS ✓" : "FAILED ✗");
    printf("  Block 4 recovery: %s\n\n", block4_ok ? "SUCCESS ✓" : "FAILED ✗");

    if (block1_ok && block4_ok) {
        printf("✓✓✓ DUAL FAILURE RECOVERY SUCCESSFUL ✓✓✓\n\n");
    } else {
        printf("✗✗✗ RECOVERY FAILED ✗✗✗\n\n");
    }

    // Performance statistics
    printf("Final Performance Statistics:\n");

    gpu_raid_stats_t stats;
    gpu_raid_get_stats(handle, &stats);

    printf("  Total operations: %lu encode, %lu decode\n",
           stats.total_encodes, stats.total_decodes);
    printf("  Total data processed: %.2f GB encoded, %.2f GB decoded\n",
           stats.total_bytes_encoded / (1024.0 * 1024.0 * 1024.0),
           stats.total_bytes_decoded / (1024.0 * 1024.0 * 1024.0));
    printf("  Average times: %.3f ms encode, %.3f ms decode\n",
           stats.avg_encode_time_ms, stats.avg_decode_time_ms);
    printf("  Peak throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
    printf("  GPU utilization: %u%%\n", stats.gpu_utilization_percent);
    printf("  GPU temperature: %.1f°C\n", stats.gpu_temperature_c);
    printf("  GPU power draw: %u W\n\n", stats.gpu_power_watts);

    // Cleanup
    printf("Cleaning up resources...\n");

    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        free(data_blocks[i]);
    }
    free(parity_blocks[0]);
    free(parity_blocks[1]);
    free(original_block1);
    free(original_block4);
    free(recovered_blocks[0]);
    free(recovered_blocks[1]);

    gpu_raid_destroy(handle);

    printf("\n=== Example completed ===\n");

    return 0;
}
