/**
 * Simple RAID 5 Example
 *
 * Demonstrates basic GPU RAID 5 encoding and recovery
 * Uses 4 data drives + 1 parity drive
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/gpu_raid.h"

#define NUM_DATA_DRIVES 4
#define BLOCK_SIZE (256 * 1024)  // 256 KB per block

// Fill block with pattern for testing
void fill_test_pattern(uint8_t* block, size_t size, uint8_t pattern) {
    for (size_t i = 0; i < size; i++) {
        block[i] = (pattern + i) & 0xFF;
    }
}

// Verify blocks match
bool verify_blocks(const uint8_t* expected, const uint8_t* actual, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (expected[i] != actual[i]) {
            printf("Mismatch at byte %zu: expected 0x%02X, got 0x%02X\n",
                   i, expected[i], actual[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    printf("=== GPU RAID 5 Simple Example ===\n\n");

    // Step 1: Initialize GPU RAID
    printf("Step 1: Initializing GPU RAID 5...\n");

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = NUM_DATA_DRIVES,
        .stripe_size_kb = 256,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 512,
        .num_streams = 1,
        .enable_profiling = true
    };

    gpu_raid_handle_t handle;
    gpu_raid_error_t err = gpu_raid_init(&config, &handle);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Failed to initialize GPU RAID: %s\n",
                gpu_raid_get_error_string(err));
        return 1;
    }

    printf("  Version: %s\n", gpu_raid_get_version());
    printf("  Initialized successfully!\n\n");

    // Step 2: Allocate and prepare data blocks
    printf("Step 2: Preparing data blocks...\n");

    uint8_t* data_blocks[NUM_DATA_DRIVES];
    const uint8_t* data_blocks_const[NUM_DATA_DRIVES];
    uint8_t* parity_block;

    // Allocate memory
    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        data_blocks[i] = (uint8_t*)malloc(BLOCK_SIZE);
        if (!data_blocks[i]) {
            fprintf(stderr, "Failed to allocate data block %d\n", i);
            return 1;
        }
        fill_test_pattern(data_blocks[i], BLOCK_SIZE, i * 17);
        data_blocks_const[i] = data_blocks[i];
        printf("  Data block %d: filled with pattern 0x%02X\n", i, i * 17);
    }

    parity_block = (uint8_t*)malloc(BLOCK_SIZE);
    if (!parity_block) {
        fprintf(stderr, "Failed to allocate parity block\n");
        return 1;
    }

    printf("  Total data: %zu KB\n\n", (NUM_DATA_DRIVES * BLOCK_SIZE) / 1024);

    // Step 3: Encode (generate parity)
    printf("Step 3: Generating parity...\n");

    clock_t start = clock();

    err = gpu_raid_encode(
        handle,
        data_blocks_const,
        &parity_block,
        NUM_DATA_DRIVES,
        BLOCK_SIZE
    );

    clock_t end = clock();
    double encode_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Encoding failed: %s\n", gpu_raid_get_error_string(err));
        return 1;
    }

    printf("  Parity generated successfully\n");
    printf("  Encoding time: %.3f ms\n", encode_time_ms);
    printf("  Throughput: %.2f GB/s\n\n",
           (NUM_DATA_DRIVES * BLOCK_SIZE) / (encode_time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));

    // Step 4: Verify parity
    printf("Step 4: Verifying parity...\n");

    bool is_valid;
    err = gpu_raid_verify(
        handle,
        data_blocks_const,
        (const uint8_t**)&parity_block,
        NUM_DATA_DRIVES,
        BLOCK_SIZE,
        &is_valid
    );

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Verification failed: %s\n", gpu_raid_get_error_string(err));
        return 1;
    }

    printf("  Parity validation: %s\n\n", is_valid ? "PASSED ✓" : "FAILED ✗");

    // Step 5: Simulate drive failure and recovery
    printf("Step 5: Simulating drive failure and recovery...\n");

    // Save original block 2 for verification
    uint8_t* original_block2 = (uint8_t*)malloc(BLOCK_SIZE);
    memcpy(original_block2, data_blocks[2], BLOCK_SIZE);

    // Simulate block 2 failure
    printf("  Simulating failure of data block 2...\n");
    memset(data_blocks[2], 0, BLOCK_SIZE);

    // Prepare for reconstruction
    const uint8_t* all_blocks[NUM_DATA_DRIVES];
    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        all_blocks[i] = (i == 2) ? NULL : data_blocks[i];
    }

    uint32_t failed_index = 2;
    uint8_t* recovered_block = (uint8_t*)malloc(BLOCK_SIZE);

    // Reconstruct
    start = clock();

    err = gpu_raid_reconstruct(
        handle,
        all_blocks,
        (const uint8_t**)&parity_block,
        &failed_index,
        1,
        &recovered_block,
        NUM_DATA_DRIVES,
        BLOCK_SIZE
    );

    end = clock();
    double rebuild_time_ms = ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0;

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Reconstruction failed: %s\n", gpu_raid_get_error_string(err));
        return 1;
    }

    printf("  Block reconstructed successfully\n");
    printf("  Rebuild time: %.3f ms\n", rebuild_time_ms);
    printf("  Rebuild speed: %.2f GB/s\n",
           BLOCK_SIZE / (rebuild_time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0));

    // Verify recovery
    bool recovery_correct = verify_blocks(original_block2, recovered_block, BLOCK_SIZE);
    printf("  Recovery verification: %s\n\n", recovery_correct ? "PASSED ✓" : "FAILED ✗");

    // Step 6: Get statistics
    printf("Step 6: Performance statistics...\n");

    gpu_raid_stats_t stats;
    gpu_raid_get_stats(handle, &stats);

    printf("  Total encodes: %lu\n", stats.total_encodes);
    printf("  Total decodes: %lu\n", stats.total_decodes);
    printf("  Average encode time: %.3f ms\n", stats.avg_encode_time_ms);
    printf("  Average decode time: %.3f ms\n", stats.avg_decode_time_ms);
    printf("  Peak throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
    printf("  GPU utilization: %u%%\n", stats.gpu_utilization_percent);
    printf("  GPU temperature: %.1f°C\n", stats.gpu_temperature_c);
    printf("  GPU power: %u W\n\n", stats.gpu_power_watts);

    // Cleanup
    printf("Cleaning up...\n");

    for (int i = 0; i < NUM_DATA_DRIVES; i++) {
        free(data_blocks[i]);
    }
    free(parity_block);
    free(original_block2);
    free(recovered_block);

    gpu_raid_destroy(handle);

    printf("\n=== Example completed successfully ===\n");

    return 0;
}
