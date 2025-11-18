/**
 * Async Operations Example
 *
 * Demonstrates concurrent GPU operations using multiple CUDA streams
 * Shows how to achieve higher throughput through parallelism
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/gpu_raid.h"

#define NUM_DATA_DRIVES 4
#define BLOCK_SIZE (512 * 1024)  // 512 KB per block
#define NUM_STRIPES 16           // Process 16 stripes concurrently
#define NUM_STREAMS 4            // Use 4 CUDA streams

void fill_pattern(uint8_t* block, size_t size, uint32_t pattern) {
    uint32_t* ptr = (uint32_t*)block;
    for (size_t i = 0; i < size / sizeof(uint32_t); i++) {
        ptr[i] = pattern + i;
    }
}

int main(int argc, char** argv) {
    printf("=== GPU RAID Async Operations Example ===\n\n");

    // Configure with multiple streams
    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = NUM_DATA_DRIVES,
        .stripe_size_kb = 512,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 2048,
        .num_streams = NUM_STREAMS,  // Enable multiple streams
        .enable_profiling = true
    };

    gpu_raid_handle_t handle;
    gpu_raid_error_t err = gpu_raid_init(&config, &handle);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Initialization failed: %s\n",
                gpu_raid_get_error_string(err));
        return 1;
    }

    printf("Initialized with %d CUDA streams for concurrent operations\n\n", NUM_STREAMS);

    // Allocate buffers for all stripes
    printf("Allocating buffers for %d stripes...\n", NUM_STRIPES);

    uint8_t*** data_stripes = (uint8_t***)malloc(NUM_STRIPES * sizeof(uint8_t**));
    const uint8_t*** data_stripes_const = (const uint8_t***)malloc(NUM_STRIPES * sizeof(const uint8_t**));
    uint8_t** parity_stripes = (uint8_t**)malloc(NUM_STRIPES * sizeof(uint8_t*));

    for (int s = 0; s < NUM_STRIPES; s++) {
        data_stripes[s] = (uint8_t**)malloc(NUM_DATA_DRIVES * sizeof(uint8_t*));
        data_stripes_const[s] = (const uint8_t**)malloc(NUM_DATA_DRIVES * sizeof(const uint8_t*));

        for (int d = 0; d < NUM_DATA_DRIVES; d++) {
            data_stripes[s][d] = (uint8_t*)malloc(BLOCK_SIZE);
            fill_pattern(data_stripes[s][d], BLOCK_SIZE, (s * NUM_DATA_DRIVES + d) * 0x1000);
            data_stripes_const[s][d] = data_stripes[s][d];
        }

        parity_stripes[s] = (uint8_t*)malloc(BLOCK_SIZE);
    }

    printf("  Allocated %zu MB total\n\n",
           (NUM_STRIPES * (NUM_DATA_DRIVES + 1) * BLOCK_SIZE) / (1024 * 1024));

    // === Synchronous (Sequential) Encoding ===
    printf("=== Method 1: Synchronous Encoding ===\n");

    struct timespec sync_start, sync_end;
    clock_gettime(CLOCK_MONOTONIC, &sync_start);

    for (int s = 0; s < NUM_STRIPES; s++) {
        err = gpu_raid_encode(
            handle,
            data_stripes_const[s],
            &parity_stripes[s],
            NUM_DATA_DRIVES,
            BLOCK_SIZE
        );

        if (err != GPU_RAID_SUCCESS) {
            fprintf(stderr, "Encode %d failed: %s\n", s,
                    gpu_raid_get_error_string(err));
            return 1;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &sync_end);

    double sync_time = (sync_end.tv_sec - sync_start.tv_sec) +
                       (sync_end.tv_nsec - sync_start.tv_nsec) / 1e9;
    double sync_throughput = (NUM_STRIPES * NUM_DATA_DRIVES * BLOCK_SIZE) /
                             sync_time / (1024.0 * 1024.0 * 1024.0);

    printf("  Time: %.3f seconds\n", sync_time);
    printf("  Throughput: %.2f GB/s\n\n", sync_throughput);

    // === Asynchronous (Concurrent) Encoding ===
    printf("=== Method 2: Asynchronous Encoding (Multi-Stream) ===\n");

    struct timespec async_start, async_end;
    clock_gettime(CLOCK_MONOTONIC, &async_start);

    // Launch all encodes asynchronously
    for (int s = 0; s < NUM_STRIPES; s++) {
        uint32_t stream_id = s % NUM_STREAMS;  // Round-robin stream assignment

        err = gpu_raid_encode_async(
            handle,
            data_stripes_const[s],
            &parity_stripes[s],
            NUM_DATA_DRIVES,
            BLOCK_SIZE,
            stream_id
        );

        if (err != GPU_RAID_SUCCESS) {
            fprintf(stderr, "Async encode %d failed: %s\n", s,
                    gpu_raid_get_error_string(err));
            return 1;
        }
    }

    // Wait for all streams to complete
    err = gpu_raid_sync(handle);
    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Sync failed: %s\n", gpu_raid_get_error_string(err));
        return 1;
    }

    clock_gettime(CLOCK_MONOTONIC, &async_end);

    double async_time = (async_end.tv_sec - async_start.tv_sec) +
                        (async_end.tv_nsec - async_start.tv_nsec) / 1e9;
    double async_throughput = (NUM_STRIPES * NUM_DATA_DRIVES * BLOCK_SIZE) /
                              async_time / (1024.0 * 1024.0 * 1024.0);

    printf("  Time: %.3f seconds\n", async_time);
    printf("  Throughput: %.2f GB/s\n\n", async_throughput);

    // === Results Comparison ===
    printf("=== Performance Comparison ===\n");
    printf("  Synchronous:   %.2f GB/s\n", sync_throughput);
    printf("  Asynchronous:  %.2f GB/s\n", async_throughput);
    printf("  Speedup:       %.2fx\n", async_throughput / sync_throughput);
    printf("  Latency saved: %.3f seconds\n\n", sync_time - async_time);

    if (async_throughput > sync_throughput * 1.5) {
        printf("✓ Excellent! Async operations show significant speedup.\n");
        printf("  Multi-stream parallelism is working effectively.\n");
    } else if (async_throughput > sync_throughput) {
        printf("✓ Good! Some speedup from async operations.\n");
        printf("  Consider increasing num_streams or stripe count for more gain.\n");
    } else {
        printf("⚠ Warning: Async not faster than sync.\n");
        printf("  Possible issues:\n");
        printf("  - Stripes too small (increase BLOCK_SIZE)\n");
        printf("  - Not enough concurrent work\n");
        printf("  - GPU bottleneck\n");
    }

    printf("\n");

    // === Verify parity correctness ===
    printf("=== Verifying Parity Correctness ===\n");

    bool all_valid = true;
    for (int s = 0; s < NUM_STRIPES; s++) {
        bool is_valid;
        err = gpu_raid_verify(
            handle,
            data_stripes_const[s],
            (const uint8_t**)&parity_stripes[s],
            NUM_DATA_DRIVES,
            BLOCK_SIZE,
            &is_valid
        );

        if (err != GPU_RAID_SUCCESS || !is_valid) {
            printf("  Stripe %d: INVALID ✗\n", s);
            all_valid = false;
        }
    }

    if (all_valid) {
        printf("  All %d stripes: VALID ✓\n", NUM_STRIPES);
    }

    printf("\n");

    // === Statistics ===
    gpu_raid_stats_t stats;
    gpu_raid_get_stats(handle, &stats);

    printf("=== GPU RAID Statistics ===\n");
    printf("  Total encodes: %lu\n", stats.total_encodes);
    printf("  Total bytes processed: %.2f GB\n",
           stats.total_bytes_encoded / (1024.0 * 1024.0 * 1024.0));
    printf("  Average encode time: %.3f ms\n", stats.avg_encode_time_ms);
    printf("  Peak throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
    printf("  GPU utilization: %u%%\n", stats.gpu_utilization_percent);
    printf("  GPU temperature: %.1f°C\n", stats.gpu_temperature_c);
    printf("\n");

    // Cleanup
    printf("Cleaning up...\n");

    for (int s = 0; s < NUM_STRIPES; s++) {
        for (int d = 0; d < NUM_DATA_DRIVES; d++) {
            free(data_stripes[s][d]);
        }
        free(data_stripes[s]);
        free(data_stripes_const[s]);
        free(parity_stripes[s]);
    }

    free(data_stripes);
    free(data_stripes_const);
    free(parity_stripes);

    gpu_raid_destroy(handle);

    printf("\n=== Example completed successfully ===\n");

    return 0;
}
