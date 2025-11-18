/**
 * GPU RAID Stress Test
 *
 * Continuous testing for stability and correctness
 * Runs encode/decode cycles to detect errors
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include "../../include/gpu_raid.h"

static volatile bool g_running = true;

void signal_handler(int sig) {
    printf("\nReceived signal %d, stopping...\n", sig);
    g_running = false;
}

bool verify_data_integrity(const uint8_t* original, const uint8_t* recovered, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (original[i] != recovered[i]) {
            printf("ERROR: Data mismatch at byte %zu: %02X != %02X\n",
                   i, original[i], recovered[i]);
            return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║  GPU RAID Stress Test                  ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    int duration_seconds = 60;
    if (argc > 1) {
        duration_seconds = atoi(argv[1]);
    }

    printf("Duration: %d seconds\n", duration_seconds);
    printf("Press Ctrl+C to stop early\n\n");

    // Test configuration
    const int num_drives = 6;
    const size_t block_size = 512 * 1024;  // 512 KB

    // Initialize RAID 6
    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_6,
        .num_data_drives = num_drives,
        .stripe_size_kb = 512,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .memory_pool_size_mb = 2048,
        .num_streams = 4,
        .enable_profiling = true
    };

    gpu_raid_handle_t handle;
    gpu_raid_error_t err = gpu_raid_init(&config, &handle);

    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Initialization failed: %s\n",
                gpu_raid_get_error_string(err));
        return 1;
    }

    printf("Testing RAID 6 with %d drives\n\n", num_drives);

    // Allocate buffers
    uint8_t** data = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
    const uint8_t** data_const = (const uint8_t**)malloc(num_drives * sizeof(const uint8_t*));

    for (int i = 0; i < num_drives; i++) {
        data[i] = (uint8_t*)malloc(block_size);
        data_const[i] = data[i];
    }

    uint8_t* parities[2];
    parities[0] = (uint8_t*)malloc(block_size);
    parities[1] = (uint8_t*)malloc(block_size);

    uint8_t* recovered[2];
    recovered[0] = (uint8_t*)malloc(block_size);
    recovered[1] = (uint8_t*)malloc(block_size);

    // Statistics
    uint64_t total_operations = 0;
    uint64_t total_errors = 0;
    uint64_t total_bytes_processed = 0;

    time_t start_time = time(NULL);
    time_t last_report = start_time;

    printf("Starting stress test...\n\n");
    printf("%-10s %-15s %-15s %-15s %-15s\n",
           "Time", "Operations", "Errors", "Throughput", "GPU Temp");
    printf("------------------------------------------------------------------------\n");

    while (g_running && (time(NULL) - start_time) < duration_seconds) {
        // Fill data with random pattern
        uint32_t pattern = rand();
        for (int i = 0; i < num_drives; i++) {
            uint32_t* ptr = (uint32_t*)data[i];
            for (size_t j = 0; j < block_size / sizeof(uint32_t); j++) {
                ptr[j] = pattern + i * 1000 + j;
            }
        }

        // Encode
        err = gpu_raid_encode(handle, data_const, parities, num_drives, block_size);
        if (err != GPU_RAID_SUCCESS) {
            printf("ERROR: Encode failed: %s\n", gpu_raid_get_error_string(err));
            total_errors++;
        }

        // Verify parity
        bool valid;
        gpu_raid_verify(handle, data_const, (const uint8_t**)parities,
                        num_drives, block_size, &valid);
        if (!valid) {
            printf("ERROR: Parity verification failed\n");
            total_errors++;
        }

        // Test single failure recovery
        uint8_t* backup = (uint8_t*)malloc(block_size);
        memcpy(backup, data[3], block_size);

        const uint8_t* all_blocks[num_drives];
        for (int i = 0; i < num_drives; i++) {
            all_blocks[i] = (i == 3) ? NULL : data[i];
        }

        uint32_t failed_idx = 3;
        err = gpu_raid_reconstruct(
            handle, all_blocks, (const uint8_t**)parities,
            &failed_idx, 1, &recovered[0], num_drives, block_size
        );

        if (err != GPU_RAID_SUCCESS) {
            printf("ERROR: Reconstruction failed: %s\n",
                   gpu_raid_get_error_string(err));
            total_errors++;
        }

        if (!verify_data_integrity(backup, recovered[0], block_size)) {
            total_errors++;
        }

        free(backup);

        total_operations++;
        total_bytes_processed += num_drives * block_size;

        // Report every second
        time_t now = time(NULL);
        if (now > last_report) {
            gpu_raid_stats_t stats;
            gpu_raid_get_stats(handle, &stats);

            double elapsed = now - start_time;
            double throughput = total_bytes_processed / elapsed / (1024*1024*1024);

            printf("%-10ld %-15lu %-15lu %-15.2f %-15.1f\n",
                   (long)elapsed, total_operations, total_errors,
                   throughput, stats.gpu_temperature_c);

            last_report = now;
        }
    }

    printf("\n");
    printf("=== Stress Test Results ===\n");
    printf("Total Operations: %lu\n", total_operations);
    printf("Total Errors: %lu\n", total_errors);
    printf("Error Rate: %.6f%%\n", 100.0 * total_errors / total_operations);
    printf("Total Data Processed: %.2f GB\n",
           total_bytes_processed / (1024.0*1024.0*1024.0));
    printf("Duration: %ld seconds\n", (long)(time(NULL) - start_time));

    if (total_errors == 0) {
        printf("\n✓ SUCCESS: No errors detected!\n");
    } else {
        printf("\n✗ FAILED: %lu errors detected\n", total_errors);
    }

    // Final statistics
    gpu_raid_stats_t stats;
    gpu_raid_get_stats(handle, &stats);

    printf("\nGPU Statistics:\n");
    printf("  Peak Throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
    printf("  Avg Encode Time: %.3f ms\n", stats.avg_encode_time_ms);
    printf("  Avg Decode Time: %.3f ms\n", stats.avg_decode_time_ms);
    printf("  GPU Utilization: %u%%\n", stats.gpu_utilization_percent);
    printf("  GPU Temperature: %.1f°C\n", stats.gpu_temperature_c);
    printf("  GPU Power: %u W\n", stats.gpu_power_watts);

    // Cleanup
    for (int i = 0; i < num_drives; i++) {
        free(data[i]);
    }
    free(data);
    free(data_const);
    free(parities[0]);
    free(parities[1]);
    free(recovered[0]);
    free(recovered[1]);

    gpu_raid_destroy(handle);

    printf("\n========================================\n\n");

    return (total_errors == 0) ? 0 : 1;
}
