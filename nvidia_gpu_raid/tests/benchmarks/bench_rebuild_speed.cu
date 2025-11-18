/**
 * GPU RAID Rebuild Speed Benchmark
 *
 * Measures drive reconstruction speed for RAID 5/6
 * Simulates real-world rebuild scenarios
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../include/gpu_raid.h"

#define DRIVE_SIZE_GB 8  // Simulate 8TB drive
#define CHUNK_SIZE_MB 256  // Process in 256MB chunks

double benchmark_raid5_rebuild(int num_drives, size_t chunk_size) {
    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = (uint32_t)num_drives,
        .stripe_size_kb = 256,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .memory_pool_size_mb = 2048,
        .num_streams = 2,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        return 0.0;
    }

    // Allocate buffers
    uint8_t** data = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
    const uint8_t** data_const = (const uint8_t**)malloc(num_drives * sizeof(const uint8_t*));

    for (int i = 0; i < num_drives; i++) {
        data[i] = (uint8_t*)malloc(chunk_size);
        memset(data[i], i * 17, chunk_size);
        data_const[i] = data[i];
    }

    uint8_t* parity = (uint8_t*)malloc(chunk_size);

    // Generate parity
    gpu_raid_encode(handle, data_const, &parity, num_drives, chunk_size);

    // Simulate failure (remove drive 2)
    const uint8_t* all_blocks[num_drives];
    for (int i = 0; i < num_drives; i++) {
        all_blocks[i] = (i == 2) ? NULL : data[i];
    }

    uint32_t failed_idx = 2;
    uint8_t* recovered = (uint8_t*)malloc(chunk_size);

    // Benchmark rebuild
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    int iterations = 100;
    for (int iter = 0; iter < iterations; iter++) {
        gpu_raid_reconstruct(
            handle, all_blocks, (const uint8_t**)&parity,
            &failed_idx, 1, &recovered, num_drives, chunk_size
        );
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    double rebuild_gbs = (iterations * chunk_size) / elapsed / (1024*1024*1024);

    // Cleanup
    for (int i = 0; i < num_drives; i++) {
        free(data[i]);
    }
    free(data);
    free(data_const);
    free(parity);
    free(recovered);

    gpu_raid_destroy(handle);

    return rebuild_gbs;
}

double benchmark_raid6_rebuild(int num_drives, size_t chunk_size, int num_failures) {
    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_6,
        .num_data_drives = (uint32_t)num_drives,
        .stripe_size_kb = 256,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .memory_pool_size_mb = 2048,
        .num_streams = 2,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        return 0.0;
    }

    // Allocate buffers
    uint8_t** data = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
    const uint8_t** data_const = (const uint8_t**)malloc(num_drives * sizeof(const uint8_t*));

    for (int i = 0; i < num_drives; i++) {
        data[i] = (uint8_t*)malloc(chunk_size);
        memset(data[i], i * 13, chunk_size);
        data_const[i] = data[i];
    }

    uint8_t* parities[2];
    parities[0] = (uint8_t*)malloc(chunk_size);
    parities[1] = (uint8_t*)malloc(chunk_size);

    // Generate parities
    gpu_raid_encode(handle, data_const, parities, num_drives, chunk_size);

    if (num_failures == 1) {
        // Single failure rebuild
        const uint8_t* all_blocks[num_drives];
        for (int i = 0; i < num_drives; i++) {
            all_blocks[i] = (i == 2) ? NULL : data[i];
        }

        uint32_t failed_idx = 2;
        uint8_t* recovered = (uint8_t*)malloc(chunk_size);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        int iterations = 100;
        for (int iter = 0; iter < iterations; iter++) {
            gpu_raid_reconstruct(
                handle, all_blocks, (const uint8_t**)parities,
                &failed_idx, 1, &recovered, num_drives, chunk_size
            );
        }

        clock_gettime(CLOCK_MONOTONIC, &end);

        double elapsed = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / 1e9;

        double rebuild_gbs = (iterations * chunk_size) / elapsed / (1024*1024*1024);

        free(recovered);
        gpu_raid_destroy(handle);

        for (int i = 0; i < num_drives; i++) free(data[i]);
        free(data);
        free(data_const);
        free(parities[0]);
        free(parities[1]);

        return rebuild_gbs;

    } else {
        // Dual failure rebuild
        // Placeholder - would use RAID 6 dual reconstruction kernel
        gpu_raid_destroy(handle);

        for (int i = 0; i < num_drives; i++) free(data[i]);
        free(data);
        free(data_const);
        free(parities[0]);
        free(parities[1]);

        return 0.0;  // Not yet implemented
    }
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║  GPU RAID Rebuild Speed Benchmark     ║\n");
    printf("╚════════════════════════════════════════╝\n\n");

    // Query GPU
    gpu_raid_device_type_t device_type;
    uint32_t cuda_cores;
    float memory_gb;

    gpu_raid_query_device(0, &device_type, &cuda_cores, &memory_gb);

    printf("GPU: ");
    switch (device_type) {
        case GPU_RAID_DEVICE_RTX_3080:
            printf("RTX 3080");
            break;
        case GPU_RAID_DEVICE_RTX_3060:
            printf("RTX 3060");
            break;
        case GPU_RAID_DEVICE_QUADRO_4000:
            printf("Quadro RTX 4000");
            break;
        default:
            printf("Generic");
    }
    printf(" (%u cores, %.1f GB)\n\n", cuda_cores, memory_gb);

    // Test configurations
    size_t chunk_sizes[] = {64*1024*1024, 128*1024*1024, 256*1024*1024};
    int drive_counts[] = {4, 6, 8};

    // === RAID 5 Rebuild ===
    printf("=== RAID 5 Single Drive Rebuild ===\n\n");
    printf("%-12s %-10s %-20s %-20s\n", "Chunk Size", "Drives", "Rebuild Speed", "8TB Time");
    printf("------------------------------------------------------------------------\n");

    for (int d = 0; d < 3; d++) {
        for (int c = 0; c < 3; c++) {
            size_t chunk_size = chunk_sizes[c];
            int drives = drive_counts[d];

            double rebuild_gbs = benchmark_raid5_rebuild(drives, chunk_size);

            // Calculate time to rebuild 8TB drive
            double tb_8_time_hours = (8.0 * 1024.0) / rebuild_gbs / 3600.0;

            printf("%-12zuMB %-10d %-20.2f %-20.1f\n",
                   chunk_size / (1024*1024), drives,
                   rebuild_gbs, tb_8_time_hours * 60);  // Convert to minutes
        }
    }

    printf("\n");

    // === RAID 6 Rebuild ===
    printf("=== RAID 6 Single Drive Rebuild ===\n\n");
    printf("%-12s %-10s %-20s %-20s\n", "Chunk Size", "Drives", "Rebuild Speed", "8TB Time");
    printf("------------------------------------------------------------------------\n");

    for (int d = 0; d < 3; d++) {
        for (int c = 0; c < 3; c++) {
            size_t chunk_size = chunk_sizes[c];
            int drives = drive_counts[d];

            double rebuild_gbs = benchmark_raid6_rebuild(drives, chunk_size, 1);

            double tb_8_time_hours = (8.0 * 1024.0) / rebuild_gbs / 3600.0;

            printf("%-12zuMB %-10d %-20.2f %-20.1f\n",
                   chunk_size / (1024*1024), drives,
                   rebuild_gbs, tb_8_time_hours * 60);
        }
    }

    printf("\n");

    // === Summary ===
    printf("=== Estimated Rebuild Times (8TB Drive) ===\n\n");

    size_t optimal_chunk = 256 * 1024 * 1024;
    double raid5_speed = benchmark_raid5_rebuild(6, optimal_chunk);
    double raid6_speed = benchmark_raid6_rebuild(6, optimal_chunk, 1);

    double raid5_minutes = (8.0 * 1024.0) / raid5_speed / 60.0;
    double raid6_minutes = (8.0 * 1024.0) / raid6_speed / 60.0;

    printf("Configuration: 6 drives, 256MB chunks\n\n");
    printf("  RAID 5 (single parity):\n");
    printf("    Speed: %.2f GB/s\n", raid5_speed);
    printf("    Time: %.1f minutes (%.2f hours)\n", raid5_minutes, raid5_minutes / 60.0);
    printf("\n");
    printf("  RAID 6 (dual parity):\n");
    printf("    Speed: %.2f GB/s\n", raid6_speed);
    printf("    Time: %.1f minutes (%.2f hours)\n", raid6_minutes, raid6_minutes / 60.0);
    printf("\n");

    printf("Note: These are GPU processing speeds only.\n");
    printf("Actual rebuild time includes disk I/O overhead.\n");
    printf("\n");

    printf("========================================\n");
    printf("Benchmark complete!\n");
    printf("========================================\n\n");

    return 0;
}
