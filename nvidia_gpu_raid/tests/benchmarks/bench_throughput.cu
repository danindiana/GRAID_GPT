/**
 * GPU RAID Throughput Benchmark
 *
 * Measures encoding and decoding throughput for various configurations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../../include/gpu_raid.h"

#define MIN_BLOCK_SIZE (64 * 1024)      // 64 KB
#define MAX_BLOCK_SIZE (4 * 1024 * 1024) // 4 MB
#define NUM_ITERATIONS 100

typedef struct {
    size_t block_size;
    int num_drives;
    double encode_gbs;
    double decode_gbs;
} benchmark_result_t;

double measure_encode_throughput(
    gpu_raid_handle_t handle,
    int num_drives,
    size_t block_size,
    int iterations
) {
    uint8_t** data_blocks = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
    const uint8_t** data_const = (const uint8_t**)malloc(num_drives * sizeof(uint8_t*));
    uint8_t* parity_block = (uint8_t*)malloc(block_size);

    for (int i = 0; i < num_drives; i++) {
        data_blocks[i] = (uint8_t*)malloc(block_size);
        memset(data_blocks[i], i & 0xFF, block_size);
        data_const[i] = data_blocks[i];
    }

    // Warmup
    gpu_raid_encode(handle, data_const, &parity_block, num_drives, block_size);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < iterations; iter++) {
        gpu_raid_encode(handle, data_const, &parity_block, num_drives, block_size);
    }

    gpu_raid_sync(handle);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    double total_bytes = (double)num_drives * block_size * iterations;
    double throughput_gbs = (total_bytes / elapsed) / (1024.0 * 1024.0 * 1024.0);

    // Cleanup
    for (int i = 0; i < num_drives; i++) {
        free(data_blocks[i]);
    }
    free(data_blocks);
    free(data_const);
    free(parity_block);

    return throughput_gbs;
}

void run_benchmark_suite(gpu_raid_level_t raid_level, const char* gpu_name) {
    printf("\n========================================\n");
    printf("GPU RAID %d Throughput Benchmark\n", raid_level);
    printf("GPU: %s\n", gpu_name);
    printf("========================================\n\n");

    int drive_counts[] = {3, 4, 6, 8};
    size_t block_sizes[] = {64*1024, 128*1024, 256*1024, 512*1024, 1024*1024, 2*1024*1024};

    printf("%-12s %-12s %-20s %-20s\n",
           "Block Size", "Num Drives", "Encode (GB/s)", "Iterations");
    printf("------------------------------------------------------------------------\n");

    for (int d = 0; d < sizeof(drive_counts) / sizeof(drive_counts[0]); d++) {
        int num_drives = drive_counts[d];

        gpu_raid_config_t config = {
            .raid_level = raid_level,
            .num_data_drives = num_drives,
            .stripe_size_kb = 256,
            .gpu_device_id = 0,
            .device_type = GPU_RAID_DEVICE_AUTO,
            .enable_tenstorrent = false,
            .memory_pool_size_mb = 2048,
            .num_streams = 4,
            .enable_profiling = false
        };

        gpu_raid_handle_t handle;
        if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
            printf("Failed to initialize for %d drives\n", num_drives);
            continue;
        }

        for (int b = 0; b < sizeof(block_sizes) / sizeof(block_sizes[0]); b++) {
            size_t block_size = block_sizes[b];

            // Adjust iterations for larger blocks
            int iterations = (block_size > 1024*1024) ? 50 : NUM_ITERATIONS;

            double throughput = measure_encode_throughput(
                handle, num_drives, block_size, iterations
            );

            printf("%-12zu %-12d %-20.2f %-20d\n",
                   block_size / 1024, num_drives, throughput, iterations);
        }

        gpu_raid_destroy(handle);
        printf("\n");
    }
}

int main(int argc, char** argv) {
    printf("\n");
    printf("╔════════════════════════════════════════╗\n");
    printf("║  GPU RAID Throughput Benchmark Suite  ║\n");
    printf("╚════════════════════════════════════════╝\n");

    // Query GPU
    gpu_raid_device_type_t device_type;
    uint32_t cuda_cores;
    float memory_gb;

    if (gpu_raid_query_device(0, &device_type, &cuda_cores, &memory_gb) != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Failed to query GPU device\n");
        return 1;
    }

    char gpu_name[256];
    switch (device_type) {
        case GPU_RAID_DEVICE_RTX_3080:
            snprintf(gpu_name, sizeof(gpu_name), "RTX 3080 (%u cores, %.1f GB)",
                     cuda_cores, memory_gb);
            break;
        case GPU_RAID_DEVICE_RTX_3060:
            snprintf(gpu_name, sizeof(gpu_name), "RTX 3060 (%u cores, %.1f GB)",
                     cuda_cores, memory_gb);
            break;
        case GPU_RAID_DEVICE_QUADRO_4000:
            snprintf(gpu_name, sizeof(gpu_name), "Quadro RTX 4000 (%u cores, %.1f GB)",
                     cuda_cores, memory_gb);
            break;
        default:
            snprintf(gpu_name, sizeof(gpu_name), "Generic GPU (%u cores, %.1f GB)",
                     cuda_cores, memory_gb);
    }

    // Run benchmarks
    run_benchmark_suite(GPU_RAID_LEVEL_5, gpu_name);
    run_benchmark_suite(GPU_RAID_LEVEL_6, gpu_name);

    printf("\n========================================\n");
    printf("Benchmark suite completed\n");
    printf("========================================\n\n");

    return 0;
}
