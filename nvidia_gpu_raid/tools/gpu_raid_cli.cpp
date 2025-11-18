/**
 * GPU RAID Command-Line Interface
 *
 * Command-line tool for GPU RAID operations
 *
 * Usage:
 *   gpu_raid_cli encode --level 5 --drives 4 --input data/ --output parity/
 *   gpu_raid_cli rebuild --level 6 --drives 6 --failed 2,4 --output recovered/
 *   gpu_raid_cli bench --gpu 0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sys/stat.h>
#include <dirent.h>
#include <time.h>
#include "../include/gpu_raid.h"

#define VERSION "0.1.0"
#define MAX_FILES 256

// Command types
typedef enum {
    CMD_NONE,
    CMD_ENCODE,
    CMD_REBUILD,
    CMD_VERIFY,
    CMD_BENCH,
    CMD_INFO,
    CMD_HELP
} command_t;

// CLI arguments
typedef struct {
    command_t command;
    int raid_level;
    int num_drives;
    int gpu_id;
    char input_dir[512];
    char output_dir[512];
    char config_file[512];
    int failed_drives[16];
    int num_failed;
    bool verbose;
    size_t block_size;
} cli_args_t;

void print_usage() {
    printf("GPU RAID CLI v%s\n\n", VERSION);
    printf("Usage:\n");
    printf("  gpu_raid_cli <command> [options]\n\n");
    printf("Commands:\n");
    printf("  encode    Generate RAID parity\n");
    printf("  rebuild   Reconstruct failed drives\n");
    printf("  verify    Verify parity consistency\n");
    printf("  bench     Run performance benchmark\n");
    printf("  info      Display GPU information\n");
    printf("  help      Show this help\n\n");
    printf("Options:\n");
    printf("  -l, --level <5|6>         RAID level\n");
    printf("  -d, --drives <N>          Number of data drives\n");
    printf("  -g, --gpu <ID>            GPU device ID\n");
    printf("  -i, --input <dir>         Input directory\n");
    printf("  -o, --output <dir>        Output directory\n");
    printf("  -c, --config <file>       Configuration file\n");
    printf("  -f, --failed <list>       Failed drive indices (comma-separated)\n");
    printf("  -b, --block-size <KB>     Block size in KB\n");
    printf("  -v, --verbose             Verbose output\n");
    printf("  -h, --help                Show this help\n\n");
    printf("Examples:\n");
    printf("  # Encode 4 drives with RAID 5\n");
    printf("  gpu_raid_cli encode -l 5 -d 4 -i ./data -o ./parity\n\n");
    printf("  # Rebuild drive 2 in RAID 6\n");
    printf("  gpu_raid_cli rebuild -l 6 -d 6 -f 2 -i ./data -o ./recovered\n\n");
    printf("  # Run benchmark on GPU 0\n");
    printf("  gpu_raid_cli bench -g 0\n\n");
}

void print_gpu_info(int gpu_id) {
    gpu_raid_device_type_t type;
    uint32_t cuda_cores;
    float memory_gb;

    gpu_raid_error_t err = gpu_raid_query_device(gpu_id, &type, &cuda_cores, &memory_gb);
    if (err != GPU_RAID_SUCCESS) {
        fprintf(stderr, "Failed to query GPU %d: %s\n",
                gpu_id, gpu_raid_get_error_string(err));
        return;
    }

    const char* type_names[] = {"Auto", "RTX 3080", "RTX 3060", "Quadro RTX 4000", "Generic"};
    const char* type_name = (type >= 0 && type <= 4) ? type_names[type] : "Unknown";

    printf("\n=== GPU %d Information ===\n", gpu_id);
    printf("Model: %s\n", type_name);
    printf("CUDA Cores: %u\n", cuda_cores);
    printf("Memory: %.1f GB\n", memory_gb);

    // Additional details via CUDA
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, gpu_id) == cudaSuccess) {
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("Memory Bandwidth: %.1f GB/s\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("L2 Cache: %d KB\n", prop.l2CacheSize / 1024);
    }
    printf("========================\n\n");
}

void run_benchmark(int gpu_id) {
    printf("\n=== GPU RAID Benchmark ===\n");
    printf("GPU: %d\n\n", gpu_id);

    size_t block_sizes[] = {64*1024, 128*1024, 256*1024, 512*1024, 1024*1024};
    int drive_counts[] = {3, 4, 6};

    printf("%-12s %-12s %-15s %-15s\n", "Block Size", "Drives", "RAID 5 (GB/s)", "RAID 6 (GB/s)");
    printf("----------------------------------------------------------------\n");

    for (int d = 0; d < 3; d++) {
        int num_drives = drive_counts[d];

        for (int b = 0; b < 5; b++) {
            size_t block_size = block_sizes[b];

            // Test RAID 5
            gpu_raid_config_t config5 = {
                .raid_level = GPU_RAID_LEVEL_5,
                .num_data_drives = (uint32_t)num_drives,
                .stripe_size_kb = 256,
                .gpu_device_id = gpu_id,
                .device_type = GPU_RAID_DEVICE_AUTO,
                .memory_pool_size_mb = 1024,
                .num_streams = 1,
                .enable_profiling = false
            };

            gpu_raid_handle_t handle5;
            double raid5_gbs = 0;

            if (gpu_raid_init(&config5, &handle5) == GPU_RAID_SUCCESS) {
                uint8_t** data = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
                for (int i = 0; i < num_drives; i++) {
                    data[i] = (uint8_t*)malloc(block_size);
                    memset(data[i], i, block_size);
                }
                uint8_t* parity = (uint8_t*)malloc(block_size);

                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);

                for (int iter = 0; iter < 10; iter++) {
                    gpu_raid_encode(handle5, (const uint8_t**)data, &parity,
                                   num_drives, block_size);
                }

                clock_gettime(CLOCK_MONOTONIC, &end);
                double elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9;

                raid5_gbs = (10 * num_drives * block_size) / elapsed / (1024*1024*1024);

                for (int i = 0; i < num_drives; i++) free(data[i]);
                free(data);
                free(parity);
                gpu_raid_destroy(handle5);
            }

            // Test RAID 6
            gpu_raid_config_t config6 = config5;
            config6.raid_level = GPU_RAID_LEVEL_6;

            gpu_raid_handle_t handle6;
            double raid6_gbs = 0;

            if (gpu_raid_init(&config6, &handle6) == GPU_RAID_SUCCESS) {
                uint8_t** data = (uint8_t**)malloc(num_drives * sizeof(uint8_t*));
                for (int i = 0; i < num_drives; i++) {
                    data[i] = (uint8_t*)malloc(block_size);
                    memset(data[i], i, block_size);
                }
                uint8_t* parities[2];
                parities[0] = (uint8_t*)malloc(block_size);
                parities[1] = (uint8_t*)malloc(block_size);

                struct timespec start, end;
                clock_gettime(CLOCK_MONOTONIC, &start);

                for (int iter = 0; iter < 10; iter++) {
                    gpu_raid_encode(handle6, (const uint8_t**)data, parities,
                                   num_drives, block_size);
                }

                clock_gettime(CLOCK_MONOTONIC, &end);
                double elapsed = (end.tv_sec - start.tv_sec) +
                               (end.tv_nsec - start.tv_nsec) / 1e9;

                raid6_gbs = (10 * num_drives * block_size) / elapsed / (1024*1024*1024);

                for (int i = 0; i < num_drives; i++) free(data[i]);
                free(data);
                free(parities[0]);
                free(parities[1]);
                gpu_raid_destroy(handle6);
            }

            printf("%-12zu %-12d %-15.2f %-15.2f\n",
                   block_size / 1024, num_drives, raid5_gbs, raid6_gbs);
        }
    }

    printf("\nBenchmark complete!\n\n");
}

int parse_failed_drives(const char* str, int* failed, int max) {
    int count = 0;
    char* copy = strdup(str);
    char* token = strtok(copy, ",");

    while (token && count < max) {
        failed[count++] = atoi(token);
        token = strtok(NULL, ",");
    }

    free(copy);
    return count;
}

int main(int argc, char** argv) {
    cli_args_t args = {0};
    args.gpu_id = 0;
    args.raid_level = 5;
    args.num_drives = 4;
    args.block_size = 256 * 1024;

    // Parse command
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char* cmd_str = argv[1];
    if (strcmp(cmd_str, "encode") == 0) {
        args.command = CMD_ENCODE;
    } else if (strcmp(cmd_str, "rebuild") == 0) {
        args.command = CMD_REBUILD;
    } else if (strcmp(cmd_str, "verify") == 0) {
        args.command = CMD_VERIFY;
    } else if (strcmp(cmd_str, "bench") == 0) {
        args.command = CMD_BENCH;
    } else if (strcmp(cmd_str, "info") == 0) {
        args.command = CMD_INFO;
    } else if (strcmp(cmd_str, "help") == 0 || strcmp(cmd_str, "--help") == 0) {
        print_usage();
        return 0;
    } else {
        fprintf(stderr, "Unknown command: %s\n", cmd_str);
        print_usage();
        return 1;
    }

    // Parse options
    static struct option long_options[] = {
        {"level",      required_argument, 0, 'l'},
        {"drives",     required_argument, 0, 'd'},
        {"gpu",        required_argument, 0, 'g'},
        {"input",      required_argument, 0, 'i'},
        {"output",     required_argument, 0, 'o'},
        {"config",     required_argument, 0, 'c'},
        {"failed",     required_argument, 0, 'f'},
        {"block-size", required_argument, 0, 'b'},
        {"verbose",    no_argument,       0, 'v'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc - 1, argv + 1, "l:d:g:i:o:c:f:b:vh", long_options, NULL)) != -1) {
        switch (opt) {
            case 'l': args.raid_level = atoi(optarg); break;
            case 'd': args.num_drives = atoi(optarg); break;
            case 'g': args.gpu_id = atoi(optarg); break;
            case 'i': strncpy(args.input_dir, optarg, sizeof(args.input_dir)-1); break;
            case 'o': strncpy(args.output_dir, optarg, sizeof(args.output_dir)-1); break;
            case 'c': strncpy(args.config_file, optarg, sizeof(args.config_file)-1); break;
            case 'f': args.num_failed = parse_failed_drives(optarg, args.failed_drives, 16); break;
            case 'b': args.block_size = atoi(optarg) * 1024; break;
            case 'v': args.verbose = true; break;
            case 'h': print_usage(); return 0;
            default: print_usage(); return 1;
        }
    }

    // Execute command
    switch (args.command) {
        case CMD_INFO:
            print_gpu_info(args.gpu_id);
            break;

        case CMD_BENCH:
            run_benchmark(args.gpu_id);
            break;

        case CMD_ENCODE:
            printf("Encode command not yet implemented\n");
            printf("Use API example: examples/simple_raid5_example\n");
            break;

        case CMD_REBUILD:
            printf("Rebuild command not yet implemented\n");
            printf("Use API example: examples/raid6_dual_failure_example\n");
            break;

        case CMD_VERIFY:
            printf("Verify command not yet implemented\n");
            break;

        default:
            fprintf(stderr, "Command not implemented\n");
            return 1;
    }

    return 0;
}
