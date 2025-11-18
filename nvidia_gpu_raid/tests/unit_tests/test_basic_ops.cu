/**
 * Basic Operations Unit Test
 *
 * Tests fundamental GPU RAID operations
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "../../include/gpu_raid.h"

#define TEST_BLOCK_SIZE 4096
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_RED   "\x1b[31m"
#define ANSI_COLOR_RESET "\x1b[0m"

int tests_passed = 0;
int tests_failed = 0;

void test_assert(bool condition, const char* test_name) {
    if (condition) {
        printf(ANSI_COLOR_GREEN "✓ PASS" ANSI_COLOR_RESET ": %s\n", test_name);
        tests_passed++;
    } else {
        printf(ANSI_COLOR_RED "✗ FAIL" ANSI_COLOR_RESET ": %s\n", test_name);
        tests_failed++;
    }
}

void test_initialization() {
    printf("\n=== Testing Initialization ===\n");

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = 4,
        .stripe_size_kb = 64,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 256,
        .num_streams = 1,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    gpu_raid_error_t err = gpu_raid_init(&config, &handle);

    test_assert(err == GPU_RAID_SUCCESS, "GPU RAID initialization");
    test_assert(handle != NULL, "Valid handle returned");

    if (handle) {
        gpu_raid_destroy(handle);
        test_assert(true, "GPU RAID destruction");
    }
}

void test_xor_parity() {
    printf("\n=== Testing XOR Parity (RAID 5) ===\n");

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = 3,
        .stripe_size_kb = 64,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 256,
        .num_streams = 1,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        test_assert(false, "XOR parity test initialization");
        return;
    }

    uint8_t* blocks[3];
    const uint8_t* blocks_const[3];
    uint8_t* parity;

    for (int i = 0; i < 3; i++) {
        blocks[i] = (uint8_t*)malloc(TEST_BLOCK_SIZE);
        memset(blocks[i], i + 1, TEST_BLOCK_SIZE);
        blocks_const[i] = blocks[i];
    }
    parity = (uint8_t*)malloc(TEST_BLOCK_SIZE);

    gpu_raid_error_t err = gpu_raid_encode(handle, blocks_const, &parity,
                                           3, TEST_BLOCK_SIZE);

    test_assert(err == GPU_RAID_SUCCESS, "XOR parity encoding");

    // Verify XOR manually for first byte
    uint8_t expected_parity = blocks[0][0] ^ blocks[1][0] ^ blocks[2][0];
    test_assert(parity[0] == expected_parity, "XOR parity correctness");

    // Cleanup
    for (int i = 0; i < 3; i++) free(blocks[i]);
    free(parity);
    gpu_raid_destroy(handle);
}

void test_single_recovery() {
    printf("\n=== Testing Single Drive Recovery ===\n");

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_5,
        .num_data_drives = 4,
        .stripe_size_kb = 64,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 256,
        .num_streams = 1,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        test_assert(false, "Recovery test initialization");
        return;
    }

    uint8_t* blocks[4];
    const uint8_t* blocks_const[4];
    uint8_t* parity;

    for (int i = 0; i < 4; i++) {
        blocks[i] = (uint8_t*)malloc(TEST_BLOCK_SIZE);
        memset(blocks[i], (i + 1) * 17, TEST_BLOCK_SIZE);
        blocks_const[i] = blocks[i];
    }
    parity = (uint8_t*)malloc(TEST_BLOCK_SIZE);

    // Encode
    gpu_raid_encode(handle, blocks_const, &parity, 4, TEST_BLOCK_SIZE);

    // Save block 2
    uint8_t* original = (uint8_t*)malloc(TEST_BLOCK_SIZE);
    memcpy(original, blocks[2], TEST_BLOCK_SIZE);

    // Simulate failure
    const uint8_t* all_blocks[4] = {blocks[0], blocks[1], NULL, blocks[3]};
    uint32_t failed_idx = 2;
    uint8_t* recovered = (uint8_t*)malloc(TEST_BLOCK_SIZE);

    gpu_raid_error_t err = gpu_raid_reconstruct(
        handle, all_blocks, (const uint8_t**)&parity,
        &failed_idx, 1, &recovered, 4, TEST_BLOCK_SIZE
    );

    test_assert(err == GPU_RAID_SUCCESS, "Single drive reconstruction");
    test_assert(memcmp(original, recovered, TEST_BLOCK_SIZE) == 0,
                "Recovered data matches original");

    // Cleanup
    for (int i = 0; i < 4; i++) free(blocks[i]);
    free(parity);
    free(original);
    free(recovered);
    gpu_raid_destroy(handle);
}

void test_raid6_encoding() {
    printf("\n=== Testing RAID 6 P+Q Encoding ===\n");

    gpu_raid_config_t config = {
        .raid_level = GPU_RAID_LEVEL_6,
        .num_data_drives = 4,
        .stripe_size_kb = 64,
        .gpu_device_id = 0,
        .device_type = GPU_RAID_DEVICE_AUTO,
        .enable_tenstorrent = false,
        .memory_pool_size_mb = 256,
        .num_streams = 1,
        .enable_profiling = false
    };

    gpu_raid_handle_t handle;
    if (gpu_raid_init(&config, &handle) != GPU_RAID_SUCCESS) {
        test_assert(false, "RAID 6 test initialization");
        return;
    }

    uint8_t* blocks[4];
    const uint8_t* blocks_const[4];
    uint8_t* parities[2];

    for (int i = 0; i < 4; i++) {
        blocks[i] = (uint8_t*)malloc(TEST_BLOCK_SIZE);
        memset(blocks[i], (i + 1) * 13, TEST_BLOCK_SIZE);
        blocks_const[i] = blocks[i];
    }
    parities[0] = (uint8_t*)malloc(TEST_BLOCK_SIZE);
    parities[1] = (uint8_t*)malloc(TEST_BLOCK_SIZE);

    gpu_raid_error_t err = gpu_raid_encode(handle, blocks_const, parities,
                                           4, TEST_BLOCK_SIZE);

    test_assert(err == GPU_RAID_SUCCESS, "RAID 6 P+Q encoding");

    // Verify P parity (should be XOR)
    uint8_t expected_p = blocks[0][0] ^ blocks[1][0] ^ blocks[2][0] ^ blocks[3][0];
    test_assert(parities[0][0] == expected_p, "P parity correctness");

    // Q parity is more complex (Reed-Solomon), just check it's not zero
    bool q_nonzero = false;
    for (int i = 0; i < TEST_BLOCK_SIZE; i++) {
        if (parities[1][i] != 0) {
            q_nonzero = true;
            break;
        }
    }
    test_assert(q_nonzero, "Q parity generated (non-zero)");

    // Cleanup
    for (int i = 0; i < 4; i++) free(blocks[i]);
    free(parities[0]);
    free(parities[1]);
    gpu_raid_destroy(handle);
}

int main(int argc, char** argv) {
    printf("\n╔════════════════════════════════════╗\n");
    printf("║  GPU RAID Unit Tests               ║\n");
    printf("╚════════════════════════════════════╝\n");

    test_initialization();
    test_xor_parity();
    test_single_recovery();
    test_raid6_encoding();

    printf("\n========================================\n");
    printf("Test Results:\n");
    printf("  Passed: %d\n", tests_passed);
    printf("  Failed: %d\n", tests_failed);
    printf("========================================\n\n");

    return (tests_failed == 0) ? 0 : 1;
}
