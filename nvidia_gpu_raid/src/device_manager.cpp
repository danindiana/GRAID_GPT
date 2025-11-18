/**
 * GPU Device Manager
 *
 * Handles GPU detection, initialization, and capability queries
 * Automatically identifies RTX 3080, RTX 3060, and Quadro RTX 4000
 */

#include "../include/gpu_raid.h"
#include "../include/cuda_utils.h"
#include "../include/raid_types.h"
#include <cuda_runtime.h>
#include <string.h>
#include <stdio.h>

// GPU identification by name patterns
static gpu_raid_device_type_t identify_gpu(const char* name) {
    if (strstr(name, "RTX 3080")) {
        return GPU_RAID_DEVICE_RTX_3080;
    } else if (strstr(name, "RTX 3060")) {
        return GPU_RAID_DEVICE_RTX_3060;
    } else if (strstr(name, "Quadro RTX 4000")) {
        return GPU_RAID_DEVICE_QUADRO_4000;
    }
    return GPU_RAID_DEVICE_GENERIC;
}

// Estimate CUDA cores from compute capability and SM count
static uint32_t estimate_cuda_cores(int major, int minor, int sm_count) {
    int cores_per_sm = 0;

    // Ampere (8.x)
    if (major == 8) {
        cores_per_sm = 128;  // RTX 30xx series
    }
    // Turing (7.5)
    else if (major == 7 && minor == 5) {
        cores_per_sm = 64;   // Quadro RTX 4000
    }
    // Volta (7.0)
    else if (major == 7 && minor == 0) {
        cores_per_sm = 64;
    }
    // Pascal (6.x)
    else if (major == 6) {
        cores_per_sm = 64;
    }
    // Maxwell (5.x)
    else if (major == 5) {
        cores_per_sm = 128;
    }
    // Default estimate
    else {
        cores_per_sm = 64;
    }

    return sm_count * cores_per_sm;
}

/**
 * Query GPU device capabilities
 */
gpu_raid_error_t gpu_raid_query_device(
    int device_id,
    gpu_raid_device_type_t* device_type,
    uint32_t* cuda_cores,
    float* memory_gb
) {
    int device_count = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&device_count), GPU_RAID_ERROR_NO_GPU);

    if (device_count == 0) {
        return GPU_RAID_ERROR_NO_GPU;
    }

    if (device_id < 0 || device_id >= device_count) {
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    cudaDeviceProp prop;
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop, device_id),
                      GPU_RAID_ERROR_GPU_INIT_FAILED);

    // Identify GPU type
    if (device_type) {
        *device_type = identify_gpu(prop.name);
    }

    // Calculate CUDA cores
    if (cuda_cores) {
        *cuda_cores = estimate_cuda_cores(
            prop.major,
            prop.minor,
            prop.multiProcessorCount
        );
    }

    // Memory in GB
    if (memory_gb) {
        *memory_gb = prop.totalGlobalMem / (1024.0f * 1024.0f * 1024.0f);
    }

    return GPU_RAID_SUCCESS;
}

/**
 * Initialize GPU device for RAID operations
 */
gpu_raid_error_t device_manager_init(gpu_raid_context_t* ctx) {
    int device_count = 0;
    CUDA_CHECK_RETURN(cudaGetDeviceCount(&device_count), GPU_RAID_ERROR_NO_GPU);

    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable GPU found\n");
        return GPU_RAID_ERROR_NO_GPU;
    }

    // Determine device ID
    int device_id = ctx->config.gpu_device_id;
    if (device_id < 0) {
        // Auto-select: use device 0
        device_id = 0;
    }

    if (device_id >= device_count) {
        fprintf(stderr, "Invalid GPU device ID: %d (only %d devices available)\n",
                device_id, device_count);
        return GPU_RAID_ERROR_INVALID_PARAM;
    }

    ctx->cuda_device_id = device_id;

    // Set device
    CUDA_CHECK_RETURN(cudaSetDevice(device_id), GPU_RAID_ERROR_GPU_INIT_FAILED);

    // Get device properties
    CUDA_CHECK_RETURN(cudaGetDeviceProperties(&ctx->device_props, device_id),
                      GPU_RAID_ERROR_GPU_INIT_FAILED);

    // Identify GPU type
    ctx->device_type = identify_gpu(ctx->device_props.name);

    // Calculate CUDA cores
    ctx->cuda_cores = estimate_cuda_cores(
        ctx->device_props.major,
        ctx->device_props.minor,
        ctx->device_props.multiProcessorCount
    );

    ctx->total_memory = ctx->device_props.totalGlobalMem;

    // Verify compute capability
    int compute_cap = ctx->device_props.major * 10 + ctx->device_props.minor;

    if (compute_cap < 70) {
        fprintf(stderr, "Warning: GPU compute capability %d.%d is below recommended 7.0\n",
                ctx->device_props.major, ctx->device_props.minor);
    }

    // Print device info
    printf("GPU RAID Device Information:\n");
    printf("  Device %d: %s\n", device_id, ctx->device_props.name);
    printf("  Compute Capability: %d.%d\n",
           ctx->device_props.major, ctx->device_props.minor);
    printf("  CUDA Cores: %u\n", ctx->cuda_cores);
    printf("  Total Memory: %.2f GB\n",
           ctx->total_memory / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Bandwidth: %.1f GB/s\n",
           2.0 * ctx->device_props.memoryClockRate * (ctx->device_props.memoryBusWidth / 8) / 1.0e6);
    printf("  L2 Cache: %d KB\n", ctx->device_props.l2CacheSize / 1024);
    printf("  SMs: %d\n", ctx->device_props.multiProcessorCount);
    printf("  Max Threads/Block: %d\n", ctx->device_props.maxThreadsPerBlock);
    printf("\n");

    return GPU_RAID_SUCCESS;
}

/**
 * Create CUDA streams for async operations
 */
gpu_raid_error_t device_manager_create_streams(gpu_raid_context_t* ctx) {
    uint32_t num_streams = ctx->config.num_streams;
    if (num_streams > MAX_CUDA_STREAMS) {
        num_streams = MAX_CUDA_STREAMS;
    }

    ctx->num_streams = num_streams;

    for (uint32_t i = 0; i < num_streams; i++) {
        CUDA_CHECK_RETURN(
            cudaStreamCreateWithFlags(&ctx->streams[i], cudaStreamNonBlocking),
            GPU_RAID_ERROR_GPU_INIT_FAILED
        );
    }

    printf("Created %u CUDA streams for async operations\n", num_streams);
    return GPU_RAID_SUCCESS;
}

/**
 * Destroy CUDA streams
 */
void device_manager_destroy_streams(gpu_raid_context_t* ctx) {
    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        if (ctx->streams[i]) {
            cudaStreamDestroy(ctx->streams[i]);
            ctx->streams[i] = nullptr;
        }
    }
    ctx->num_streams = 0;
}

/**
 * Synchronize device
 */
gpu_raid_error_t device_manager_sync(gpu_raid_context_t* ctx) {
    CUDA_CHECK_RETURN(cudaDeviceSynchronize(), GPU_RAID_ERROR_CUDA_ERROR);
    return GPU_RAID_SUCCESS;
}

/**
 * Get GPU temperature (if supported)
 */
float device_manager_get_temperature(gpu_raid_context_t* ctx) {
    // Note: Direct temperature query via CUDA Runtime API is not available
    // Would need NVML (NVIDIA Management Library) for this
    // For now, return placeholder
    return 0.0f;
}

/**
 * Get GPU power consumption (if supported)
 */
uint32_t device_manager_get_power(gpu_raid_context_t* ctx) {
    // Note: Power monitoring requires NVML
    // Return placeholder
    return 0;
}

/**
 * Get GPU utilization (if supported)
 */
uint32_t device_manager_get_utilization(gpu_raid_context_t* ctx) {
    // Note: Utilization monitoring requires NVML
    // Return placeholder
    return 0;
}

/**
 * Check if peer-to-peer access is available (for multi-GPU)
 */
bool device_manager_check_p2p(int device1, int device2) {
    int can_access = 0;
    cudaError_t err = cudaDeviceCanAccessPeer(&can_access, device1, device2);

    if (err != cudaSuccess) {
        return false;
    }

    return can_access != 0;
}

/**
 * Enable peer-to-peer access between GPUs
 */
gpu_raid_error_t device_manager_enable_p2p(int device1, int device2) {
    cudaError_t err = cudaDeviceEnablePeerAccess(device2, 0);

    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        return GPU_RAID_SUCCESS;  // Already enabled is OK
    }

    CUDA_CHECK_RETURN(err, GPU_RAID_ERROR_GPU_INIT_FAILED);
    return GPU_RAID_SUCCESS;
}

/**
 * Reset GPU device
 */
void device_manager_reset(gpu_raid_context_t* ctx) {
    if (ctx->cuda_device_id >= 0) {
        cudaSetDevice(ctx->cuda_device_id);
        cudaDeviceReset();
    }
}

/**
 * Print detailed device information
 */
void device_manager_print_info(gpu_raid_context_t* ctx) {
    cudaDeviceProp* prop = &ctx->device_props;

    printf("\n========================================\n");
    printf("Detailed GPU Information\n");
    printf("========================================\n");
    printf("Device Name: %s\n", prop->name);
    printf("Compute Capability: %d.%d\n", prop->major, prop->minor);
    printf("Total Global Memory: %zu MB\n", prop->totalGlobalMem / (1024 * 1024));
    printf("Shared Memory per Block: %zu KB\n", prop->sharedMemPerBlock / 1024);
    printf("Registers per Block: %d\n", prop->regsPerBlock);
    printf("Warp Size: %d\n", prop->warpSize);
    printf("Max Threads per Block: %d\n", prop->maxThreadsPerBlock);
    printf("Max Threads Dimensions: [%d, %d, %d]\n",
           prop->maxThreadsDim[0], prop->maxThreadsDim[1], prop->maxThreadsDim[2]);
    printf("Max Grid Size: [%d, %d, %d]\n",
           prop->maxGridSize[0], prop->maxGridSize[1], prop->maxGridSize[2]);
    printf("Multiprocessor Count: %d\n", prop->multiProcessorCount);
    printf("Clock Rate: %d MHz\n", prop->clockRate / 1000);
    printf("Memory Clock Rate: %d MHz\n", prop->memoryClockRate / 1000);
    printf("Memory Bus Width: %d-bit\n", prop->memoryBusWidth);
    printf("L2 Cache Size: %d KB\n", prop->l2CacheSize / 1024);
    printf("Max Texture Dimension 1D: %d\n", prop->maxTexture1D);
    printf("Max Texture Dimension 2D: [%d, %d]\n",
           prop->maxTexture2D[0], prop->maxTexture2D[1]);
    printf("Concurrent Kernels: %s\n", prop->concurrentKernels ? "Yes" : "No");
    printf("ECC Enabled: %s\n", prop->ECCEnabled ? "Yes" : "No");
    printf("Unified Addressing: %s\n", prop->unifiedAddressing ? "Yes" : "No");
    printf("Managed Memory: %s\n", prop->managedMemory ? "Yes" : "No");
    printf("Multi-GPU Board: %s\n", prop->isMultiGpuBoard ? "Yes" : "No");
    printf("PCI Bus ID: %d\n", prop->pciBusID);
    printf("PCI Device ID: %d\n", prop->pciDeviceID);
    printf("PCI Domain ID: %d\n", prop->pciDomainID);
    printf("Async Engine Count: %d\n", prop->asyncEngineCount);
    printf("Memory Pools Supported: %s\n", prop->memoryPoolsSupported ? "Yes" : "No");
    printf("========================================\n\n");
}
