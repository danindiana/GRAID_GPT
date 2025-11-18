/**
 * CUDA Utility Functions
 *
 * Helper functions for CUDA operations, error checking, and debugging
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return err; \
        } \
    } while (0)

// CUDA error checking macro with custom error handling
#define CUDA_CHECK_RETURN(call, retval) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            return retval; \
        } \
    } while (0)

// CUDA kernel launch error check
#define CUDA_CHECK_LAST() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get GPU device name
 */
inline const char* cuda_get_device_name(int device_id) {
    static char name[256];
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        snprintf(name, sizeof(name), "%s", prop.name);
        return name;
    }
    return "Unknown GPU";
}

/**
 * Get GPU compute capability
 */
inline int cuda_get_compute_capability(int device_id) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.major * 10 + prop.minor;
    }
    return 0;
}

/**
 * Get GPU memory in bytes
 */
inline size_t cuda_get_total_memory(int device_id) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) == cudaSuccess) {
        return prop.totalGlobalMem;
    }
    return 0;
}

/**
 * Print GPU information
 */
inline void cuda_print_device_info(int device_id) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties\n");
        return;
    }

    printf("GPU Device %d: %s\n", device_id, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  Memory Clock: %d MHz\n", prop.memoryClockRate / 1000);
    printf("  Memory Bus Width: %d-bit\n", prop.memoryBusWidth);
    printf("  L2 Cache Size: %d KB\n", prop.l2CacheSize / 1024);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Number of SMs: %d\n", prop.multiProcessorCount);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  PCIe Bus ID: %d:%d.%d\n", prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
}

/**
 * Allocate pinned host memory (for fast transfers)
 */
inline cudaError_t cuda_alloc_pinned(void** ptr, size_t size) {
    return cudaMallocHost(ptr, size);
}

/**
 * Free pinned host memory
 */
inline cudaError_t cuda_free_pinned(void* ptr) {
    return cudaFreeHost(ptr);
}

/**
 * Allocate device memory
 */
inline cudaError_t cuda_alloc_device(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

/**
 * Free device memory
 */
inline cudaError_t cuda_free_device(void* ptr) {
    return cudaFree(ptr);
}

/**
 * Memset device memory
 */
inline cudaError_t cuda_memset_device(void* ptr, int value, size_t size) {
    return cudaMemset(ptr, value, size);
}

#ifdef __cplusplus
}
#endif

#endif // CUDA_UTILS_H
