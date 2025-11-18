/**
 * Performance Monitor
 *
 * Collects and reports performance metrics
 * Integrates with NVML for GPU monitoring if available
 */

#include "../include/gpu_raid.h"
#include "../include/raid_types.h"
#include <stdio.h>
#include <time.h>

// NVML integration (optional)
#ifdef ENABLE_NVML
#include <nvml.h>
static bool g_nvml_initialized = false;
static nvmlDevice_t g_nvml_device;
#endif

/**
 * Initialize performance monitoring
 */
gpu_raid_error_t performance_monitor_init(gpu_raid_context_t* ctx) {
#ifdef ENABLE_NVML
    nvmlReturn_t result = nvmlInit();
    if (result == NVML_SUCCESS) {
        result = nvmlDeviceGetHandleByIndex(ctx->cuda_device_id, &g_nvml_device);
        if (result == NVML_SUCCESS) {
            g_nvml_initialized = true;
            printf("NVML monitoring enabled\n");
        }
    }
#endif

    return GPU_RAID_SUCCESS;
}

/**
 * Shutdown performance monitoring
 */
void performance_monitor_shutdown() {
#ifdef ENABLE_NVML
    if (g_nvml_initialized) {
        nvmlShutdown();
        g_nvml_initialized = false;
    }
#endif
}

/**
 * Update GPU metrics (temperature, power, utilization)
 */
void performance_monitor_update_gpu_metrics(gpu_raid_context_t* ctx) {
#ifdef ENABLE_NVML
    if (!g_nvml_initialized) return;

    unsigned int temp = 0;
    nvmlDeviceGetTemperature(g_nvml_device, NVML_TEMPERATURE_GPU, &temp);
    ctx->stats.gpu_temperature_c = (float)temp;

    unsigned int power = 0;
    nvmlDeviceGetPowerUsage(g_nvml_device, &power);
    ctx->stats.gpu_power_watts = power / 1000;  // mW to W

    nvmlUtilization_t util;
    nvmlDeviceGetUtilizationRates(g_nvml_device, &util);
    ctx->stats.gpu_utilization_percent = util.gpu;
#else
    // Placeholder values if NVML not available
    ctx->stats.gpu_temperature_c = 0.0f;
    ctx->stats.gpu_power_watts = 0;
    ctx->stats.gpu_utilization_percent = 0;
#endif
}

/**
 * Record encode operation
 */
void performance_monitor_record_encode(
    gpu_raid_context_t* ctx,
    size_t bytes,
    double time_ms
) {
    ctx->stats.total_encodes++;
    ctx->stats.total_bytes_encoded += bytes;

    // Update average
    ctx->stats.avg_encode_time_ms =
        (ctx->stats.avg_encode_time_ms * (ctx->stats.total_encodes - 1) +
         time_ms) / ctx->stats.total_encodes;

    // Update peak throughput
    double throughput = bytes / (time_ms / 1000.0) / (1024.0 * 1024.0 * 1024.0);
    if (throughput > ctx->stats.peak_throughput_gbs) {
        ctx->stats.peak_throughput_gbs = throughput;
    }

    performance_monitor_update_gpu_metrics(ctx);
}

/**
 * Record decode/rebuild operation
 */
void performance_monitor_record_decode(
    gpu_raid_context_t* ctx,
    size_t bytes,
    double time_ms
) {
    ctx->stats.total_decodes++;
    ctx->stats.total_bytes_decoded += bytes;

    ctx->stats.avg_decode_time_ms =
        (ctx->stats.avg_decode_time_ms * (ctx->stats.total_decodes - 1) +
         time_ms) / ctx->stats.total_decodes;

    performance_monitor_update_gpu_metrics(ctx);
}

/**
 * Print performance report
 */
void performance_monitor_print_report(gpu_raid_context_t* ctx) {
    printf("\n=== Performance Report ===\n");
    printf("Operations:\n");
    printf("  Total Encodes: %lu\n", ctx->stats.total_encodes);
    printf("  Total Decodes: %lu\n", ctx->stats.total_decodes);
    printf("\nData Processed:\n");
    printf("  Encoded: %.2f GB\n",
           ctx->stats.total_bytes_encoded / (1024.0 * 1024.0 * 1024.0));
    printf("  Decoded: %.2f GB\n",
           ctx->stats.total_bytes_decoded / (1024.0 * 1024.0 * 1024.0));
    printf("\nTiming:\n");
    printf("  Avg Encode Time: %.3f ms\n", ctx->stats.avg_encode_time_ms);
    printf("  Avg Decode Time: %.3f ms\n", ctx->stats.avg_decode_time_ms);
    printf("\nThroughput:\n");
    printf("  Peak: %.2f GB/s\n", ctx->stats.peak_throughput_gbs);
    printf("\nGPU Metrics:\n");
    printf("  Utilization: %u%%\n", ctx->stats.gpu_utilization_percent);
    printf("  Temperature: %.1f°C\n", ctx->stats.gpu_temperature_c);
    printf("  Power: %u W\n", ctx->stats.gpu_power_watts);
    printf("==========================\n\n");
}

/**
 * Get current timestamp in milliseconds
 */
double performance_monitor_get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
