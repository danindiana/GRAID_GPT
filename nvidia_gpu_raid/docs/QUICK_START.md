## Quick Start Guide

### Prerequisites

1. **NVIDIA GPU** - One of the following:
   - GeForce RTX 3080 (recommended)
   - GeForce RTX 3060
   - Quadro RTX 4000

2. **Software Requirements:**
   ```bash
   # CUDA Toolkit 11.8+ or 12.x
   sudo apt-get install nvidia-cuda-toolkit

   # Build tools
   sudo apt-get install build-essential cmake

   # Optional: nvidia-smi for GPU monitoring
   sudo apt-get install nvidia-utils-535  # or your driver version
   ```

3. **Verify GPU:**
   ```bash
   nvidia-smi
   ```

### Building

```bash
cd nvidia_gpu_raid
mkdir build && cd build

# Configure (auto-detect GPU architecture)
cmake ..

# Or specify GPU architecture manually:
# cmake .. -DCUDA_ARCH=86  # For RTX 3080/3060
# cmake .. -DCUDA_ARCH=75  # For Quadro RTX 4000

# Build
make -j$(nproc)

# Run tests (optional)
ctest --output-on-failure
```

### Running Examples

#### Simple RAID 5 Example

```bash
cd build
./simple_raid5_example
```

Expected output:
```
=== GPU RAID 5 Simple Example ===

Step 1: Initializing GPU RAID 5...
  Version: 0.1.0
  Initialized successfully!

Step 2: Preparing data blocks...
  Data block 0: filled with pattern 0x00
  Data block 1: filled with pattern 0x11
  Data block 2: filled with pattern 0x22
  Data block 3: filled with pattern 0x33
  Total data: 1024 KB

Step 3: Generating parity...
  Parity generated successfully
  Encoding time: 2.456 ms
  Throughput: 15.23 GB/s

... (continues)
```

#### RAID 6 Dual Failure Example

```bash
./raid6_dual_failure_example
```

This demonstrates recovery from two simultaneous drive failures using Reed-Solomon P+Q parity.

### Using the Library in Your Code

#### Basic RAID 5 Usage

```c
#include <gpu_raid.h>

// 1. Configure
gpu_raid_config_t config = {
    .raid_level = GPU_RAID_LEVEL_5,
    .num_data_drives = 4,
    .stripe_size_kb = 256,
    .gpu_device_id = 0,
    .device_type = GPU_RAID_DEVICE_AUTO,
    .enable_tenstorrent = false,
    .memory_pool_size_mb = 512,
    .num_streams = 1,
    .enable_profiling = true
};

// 2. Initialize
gpu_raid_handle_t handle;
gpu_raid_error_t err = gpu_raid_init(&config, &handle);
if (err != GPU_RAID_SUCCESS) {
    fprintf(stderr, "Init failed: %s\n", gpu_raid_get_error_string(err));
    return -1;
}

// 3. Prepare data
uint8_t* data_blocks[4];
for (int i = 0; i < 4; i++) {
    data_blocks[i] = malloc(BLOCK_SIZE);
    // ... fill with data
}
uint8_t* parity = malloc(BLOCK_SIZE);

// 4. Encode (generate parity)
err = gpu_raid_encode(
    handle,
    (const uint8_t**)data_blocks,
    &parity,
    4,
    BLOCK_SIZE
);

// 5. Recover from failure (if needed)
if (drive_failed) {
    const uint8_t* all_blocks[4] = {
        data_blocks[0],
        data_blocks[1],
        NULL,  // Failed drive
        data_blocks[3]
    };

    uint32_t failed_idx = 2;
    uint8_t* recovered = malloc(BLOCK_SIZE);

    err = gpu_raid_reconstruct(
        handle,
        all_blocks,
        (const uint8_t**)&parity,
        &failed_idx,
        1,  // Number of failures
        &recovered,
        4,
        BLOCK_SIZE
    );
}

// 6. Cleanup
gpu_raid_destroy(handle);
```

#### RAID 6 Usage

```c
// Same as RAID 5, but:
config.raid_level = GPU_RAID_LEVEL_6;

// Allocate 2 parity blocks (P and Q)
uint8_t* parities[2];
parities[0] = malloc(BLOCK_SIZE);  // P parity
parities[1] = malloc(BLOCK_SIZE);  // Q parity

// Encode generates both parities
gpu_raid_encode(handle, data_blocks, parities, num_drives, BLOCK_SIZE);

// Can recover from up to 2 failures
uint32_t failed_indices[2] = {1, 3};
uint8_t* recovered[2];
recovered[0] = malloc(BLOCK_SIZE);
recovered[1] = malloc(BLOCK_SIZE);

gpu_raid_reconstruct(
    handle, all_blocks, (const uint8_t**)parities,
    failed_indices, 2, recovered, num_drives, BLOCK_SIZE
);
```

### Performance Tuning

#### GPU-Specific Configuration

Load optimized settings for your GPU:

```c
gpu_raid_config_t config;
gpu_raid_load_config("config/rtx3080_config.json", &config);
```

Or manually tune:

```c
// For RTX 3080 (high performance)
config.stripe_size_kb = 256;
config.memory_pool_size_mb = 2048;
config.num_streams = 4;

// For RTX 3060 (balanced)
config.stripe_size_kb = 128;
config.memory_pool_size_mb = 1024;
config.num_streams = 2;

// For Quadro RTX 4000 (workstation)
config.stripe_size_kb = 128;
config.memory_pool_size_mb = 1024;
config.num_streams = 2;
```

#### Monitoring Performance

```c
gpu_raid_stats_t stats;
gpu_raid_get_stats(handle, &stats);

printf("Throughput: %.2f GB/s\n", stats.peak_throughput_gbs);
printf("GPU Utilization: %u%%\n", stats.gpu_utilization_percent);
printf("Temperature: %.1f°C\n", stats.gpu_temperature_c);
```

### Benchmarking

Run the throughput benchmark:

```bash
cd build
./bench_throughput
```

This will test various block sizes and drive counts, displaying throughput for your GPU.

### Troubleshooting

#### "No CUDA-capable device detected"

```bash
# Check if GPU is visible
nvidia-smi

# Verify CUDA installation
nvcc --version

# Check driver version
cat /proc/driver/nvidia/version
```

#### Low Performance

1. **Check GPU clock speed:**
   ```bash
   nvidia-smi -q | grep -A 2 "Clocks"
   ```

2. **Monitor thermal throttling:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   If temperature > 80°C, improve cooling.

3. **Increase memory pool:**
   ```c
   config.memory_pool_size_mb = 4096;  // Larger pool
   ```

4. **Use multiple streams:**
   ```c
   config.num_streams = 4;  // Parallel operations
   ```

#### Build Errors

**CUDA architecture mismatch:**
```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Build with correct arch (e.g., for 8.6):
cmake .. -DCUDA_ARCH=86
```

**Missing CUDA libraries:**
```bash
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Next Steps

- Read [API.md](API.md) for complete API documentation
- Check [PERFORMANCE.md](PERFORMANCE.md) for optimization tips
- Review [examples/](../examples/) for more use cases
- Run benchmarks to baseline your system
- Integrate with your storage application

### Getting Help

- Check existing issues: [GitHub Issues](../../issues)
- Read troubleshooting guide: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Review performance guide: [PERFORMANCE.md](PERFORMANCE.md)
