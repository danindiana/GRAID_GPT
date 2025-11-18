## Performance Tuning Guide

Optimize GPU RAID for maximum throughput on RTX 3080, RTX 3060, and Quadro RTX 4000.

### Table of Contents

- [Hardware Configuration](#hardware-configuration)
- [Software Configuration](#software-configuration)
- [Block Size Tuning](#block-size-tuning)
- [Memory Optimization](#memory-optimization)
- [Async Operations](#async-operations)
- [Multi-GPU](#multi-gpu)
- [Benchmarking](#benchmarking)
- [Common Bottlenecks](#common-bottlenecks)

---

## Hardware Configuration

### GPU Selection

| GPU Model | Best For | Expected Throughput |
|-----------|----------|---------------------|
| RTX 3080 | High-performance RAID, fast rebuilds | 15-25 GB/s encode |
| RTX 3060 | Cost-effective, balanced | 8-15 GB/s encode |
| Quadro RTX 4000 | 24/7 workstation, ECC-like reliability | 10-17 GB/s encode |

### PCIe Configuration

**Critical:** Ensure GPU is in PCIe 4.0 x16 slot for maximum bandwidth.

```bash
# Check PCIe link speed
lspci -vv | grep -A 10 "VGA compatible" | grep LnkSta

# Expected output:
# LnkSta: Speed 16GT/s (ok), Width x16 (ok)
```

**PCIe Bandwidth:**
- PCIe 4.0 x16: ~32 GB/s bidirectional
- PCIe 3.0 x16: ~16 GB/s bidirectional

**Fix slow PCIe:**
1. Move GPU to top PCIe slot (usually x16)
2. Check BIOS - enable PCIe 4.0
3. Verify riser cable supports PCIe 4.0

### NVMe Drive Configuration

For best RAID performance:
- **Use NVMe SSDs** on PCIe 4.0 (not SATA)
- **Direct PCIe lanes** to CPU (not chipset)
- **Match drive count** to GPU capabilities

**Recommended:**
- RTX 3080: 6-8 NVMe drives
- RTX 3060: 4-6 NVMe drives
- Quadro 4000: 4-6 NVMe drives

---

## Software Configuration

### Optimal Configuration by GPU

#### RTX 3080

```c
gpu_raid_config_t config = {
    .raid_level = GPU_RAID_LEVEL_6,
    .num_data_drives = 6,
    .stripe_size_kb = 256,           // Large stripes
    .memory_pool_size_mb = 4096,     // 4 GB pool
    .num_streams = 4,                // High parallelism
    .enable_profiling = true
};
```

#### RTX 3060

```c
gpu_raid_config_t config = {
    .raid_level = GPU_RAID_LEVEL_5,
    .num_data_drives = 4,
    .stripe_size_kb = 128,           // Medium stripes
    .memory_pool_size_mb = 2048,     // 2 GB pool
    .num_streams = 2,                // Moderate parallelism
    .enable_profiling = true
};
```

#### Quadro RTX 4000

```c
gpu_raid_config_t config = {
    .raid_level = GPU_RAID_LEVEL_6,
    .num_data_drives = 4,
    .stripe_size_kb = 128,
    .memory_pool_size_mb = 2048,
    .num_streams = 2,
    .enable_profiling = true
};
```

### Load from JSON

```bash
# Use pre-tuned configurations
cp config/rtx3080_config.json my_config.json

# Customize as needed
vim my_config.json
```

```c
gpu_raid_config_t config;
gpu_raid_load_config("my_config.json", &config);
```

---

## Block Size Tuning

Block size significantly impacts throughput.

### Benchmark Block Sizes

```c
size_t block_sizes[] = {64*1024, 128*1024, 256*1024, 512*1024, 1024*1024};

for (int i = 0; i < 5; i++) {
    size_t size = block_sizes[i];

    clock_t start = clock();
    gpu_raid_encode(handle, data, parity, num_drives, size);
    clock_t end = clock();

    double time_s = (end - start) / (double)CLOCKS_PER_SEC;
    double throughput = (num_drives * size) / time_s / (1024*1024*1024);

    printf("%zu KB: %.2f GB/s\n", size/1024, throughput);
}
```

### Recommendations

| Block Size | Use Case | Pros | Cons |
|------------|----------|------|------|
| 64 KB | Low latency, random I/O | Fast response | Lower throughput |
| 128 KB | Balanced | Good latency & throughput | - |
| 256 KB | Sequential I/O | High throughput | Moderate latency |
| 512 KB - 1 MB | Bulk transfers | Max throughput | High latency |
| > 1 MB | Very large files | Peak bandwidth | Memory overhead |

**General Rule:** Larger blocks = higher throughput, but more latency.

---

## Memory Optimization

### GPU Memory Pool

The memory pool pre-allocates GPU memory to avoid `cudaMalloc` overhead.

**Sizing:**
```c
// Conservative (frequent small operations)
.memory_pool_size_mb = 512

// Balanced
.memory_pool_size_mb = 1024

// Aggressive (large stripes, high throughput)
.memory_pool_size_mb = 4096
```

**Check pool usage:**
```c
gpu_raid_stats_t stats;
gpu_raid_get_stats(handle, &stats);

// If pool is exhausted, increase size
```

### Pinned Host Memory

For frequent host-device transfers, use pinned memory:

```c
// Allocate pinned memory
uint8_t* data;
cudaMallocHost(&data, BLOCK_SIZE);

// Fill data
memset(data, 0x42, BLOCK_SIZE);

// Use with GPU RAID (faster transfers)
gpu_raid_encode(...);

// Free when done
cudaFreeHost(data);
```

**Performance gain:** 2-3x faster transfers vs. pageable memory.

### Memory Bandwidth Test

```bash
# Test GPU memory bandwidth
cd build
./bench_throughput

# Look for:
# - Memory bandwidth utilization
# - Transfer times
```

---

## Async Operations

Use multiple CUDA streams for overlapped computation.

### Single Stream (Sequential)

```c
// Stream 0: Encode stripe 1
gpu_raid_encode(handle, data1, parity1, ...);

// Stream 0: Encode stripe 2 (waits for stripe 1)
gpu_raid_encode(handle, data2, parity2, ...);
```

**Total time:** T1 + T2

### Multi-Stream (Concurrent)

```c
// Configure multiple streams
config.num_streams = 4;

// Stream 0: Encode stripe 1
gpu_raid_encode_async(handle, data1, parity1, ..., 0);

// Stream 1: Encode stripe 2 (concurrent!)
gpu_raid_encode_async(handle, data2, parity2, ..., 1);

// Stream 2: Encode stripe 3
gpu_raid_encode_async(handle, data3, parity3, ..., 2);

// Wait for all
gpu_raid_sync(handle);
```

**Total time:** max(T1, T2, T3) ≈ T1 (if streams are balanced)

**Speedup:** Up to 3-4x with 4 streams.

### Stream Scheduling

```c
for (int i = 0; i < num_stripes; i++) {
    int stream_id = i % config.num_streams;
    gpu_raid_encode_async(handle, data[i], parity[i], ..., stream_id);
}

gpu_raid_sync(handle);  // Wait for all streams
```

---

## Multi-GPU

For extreme throughput, use multiple GPUs.

### Configuration

```c
// GPU 0
gpu_raid_config_t config0 = { .gpu_device_id = 0, ... };
gpu_raid_handle_t handle0;
gpu_raid_init(&config0, &handle0);

// GPU 1
gpu_raid_config_t config1 = { .gpu_device_id = 1, ... };
gpu_raid_handle_t handle1;
gpu_raid_init(&config1, &handle1);

// Distribute work
for (int i = 0; i < num_stripes; i++) {
    if (i % 2 == 0) {
        gpu_raid_encode(handle0, ...);  // GPU 0
    } else {
        gpu_raid_encode(handle1, ...);  // GPU 1
    }
}
```

**Expected:** ~2x throughput with 2 GPUs.

---

## Benchmarking

### Run Included Benchmarks

```bash
cd build

# Throughput benchmark
./bench_throughput

# Rebuild speed
./bench_rebuild_speed

# Stress test
./stress_test --duration 60
```

### Custom Benchmark

```c
#define ITERATIONS 1000

double total_time = 0;
size_t total_bytes = 0;

for (int i = 0; i < ITERATIONS; i++) {
    clock_t start = clock();
    gpu_raid_encode(handle, data, parity, num_drives, block_size);
    clock_t end = clock();

    total_time += (end - start) / (double)CLOCKS_PER_SEC;
    total_bytes += num_drives * block_size;
}

double avg_throughput = total_bytes / total_time / (1024*1024*1024);
printf("Average throughput: %.2f GB/s\n", avg_throughput);
```

### Metrics to Track

- **Encode throughput** (GB/s)
- **Decode throughput** (GB/s)
- **Rebuild speed** (GB/s)
- **Latency** (ms per operation)
- **GPU utilization** (%)
- **GPU temperature** (°C)
- **Power consumption** (W)

---

## Common Bottlenecks

### 1. PCIe Bandwidth

**Symptom:** Low throughput despite fast GPU.

**Diagnosis:**
```bash
nvidia-smi -q | grep -A 3 "PCIe"
```

**Fix:**
- Move GPU to PCIe 4.0 x16 slot
- Enable PCIe 4.0 in BIOS
- Check for PCIe bifurcation issues

### 2. Thermal Throttling

**Symptom:** Throughput degrades over time.

**Diagnosis:**
```bash
watch -n 1 nvidia-smi
# Monitor GPU temp and clocks
```

**Fix:**
- Improve case airflow
- Lower ambient temperature
- Reduce GPU power limit if needed:
  ```bash
  sudo nvidia-smi -pl 250  # 250W limit
  ```

### 3. Small Block Sizes

**Symptom:** Low throughput with frequent operations.

**Fix:** Increase block size to 256 KB or larger.

### 4. Single Stream

**Symptom:** GPU utilization < 50%.

**Fix:** Use multiple streams:
```c
config.num_streams = 4;
```

### 5. Memory Pool Exhaustion

**Symptom:** Errors or slow allocation.

**Fix:** Increase pool size:
```c
config.memory_pool_size_mb = 4096;
```

### 6. CPU Bottleneck

**Symptom:** GPU idle while CPU is busy.

**Fix:**
- Use pinned memory for faster transfers
- Batch operations
- Reduce host-device synchronization

---

## Performance Checklist

- [ ] GPU in PCIe 4.0 x16 slot
- [ ] NVMe drives on direct PCIe lanes
- [ ] Block size ≥ 256 KB for sequential I/O
- [ ] Memory pool sized appropriately
- [ ] Multiple CUDA streams enabled
- [ ] Pinned memory for frequent transfers
- [ ] GPU temperature < 80°C
- [ ] GPU utilization > 70%
- [ ] Benchmarked actual throughput

---

## Expected Performance

### RAID 5 Encoding

| GPU | 4 Drives | 6 Drives | 8 Drives |
|-----|----------|----------|----------|
| RTX 3080 | 18-22 GB/s | 20-25 GB/s | 22-28 GB/s |
| RTX 3060 | 10-14 GB/s | 12-16 GB/s | 14-18 GB/s |
| Quadro 4000 | 12-16 GB/s | 14-18 GB/s | 16-20 GB/s |

### RAID 6 Encoding

| GPU | 4 Drives | 6 Drives | 8 Drives |
|-----|----------|----------|----------|
| RTX 3080 | 12-16 GB/s | 14-18 GB/s | 16-20 GB/s |
| RTX 3060 | 6-10 GB/s | 8-12 GB/s | 10-14 GB/s |
| Quadro 4000 | 8-12 GB/s | 10-14 GB/s | 12-16 GB/s |

### Rebuild Speed

| GPU | Single Failure | Dual Failure (RAID 6) |
|-----|----------------|------------------------|
| RTX 3080 | 8-12 GB/s | 6-10 GB/s |
| RTX 3060 | 4-8 GB/s | 3-6 GB/s |
| Quadro 4000 | 6-10 GB/s | 4-8 GB/s |

**Note:** Actual performance varies by system configuration, drive speeds, and workload.

---

## Troubleshooting Performance

### GPU Not Utilized

```bash
# Monitor GPU usage
nvidia-smi dmon -s u -i 0 -d 1

# Should show > 70% utilization
```

**If low:**
- Increase block size
- Enable multi-stream
- Check for CPU bottlenecks

### Low Throughput

```bash
# Run benchmark
./bench_throughput

# Compare to expected performance table above
```

**If below expected:**
1. Check PCIe speed: `lspci -vv | grep LnkSta`
2. Check GPU clocks: `nvidia-smi -q | grep "Clocks"`
3. Check temperature: `nvidia-smi --query-gpu=temperature.gpu --format=csv`

---

## Further Optimization

For advanced users:

1. **Kernel Tuning:** Modify CUDA kernel launch parameters in source
2. **Custom Memory Allocators:** Replace memory pool with custom allocator
3. **NUMA Awareness:** Pin CPU threads to NUMA node closest to GPU
4. **GPU Direct Storage:** Use GDS for zero-copy I/O (requires special drivers)

See source code in `kernels/` for kernel implementation details.
