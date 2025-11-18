## Troubleshooting Guide

Common issues and solutions for GPU RAID.

### Table of Contents

- [Installation Issues](#installation-issues)
- [Runtime Errors](#runtime-errors)
- [Performance Problems](#performance-problems)
- [GPU Issues](#gpu-issues)
- [Build Errors](#build-errors)
- [Getting Help](#getting-help)

---

## Installation Issues

### "No CUDA-capable device detected"

**Cause:** CUDA driver not installed or GPU not recognized.

**Solution:**
```bash
# Check if nvidia driver is loaded
lsmod | grep nvidia

# If not, install drivers
sudo apt-get install nvidia-driver-535  # Or latest version

# Verify GPU is visible
nvidia-smi

# Should show your GPU
```

**Still not working?**
```bash
# Check if GPU is in lspci
lspci | grep -i nvidia

# If present but not in nvidia-smi, driver issue
# Reinstall driver:
sudo apt-get purge nvidia-*
sudo apt-get install nvidia-driver-535
sudo reboot
```

### "CUDA error: no kernel image available"

**Cause:** Library compiled for wrong GPU architecture.

**Solution:**
```bash
# Find your GPU's compute capability
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

# Example output: 8.6 (for RTX 3080/3060)
# Example output: 7.5 (for Quadro RTX 4000)

# Rebuild with correct architecture
cd build
rm -rf *
cmake .. -DCUDA_ARCH=86  # For compute 8.6
# OR
cmake .. -DCUDA_ARCH=75  # For compute 7.5

make -j$(nproc)
```

### "nvcc: command not found"

**Cause:** CUDA Toolkit not installed or not in PATH.

**Solution:**
```bash
# Install CUDA Toolkit
sudo apt-get install nvidia-cuda-toolkit

# Or download from NVIDIA:
# https://developer.nvidia.com/cuda-downloads

# Add to PATH if not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Add to ~/.bashrc for persistence
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

---

## Runtime Errors

### GPU_RAID_ERROR_OUT_OF_MEMORY

**Cause:** Insufficient GPU memory for operation.

**Solution 1: Reduce memory pool**
```c
config.memory_pool_size_mb = 512;  // Instead of 2048
```

**Solution 2: Use smaller block sizes**
```c
// Encode smaller blocks at a time
for (int i = 0; i < large_file_size; i += BLOCK_SIZE) {
    gpu_raid_encode(handle, data + i, parity + i, ...);
}
```

**Solution 3: Check free memory**
```bash
nvidia-smi --query-gpu=memory.free --format=csv

# If low, close other GPU applications
# Or use smaller configuration
```

### GPU_RAID_ERROR_KERNEL_LAUNCH_FAILED

**Cause:** CUDA kernel failed to launch or execute.

**Check CUDA error:**
```c
cudaError_t err = gpu_raid_encode(...);
if (err != GPU_RAID_SUCCESS) {
    // Get detailed CUDA error
    cudaError_t cuda_err = cudaGetLastError();
    printf("CUDA error: %s\n", cudaGetErrorString(cuda_err));
}
```

**Common causes:**
- **Invalid parameters:** Check block_size is multiple of 16
- **Timeout:** Increase GPU timeout (see below)
- **GPU hang:** Restart driver with `sudo nvidia-smi --gpu-reset`

### GPU_RAID_ERROR_TOO_MANY_FAILURES

**Cause:** Attempting to recover more drives than RAID level supports.

**RAID 5:** Max 1 failure
**RAID 6:** Max 2 failures

**Solution:**
```c
// Verify failure count
if (num_failed > 1 && raid_level == GPU_RAID_LEVEL_5) {
    fprintf(stderr, "RAID 5 cannot recover from %d failures\n", num_failed);
    // Need RAID 6 for dual failure
}
```

### "GPU has fallen off the bus"

**Cause:** GPU driver crashed or hardware issue.

**Solution:**
```bash
# Reset GPU
sudo nvidia-smi --gpu-reset

# If persists, check:
# 1. GPU power cables properly connected
# 2. PSU sufficient wattage
# 3. GPU not overheating
# 4. PCIe slot properly seated

# Check dmesg for errors
sudo dmesg | grep -i nvidia
```

---

## Performance Problems

### Low Throughput (<50% Expected)

**Diagnosis:**
```bash
# Run benchmark
cd build
./bench_throughput

# Check GPU utilization
nvidia-smi dmon -s u -i 0 -d 1

# Check PCIe speed
lspci -vv | grep -A 10 "VGA" | grep LnkSta
```

**Common fixes:**

**1. PCIe Running at x8 or x4**
```bash
# Expected: Speed 16GT/s, Width x16
# If x8 or x4, move GPU to different slot
```

**2. PCIe 3.0 Instead of 4.0**
```bash
# Enable PCIe 4.0 in BIOS
# Some motherboards default to 3.0 for compatibility
```

**3. Thermal Throttling**
```bash
# Monitor temperature
nvidia-smi --query-gpu=temperature.gpu,clocks.gr --format=csv

# If temp > 80°C and clocks dropping:
# - Improve cooling
# - Lower power limit:
sudo nvidia-smi -pl 250  # 250W limit
```

**4. Block Size Too Small**
```c
// Increase block size
config.stripe_size_kb = 256;  // Or larger
```

### GPU Utilization Low (<30%)

**Cause:** Not enough parallelism to saturate GPU.

**Solution 1: Multi-stream**
```c
config.num_streams = 4;

// Use async operations
for (int i = 0; i < num_ops; i++) {
    int stream = i % 4;
    gpu_raid_encode_async(handle, ..., stream);
}
gpu_raid_sync(handle);
```

**Solution 2: Larger operations**
```c
// Process more data per call
block_size = 1024 * 1024;  // 1 MB blocks
```

### Inconsistent Performance

**Symptom:** Throughput varies significantly between runs.

**Causes:**
- **Thermal throttling:** Check temps
- **CPU frequency scaling:** Disable power saving
- **Background processes:** Close other GPU applications
- **Memory fragmentation:** Restart application periodically

**Fix CPU scaling:**
```bash
# Set performance governor
sudo cpupower frequency-set -g performance

# Or disable turbo boost for consistency
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

---

## GPU Issues

### GPU Not Detected

**Check GPU presence:**
```bash
lspci | grep -i nvidia

# Should show something like:
# 01:00.0 VGA compatible controller: NVIDIA Corporation GA102 [GeForce RTX 3080]
```

**If not present:**
- Reseat GPU in PCIe slot
- Check power cables connected
- Try different PCIe slot

**If present but not in nvidia-smi:**
- Driver issue (see Installation Issues)

### Wrong GPU Selected

**Symptom:** Using integrated GPU instead of discrete.

**Fix:**
```c
// Explicitly select GPU 0 (usually discrete)
config.gpu_device_id = 0;

// Query available devices
int device_count;
cudaGetDeviceCount(&device_count);
for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("GPU %d: %s\n", i, prop.name);
}
```

### GPU Timeout Errors

**Symptom:** "GPU timeout" or watchdog timer expired.

**Cause:** Long-running kernels trigger watchdog.

**Solution 1: Increase timeout (Linux)**
```bash
# Edit Xorg config
sudo vim /etc/X11/xorg.conf

# Add to Device section:
Option "Interactive" "0"

# Restart X server
sudo systemctl restart gdm
```

**Solution 2: Use headless GPU**
- Don't run display on RAID GPU
- Use integrated graphics for display
- Dedicate GPU to compute

**Solution 3: Reduce operation size**
```c
// Process in smaller chunks
for (int i = 0; i < total; i += chunk_size) {
    gpu_raid_encode(handle, data + i, ...);
}
```

---

## Build Errors

### "undefined reference to cudaMalloc"

**Cause:** Not linking against CUDA library.

**Fix:**
```cmake
# In CMakeLists.txt, ensure:
target_link_libraries(your_target ${CUDA_LIBRARIES})
```

### "cannot find -lcusparseLt"

**Cause:** cuSPARSELt not in library path.

**Fix:**
```bash
# Find cuSPARSELt
find /usr -name "libcusparseLt*" 2>/dev/null

# Add to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### CMake: "CUDA_ARCH not defined"

**Fix:**
```bash
# Specify architecture
cmake .. -DCUDA_ARCH=86
```

### Compiler Warnings About Deprecated API

**Cause:** Using newer CUDA version with older API.

**Solution:** Update source or suppress warnings:
```cmake
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-declarations")
```

---

## Data Corruption Issues

### Parity Verification Fails

**Symptom:** `gpu_raid_verify()` returns `is_valid = false`

**Diagnosis:**
```c
// Verify step-by-step
gpu_raid_encode(handle, data, parity, num, size);

// Immediately verify
bool valid;
gpu_raid_verify(handle, data, parity, num, size, &valid);

if (!valid) {
    // Check if data was modified
    // Check if parity buffer is large enough
    // Check if block_size matches
}
```

**Common causes:**
- Data modified between encode and verify
- Incorrect buffer sizes
- Using wrong RAID level for verification

### Recovered Data Doesn't Match

**Symptom:** Reconstructed block differs from original.

**Debug:**
```c
// Save original
uint8_t* original = malloc(BLOCK_SIZE);
memcpy(original, data[failed_idx], BLOCK_SIZE);

// Simulate failure and recover
data[failed_idx] = NULL;
gpu_raid_reconstruct(...);

// Compare
if (memcmp(original, recovered, BLOCK_SIZE) != 0) {
    // Find first mismatch
    for (size_t i = 0; i < BLOCK_SIZE; i++) {
        if (original[i] != recovered[i]) {
            printf("Mismatch at byte %zu: %02X != %02X\n",
                   i, original[i], recovered[i]);
            break;
        }
    }
}
```

**Possible issues:**
- Failed index incorrect
- Wrong number of failures specified
- Parity not current
- Using RAID 5 with 2 failures (need RAID 6)

---

## Logging and Debugging

### Enable Debug Output

```c
// Set environment variable before running
export CUDA_LAUNCH_BLOCKING=1

// This makes kernel launches synchronous
// Easier to pinpoint errors
```

### CUDA Memory Checker

```bash
# Run with cuda-memcheck
cuda-memcheck ./your_program

# Detects:
# - Out of bounds memory access
# - Uninitialized memory
# - Race conditions
```

### Profiling

```bash
# Profile with nvprof (CUDA 10/11)
nvprof ./your_program

# Or use Nsight Systems (CUDA 11+)
nsys profile --stats=true ./your_program
```

---

## Known Limitations

### Maximum Drive Count

- **Max 16 data drives** (MAX_DATA_DRIVES in raid_types.h)
- To increase, recompile with larger MAX_DATA_DRIVES

### Block Size

- **Min: 4 KB** (for alignment)
- **Max: 16 MB** (for performance)
- **Recommended: 256 KB - 1 MB**

### Multi-Threading

- Single handle not thread-safe
- Create separate handles per thread

### Tenstorrent Integration

- **Experimental only**
- Based on deprecated SDK
- Auto-falls back to GPU-only
- **Do not use in production**

---

## Getting Help

### Before Asking

1. **Check this guide** for your specific error
2. **Run diagnostics:**
   ```bash
   nvidia-smi
   lspci | grep -i nvidia
   nvcc --version
   ./bench_throughput
   ```
3. **Check GPU temperature** (overheat protection at ~83°C)
4. **Verify block sizes** are multiples of 16 bytes

### Information to Provide

When reporting issues, include:

```bash
# System info
uname -a
cat /etc/os-release

# GPU info
nvidia-smi --query-gpu=gpu_name,compute_cap,driver_version --format=csv

# CUDA info
nvcc --version

# Build info
cmake --version
cat CMakeCache.txt | grep CUDA_ARCH

# Error message (exact text)
# Steps to reproduce
# Code snippet (if possible)
```

### Where to Get Help

- **GitHub Issues:** [Link to issues]
- **Documentation:** Read API.md, PERFORMANCE.md
- **Examples:** Check examples/ directory
- **Source Code:** Read kernels/ for implementation details

---

## FAQ

**Q: Can I use this with SATA SSDs?**
A: Yes, but you won't see GPU acceleration benefits. Use NVMe for best performance.

**Q: Do I need ECC memory?**
A: No. Quadro GPUs have better error handling, but consumer GPUs work fine for most use cases.

**Q: Can I mix RAID levels?**
A: No. Each handle is for one RAID level. Create multiple handles if needed.

**Q: Does this work on Windows?**
A: Library is Linux-focused. Windows may require modifications.

**Q: Can I use AMD GPUs?**
A: No. This implementation requires NVIDIA CUDA.

**Q: What about RAID 0/1/10?**
A: Currently only RAID 5/6. RAID 0 is trivial (no parity). RAID 1/10 are copying, not compute-intensive.

**Q: How do I update the library?**
A: Pull latest code, rebuild:
```bash
git pull origin main
cd build && make clean && make -j$(nproc)
```

---

## Advanced Debugging

### Enable CUDA Error Checking

Edit source to add error checks:

```c
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while (0)

// Use before every CUDA call
CUDA_CHECK(cudaMalloc(&ptr, size));
```

### GPU Stack Trace

```bash
# Compile with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Run with cuda-gdb
cuda-gdb ./your_program
```

### Memory Leak Detection

```bash
# Use cuda-memcheck
cuda-memcheck --leak-check full ./your_program
```

---

**Still stuck?** Open an issue with full diagnostic output.
