# Quick Start Guide

## Tenstorrent Grayskull GRAID Implementation

This guide will help you get started with the Tenstorrent Grayskull implementation of GRAID (GPU-Accelerated RAID).

## Prerequisites

### Hardware Requirements

- Tenstorrent Grayskull e75 or e150 PCIe card
- PCIe Gen 4.0 x16 slot
- Sufficient PSU capacity:
  - e75: 75W TDP
  - e150: 200W TDP
- Linux x86_64 system (Ubuntu 20.04+ or equivalent)

### Software Requirements

**⚠️ IMPORTANT**: Grayskull software support has been discontinued by Tenstorrent.

Last supported versions:
- TT-Metalium v0.55
- TT-Buda v0.19.3

```bash
# System dependencies
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    libhwloc-dev \
    libyaml-cpp-dev
```

## Installation

### 1. Install TT-Metalium v0.55

```bash
# Clone TT-Metal repository
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout v0.55.0  # Last supported version for Grayskull

# Set environment variables
export ARCH_NAME=grayskull
export TT_METAL_HOME=$(pwd)

# Build SDK
cmake -B build -G Ninja
cmake --build build

# Install Python bindings
pip3 install -e .
```

### 2. Verify Hardware Detection

```bash
# Check if device is detected
tt-smi

# Expected output:
# Device 0: Grayskull e75/e150
#   Cores: 96/120
#   Memory: 8 GB LPDDR4
#   Status: Ready
```

### 3. Build GRAID Implementation

```bash
cd /path/to/GRAID_GPT/tenstorrent_greyskull

# Create build directory
mkdir build && cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Optionally install
sudo make install
```

## Running Examples

### Example 1: Device Enumeration

Lists all detected Grayskull devices and their properties.

```bash
./01_device_enumeration
```

Expected output:
```
=================================================================================
Tenstorrent Grayskull Device Enumeration
GRAID_GPT - GPU-Accelerated RAID
=================================================================================

Found 1 Grayskull device(s)

=================================================================================
Device 0: Grayskull e75
=================================================================================
Hardware Type:        Grayskull e75

--- Compute Configuration ---
Tensix Cores:         96
Core Clock:           1000 MHz (1.0 GHz)
Core Grid Size:       10 x 10

--- Memory Configuration ---
L1 SRAM per Core:     1 MB
Total L1 SRAM:        96 MB
Device DRAM:          8 GB LPDDR4
Memory Bandwidth:     102 GB/s

--- Compute Capabilities ---
Peak FP8 Performance: 221 TeraFLOPS
Peak FP16 Performance:55 TeraFLOPS

--- Power and Interface ---
TDP:                  75 W
PCIe Interface:       Gen 4 x16
```

### Example 2: Memory Operations

Demonstrates memory allocation and transfers.

```bash
# Test with 64 MB
./02_memory_operations 64

# Test with 256 MB
./02_memory_operations 256
```

### Example 3: Reed-Solomon RAID

Demonstrates RAID encoding and recovery.

```bash
# RAID-5: 4 data blocks + 1 parity, 1 MB per block
./03_reed_solomon_raid 4 1 1024

# RAID-6: 6 data blocks + 2 parity, 512 KB per block
./03_reed_solomon_raid 6 2 512

# Large test: 8 data blocks + 2 parity, 4 MB per block
./03_reed_solomon_raid 8 2 4096
```

## Performance Tuning

### Memory Optimization

```cpp
// Use interleaved DRAM for better bandwidth
auto buffer = memory.AllocateDRAM(size, true);  // interleaved = true

// Shard across multiple cores for parallel processing
auto shards = memory.AllocateL1Sharded(total_size, cores);
```

### Kernel Optimization

```cpp
// Configure more cores per block for better parallelism
ReedSolomonConfig config;
config.k = 8;
config.m = 2;
config.cores_per_block = 4;  // Use 4 cores per data block
```

### NoC Routing

```cpp
// Get optimal core layout based on NoC topology
auto worker_cores = device.GetWorkerCores();

// Minimize NoC latency by grouping nearby cores
std::vector<CoreCoord> core_group = {
    worker_cores[0],   // (0, 0)
    worker_cores[1],   // (0, 1)
    worker_cores[10],  // (1, 0)
    worker_cores[11]   // (1, 1)
};
```

## Benchmarking

### RAID Encoding Throughput

Expected performance (theoretical):

| Configuration | e75 Throughput | e150 Throughput |
|--------------|---------------|-----------------|
| RAID-5 (4+1) | 15-20 GB/s | 20-25 GB/s |
| RAID-6 (6+2) | 12-18 GB/s | 18-22 GB/s |

Actual performance depends on:
- Block size (larger is better)
- PCIe bandwidth utilization
- NoC routing efficiency
- Kernel optimization

### Running Benchmarks

```bash
# Quick benchmark
./03_reed_solomon_raid 4 2 1024

# Comprehensive benchmark
for k in 4 6 8 12; do
    for m in 1 2; do
        for bs in 512 1024 2048 4096; do
            echo "Testing k=$k m=$m block_size=${bs}KB"
            ./03_reed_solomon_raid $k $m $bs
        done
    done
done
```

## Troubleshooting

### Device Not Detected

```bash
# Check PCIe detection
lspci | grep -i tenstorrent

# Check kernel messages
dmesg | grep -i tenstorrent

# Verify driver loaded
lsmod | grep tenstorrent
```

### Build Errors

```bash
# Ensure TT_METAL_HOME is set
echo $TT_METAL_HOME

# Verify TT-Metal installation
ls $TT_METAL_HOME/build/lib/libtt_metal.so

# Clean rebuild
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
make VERBOSE=1
```

### Runtime Errors

```bash
# Enable verbose logging
export TT_METAL_LOGGER_LEVEL=Debug

# Run with error checking
./01_device_enumeration 2>&1 | tee debug.log

# Check device status
tt-smi --verbose
```

## Next Steps

1. Read the [full README](../README.md) for architecture details
2. Explore the [API Reference](API_REFERENCE.md)
3. Review the [Performance Guide](PERFORMANCE_GUIDE.md)
4. Check example source code in `examples/`

## Support

- Hardware: Grayskull support discontinued - refer to archived docs
- Software: Use TT-Metalium v0.55 (last supported version)
- Community: [Tenstorrent Discord](https://discord.gg/tenstorrent)
- Issues: GitHub issues for this repository

## Additional Resources

- [Tenstorrent Documentation](https://docs.tenstorrent.com/)
- [TT-Metal GitHub](https://github.com/tenstorrent/tt-metal)
- [Grayskull Specifications](https://docs.tenstorrent.com/aibs/grayskull/specifications.html)
- [Reed-Solomon Coding Overview](../README.md#reed-solomon-coding-for-raid)
