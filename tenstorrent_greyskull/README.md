# Tenstorrent Greyskull Hardware Support

> GRAID_GPT Implementation for Tenstorrent Grayskull e75 and e150 RISC-V AI Accelerators

This directory contains hardware-specific implementations for Tenstorrent's Grayskull e75 and e150 PCIe AI accelerator cards, enabling GPU-accelerated RAID operations using RISC-V based tensor processors.

## Table of Contents

- [Hardware Overview](#hardware-overview)
- [Architecture](#architecture)
- [Software Stack](#software-stack)
- [System Integration](#system-integration)
- [Memory Architecture](#memory-architecture)
- [RAID Implementation](#raid-implementation)
- [Getting Started](#getting-started)
- [Examples](#examples)
- [Important Notes](#important-notes)

## Hardware Overview

### Grayskull e75

| Specification | Value |
|--------------|-------|
| Tensix Cores | 96 @ 1.0 GHz |
| On-Chip SRAM | 96 MB (1 MB per core) |
| External Memory | 8 GB LPDDR4 |
| Memory Bandwidth | 102.4 GB/s |
| Peak Performance (FP8) | 221 TeraFLOPS |
| Peak Performance (FP16) | 55 TeraFLOPS |
| Power Consumption | 75W TDP |
| Form Factor | Single-slot, low-profile |
| Interface | PCIe Gen 4.0 x16 |
| Price | $599 USD |

### Grayskull e150

| Specification | Value |
|--------------|-------|
| Tensix Cores | 120 @ 1.2 GHz |
| On-Chip SRAM | 120 MB (1 MB per core) |
| External Memory | 8 GB LPDDR4 |
| Memory Bandwidth | 118.4 GB/s |
| Peak Performance (FP8) | 332 TeraFLOPS |
| Peak Performance (FP16) | 83 TeraFLOPS |
| Power Consumption | 200W TDP |
| Form Factor | Dual-slot, full-height |
| Interface | PCIe Gen 4.0 x16 |
| Price | $799 USD |

### Supported Data Types

- **Floating Point**: FP8, FP16, BFLOAT16
- **Block Floating Point**: BLOCKFP2, BLOCKFP4, BLOCKFP8
- **Vector**: VTF19

## Architecture

### Tensix Core Architecture

```mermaid
graph TB
    subgraph "Tensix Core (1 of 96/120)"
        A[Tensix Core Controller] --> B[5x RISC-V Processors]
        A --> C[Tensor Array Math Unit]
        A --> D[SIMD Unit]
        A --> E[Network Operations Unit]
        A --> F[Compression/Decompression Unit]

        B --> G[1 MB L1 SRAM]
        C --> G
        D --> G

        G --> H[NoC Router]
    end

    H --> I[Network-on-Chip]
    I --> J[LPDDR4 Memory Controller]
    J --> K[8 GB LPDDR4]

    style A fill:#4A90E2
    style B fill:#50C878
    style C fill:#FFD700
    style G fill:#FF6B6B
```

### Grayskull System Architecture

```mermaid
graph LR
    subgraph "Host System"
        A[CPU] --> B[PCIe Gen 4 Root Complex]
    end

    subgraph "Grayskull Card"
        B --> C[PCIe Controller]
        C --> D[Network-on-Chip]

        D --> E[Tensix Core Grid]
        D --> F[ARC RISC-V Control Processor]
        D --> G[DMA Engines]
        D --> H[Memory Controller]

        E --> I1[Row 0: 12 Cores]
        E --> I2[Row 1: 12 Cores]
        E --> I3[Row 7: 12 Cores]

        H --> J[8 GB LPDDR4]

        I1 --> K[96 MB Total SRAM]
        I2 --> K
        I3 --> K
    end

    style C fill:#4A90E2
    style E fill:#50C878
    style J fill:#FFD700
    style K fill:#FF6B6B
```

### Tensix Core Grid Layout

```mermaid
graph TD
    subgraph "Grayskull e75: 96 Cores (10x10 grid with 4 reserved)"
        A[Core 0,0] --- B[Core 0,1] --- C[Core 0,2] --- D[Core 0,9]
        E[Core 1,0] --- F[Core 1,1] --- G[Core 1,2] --- H[Core 1,9]
        I[Core 9,0] --- J[Core 9,1] --- K[Core 9,2] --- L[Core 9,9]
    end

    subgraph "Network-on-Chip Topology"
        M[2D Mesh NoC]
        N[Row-Column Routing]
        O[Multicast Support]
        P[Packet-Based Communication]
    end

    A --> M
    M --> N
    N --> O
    O --> P

    style M fill:#4A90E2
    style A fill:#50C878
```

## Software Stack

### TT-Metalium Software Stack (Deprecated)

```mermaid
graph TB
    subgraph "Application Layer"
        A[User Application]
        B[PyTorch Models]
        C[TensorFlow Models]
    end

    subgraph "Framework Layer"
        D[TT-Forge Compiler]
        E[TT-NN Neural Network Library]
    end

    subgraph "Runtime Layer"
        F[TT-Metalium SDK]
        G[Device Runtime]
        H[Host Runtime]
    end

    subgraph "Kernel Layer"
        I[Compute Kernels C++]
        J[Data Movement Kernels]
        K[Custom Operations]
    end

    subgraph "Hardware Layer"
        L[Tensix Cores]
        M[NoC]
        N[RISC-V Processors]
        O[Memory Hierarchy]
    end

    A --> D
    B --> D
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    G --> J
    H --> K
    I --> L
    J --> M
    K --> N
    L --> O

    style D fill:#4A90E2
    style F fill:#50C878
    style L fill:#FFD700
```

### Software Components

**Last Supported Versions (Deprecated):**
- TT-Metalium: v0.55
- TT-Buda: v0.19.3

```mermaid
flowchart LR
    A[TT-Forge] -->|MLIR Compilation| B[TT-NN]
    B -->|High-Level Ops| C[TT-Metalium]
    C -->|Low-Level API| D[Hardware]

    E[PyTorch/JAX/TF] --> A
    F[Custom C++ Kernels] --> C

    style A fill:#90EE90
    style B fill:#87CEEB
    style C fill:#FFD700
    style D fill:#FF6B6B
```

## System Integration

### PCIe Communication Flow

```mermaid
sequenceDiagram
    participant Host as Host CPU
    participant PCIe as PCIe Controller
    participant NoC as Network-on-Chip
    participant Tensix as Tensix Cores
    participant Mem as LPDDR4 Memory

    Host->>PCIe: Initialize Device
    PCIe->>NoC: Configure Routing
    NoC->>Tensix: Enumerate Cores

    Host->>PCIe: Upload Kernel Binary
    PCIe->>Mem: DMA Transfer
    NoC->>Tensix: Distribute Kernel

    Host->>PCIe: Upload Input Data
    PCIe->>Mem: Write Data
    NoC->>Tensix: Signal Data Ready

    Tensix->>Tensix: Execute Kernel
    Tensix->>Mem: Write Results

    Tensix-->>NoC: Completion Signal
    NoC-->>PCIe: Interrupt
    PCIe-->>Host: Notify Complete

    Host->>PCIe: Read Results
    PCIe->>Mem: DMA Read
    Mem-->>Host: Return Data
```

## Memory Architecture

### Memory Hierarchy

```mermaid
graph TB
    subgraph "Host System"
        A[Host DDR4/DDR5]
    end

    subgraph "PCIe Interconnect"
        B[PCIe Gen 4 x16<br/>32 GB/s Bidirectional]
    end

    subgraph "Grayskull Card"
        C[8 GB LPDDR4<br/>102-118 GB/s]

        subgraph "Tensix Core"
            D[L1 SRAM: 1 MB<br/>Per-Core]
            E[Register File<br/>RISC-V Registers]
        end
    end

    A <-->|PCIe| B
    B <-->|DMA| C
    C <-->|NoC| D
    D <-->|Load/Store| E

    style A fill:#90EE90
    style C fill:#87CEEB
    style D fill:#FFD700
    style E fill:#FF6B6B
```

### Memory Access Patterns

```mermaid
flowchart TD
    A[Data Input] --> B{Size Check}
    B -->|Fits in L1| C[Direct L1 Processing]
    B -->|Too Large| D[Tiling Strategy]

    D --> E[Split Data into Tiles]
    E --> F[Stream Tile to L1]
    F --> G[Process Tile]
    G --> H[Stream Results Out]
    H --> I{More Tiles?}
    I -->|Yes| F
    I -->|No| J[Combine Results]

    C --> K[Output]
    J --> K

    style C fill:#90EE90
    style D fill:#FFD700
    style G fill:#87CEEB
```

## RAID Implementation

### Reed-Solomon Coding on Tensix Cores

```mermaid
flowchart TB
    subgraph "Data Preparation"
        A[Input Data Stream] --> B[Split into k Buffers]
        B --> C[Buffer 0]
        B --> D[Buffer 1]
        B --> E[Buffer k-1]
    end

    subgraph "Tensix Core Distribution"
        C --> F[Tensix Core Group 0]
        D --> G[Tensix Core Group 1]
        E --> H[Tensix Core Group k-1]
    end

    subgraph "Reed-Solomon Encoding"
        F --> I[GF Multiplication]
        G --> I
        H --> I

        I --> J[Generate Parity 0]
        I --> K[Generate Parity 1]
        I --> L[Generate Parity m-1]
    end

    subgraph "Output"
        J --> M[Coding Buffer k]
        K --> N[Coding Buffer k+1]
        L --> O[Coding Buffer k+m-1]
    end

    style I fill:#FFD700
    style F fill:#87CEEB
    style M fill:#90EE90
```

### Galois Field Operations Mapping

```mermaid
graph LR
    subgraph "Traditional GPU"
        A[Lookup Tables in Constant Memory]
        B[Thread per Byte]
        C[Shared Memory Banks]
    end

    subgraph "Tensix Adaptation"
        D[Tables in L1 SRAM]
        E[RISC-V Thread per Block]
        F[SIMD Vector Operations]
        G[Tensor Math Unit for Matrix Ops]
    end

    A -.Adapt.-> D
    B -.Adapt.-> E
    C -.Adapt.-> F

    D --> H[Performance Optimization]
    E --> H
    F --> H
    G --> H

    style A fill:#FF6B6B
    style D fill:#90EE90
    style H fill:#FFD700
```

### RAID-6 Dual Parity Computation

```mermaid
sequenceDiagram
    participant Host as Host System
    participant DMA as DMA Engine
    participant NoC as Network-on-Chip
    participant T0 as Tensix Group 0
    participant T1 as Tensix Group 1
    participant Mem as LPDDR4

    Host->>DMA: Upload Data Buffers
    DMA->>Mem: Store in LPDDR4

    NoC->>T0: Distribute Buffer 0
    NoC->>T1: Distribute Buffer 1

    par P Parity Calculation
        T0->>T0: GF(2^8) Multiply
        T1->>T1: GF(2^8) Multiply
    and Q Parity Calculation
        T0->>T0: GF(2^8) Power Multiply
        T1->>T1: GF(2^8) Power Multiply
    end

    T0->>NoC: Send P0, Q0
    T1->>NoC: Send P1, Q1

    NoC->>Mem: XOR Reduction (P)
    NoC->>Mem: XOR Reduction (Q)

    Mem->>DMA: Parity Results
    DMA->>Host: Transfer Complete
```

## Getting Started

### Prerequisites

```mermaid
flowchart TD
    A[System Requirements] --> B{Hardware}
    A --> C{Software}

    B --> D[PCIe Gen 4.0 Slot]
    B --> E[Sufficient PSU<br/>e75: 75W<br/>e150: 200W]
    B --> F[Linux x86_64 System]

    C --> G[TT-Metalium v0.55]
    C --> H[TT-Buda v0.19.3]
    C --> I[Python 3.8+]
    C --> J[CMake 3.16+]
    C --> K[GCC 9.0+]

    G --> L[Installation]
    H --> L
    I --> L
    J --> L
    K --> L

    style A fill:#4A90E2
    style L fill:#50C878
```

### Installation Steps

**⚠️ IMPORTANT NOTE**: Software support for Grayskull has been discontinued. The following instructions reference the last supported versions.

#### 1. Install System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    libhwloc-dev \
    libyaml-cpp-dev

# Fedora/RHEL
sudo dnf install -y \
    gcc gcc-c++ \
    cmake \
    git \
    python3-pip \
    python3-devel \
    hwloc-devel \
    yaml-cpp-devel
```

#### 2. Install TT-Metalium v0.55 (Last Supported Version)

```bash
# Clone the repository (specific version)
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout v0.55.0  # Last supported Grayskull version

# Set environment variables
export ARCH_NAME=grayskull
export TT_METAL_HOME=$(pwd)

# Build the SDK
cmake -B build -G Ninja
cmake --build build

# Install Python bindings
pip3 install -e .
```

#### 3. Verify Hardware Detection

```bash
# List detected Grayskull devices
tt-smi

# Expected output:
# Device 0: Grayskull e75/e150
#   Cores: 96/120
#   Memory: 8 GB LPDDR4
#   Status: Ready
```

### Device Initialization Example

```cpp
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

int main() {
    // Enumerate devices
    int num_devices = tt::tt_metal::GetNumAvailableDevices();
    std::cout << "Found " << num_devices << " Grayskull device(s)" << std::endl;

    // Initialize first device
    Device* device = tt::tt_metal::CreateDevice(0);

    // Get device properties
    const auto& soc_desc = device->get_soc_descriptor();
    std::cout << "Device: " << soc_desc.name << std::endl;
    std::cout << "Tensix Cores: " << soc_desc.worker_grid_size.x *
                                       soc_desc.worker_grid_size.y << std::endl;
    std::cout << "DRAM Channels: " << soc_desc.dram_cores.size() << std::endl;

    // Close device
    tt::tt_metal::CloseDevice(device);

    return 0;
}
```

## Examples

### Directory Structure

```
tenstorrent_greyskull/
├── README.md                           # This file
├── docs/
│   ├── API_REFERENCE.md               # API documentation
│   ├── PERFORMANCE_GUIDE.md           # Optimization guide
│   └── MIGRATION_GUIDE.md             # GPU to Tensix migration
├── include/
│   ├── tt_graid_device.hpp            # Device management
│   ├── tt_graid_memory.hpp            # Memory operations
│   └── tt_graid_reed_solomon.hpp      # Reed-Solomon ops
├── src/
│   ├── tt_graid_device.cpp            # Device implementation
│   ├── tt_graid_memory.cpp            # Memory management
│   └── tt_graid_reed_solomon.cpp      # RS implementation
└── examples/
    ├── 01_device_enumeration.cpp      # List devices
    ├── 02_memory_transfer.cpp         # DMA examples
    ├── 03_simple_compute.cpp          # Basic kernel
    ├── 04_reed_solomon_encode.cpp     # RS encoding
    └── 05_raid_benchmark.cpp          # Performance test
```

### Quick Start Example

See `examples/01_device_enumeration.cpp` for a complete device initialization example.

### Build Examples

```bash
cd tenstorrent_greyskull
mkdir build && cd build
cmake ..
make

# Run examples
./examples/01_device_enumeration
./examples/02_memory_transfer
./examples/03_simple_compute
```

## Important Notes

### Software Support Status

⚠️ **CRITICAL**: Tenstorrent has **discontinued software support for Grayskull** hardware.

- **Last supported TT-Metalium version**: v0.55
- **Last supported TT-Buda version**: v0.19.3
- **Current development focus**: Wormhole and newer architectures

This implementation is provided for:
1. **Educational purposes**: Understanding RISC-V based AI accelerators
2. **Legacy system support**: Maintaining existing Grayskull deployments
3. **Research**: Exploring alternative accelerator architectures
4. **Archived documentation**: Preserving knowledge of this architecture

### Recommendations

- **For new projects**: Consider Tenstorrent Wormhole or other actively supported hardware
- **For existing deployments**: Pin to TT-Metalium v0.55 and TT-Buda v0.19.3
- **For development**: Use archived documentation and community resources

### Alternative Approaches

If you need current support and similar capabilities:

1. **Tenstorrent Wormhole**: Next-generation Tenstorrent hardware with active support
2. **NVIDIA GPUs**: Continue with cuSPARSELt implementation (see `/cuSPARSELt` directory)
3. **AMD MI-series**: ROCm-based alternatives
4. **Intel Habana**: Gaudi-series AI accelerators

## Architecture Comparison

```mermaid
graph TB
    subgraph "NVIDIA GPU (cuSPARSELt)"
        A1[Thousands of CUDA Cores]
        A2[Tensor Cores]
        A3[Shared Memory]
        A4[Global Memory]
    end

    subgraph "Tenstorrent Grayskull"
        B1[96-120 Tensix Cores]
        B2[5 RISC-V per Tensix]
        B3[1 MB L1 SRAM per Core]
        B4[8 GB LPDDR4]
        B5[Network-on-Chip]
    end

    A1 -.Similar to.-> B2
    A2 -.Similar to.-> B1
    A3 -.Similar to.-> B3
    A4 -.Similar to.-> B4

    style A2 fill:#90EE90
    style B1 fill:#87CEEB
```

## Performance Expectations

### Theoretical Performance

| Operation | e75 | e150 | NVIDIA A100 |
|-----------|-----|------|-------------|
| FP8 TFLOPS | 221 | 332 | 624 |
| FP16 TFLOPS | 55 | 83 | 312 |
| Memory BW | 102 GB/s | 118 GB/s | 1555 GB/s |
| Power | 75W | 200W | 400W |
| TFLOPS/Watt (FP8) | 2.95 | 1.66 | 1.56 |

### RAID Operation Estimates

Based on architecture analysis and theoretical limits:

```mermaid
graph LR
    A[RAID-5 Parity Gen] --> B[Grayskull e75: ~15-20 GB/s]
    A --> C[Grayskull e150: ~20-25 GB/s]
    A --> D[NVIDIA RTX 3090: ~40-50 GB/s]

    style B fill:#87CEEB
    style C fill:#90EE90
    style D fill:#FFD700
```

**Bottlenecks:**
- Limited LPDDR4 bandwidth vs GDDR6/HBM
- PCIe Gen 4 transfer overhead
- NoC routing latency for small operations
- RISC-V kernel compilation complexity

**Advantages:**
- Excellent power efficiency
- Deterministic performance
- Fine-grained control over compute
- Open-source software stack

## Contributing

This is a reference implementation for educational and research purposes. Contributions welcome for:

- Optimization improvements
- Documentation enhancements
- Alternative algorithm implementations
- Performance benchmarks
- Bug fixes

## License

See main repository [LICENSE](../LICENSE) file.

## References

- [Tenstorrent Official Documentation](https://docs.tenstorrent.com/)
- [TT-Metal GitHub Repository](https://github.com/tenstorrent/tt-metal)
- [Grayskull Hardware Specifications](https://docs.tenstorrent.com/aibs/grayskull/specifications.html)
- [RISC-V Specification](https://riscv.org/specifications/)
- [Reed-Solomon Error Correction](../README.md#reed-solomon-coding-for-raid)

## Support

- **Hardware**: Grayskull support discontinued - refer to archived documentation
- **Community**: [Tenstorrent Discord](https://discord.gg/tenstorrent)
- **Issues**: GitHub issues for this repository
- **Commercial**: Contact Tenstorrent for Wormhole and newer hardware

---

**Last Updated**: November 2024
**Target Hardware**: Tenstorrent Grayskull e75/e150
**Software Versions**: TT-Metalium v0.55, TT-Buda v0.19.3 (deprecated)
