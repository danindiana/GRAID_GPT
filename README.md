# GRAID_GPT

> GPU-Accelerated RAID with AI-Assisted Development

This repository explores GPU-accelerated RAID (Redundant Array of Independent Disks) implementations using Reed-Solomon coding and NVIDIA's cuSPARSELt library for sparse matrix operations.

## Table of Contents

- [Repository Structure](#repository-structure)
- [Project Architecture](#project-architecture)
- [Reed-Solomon Coding Workflow](#reed-solomon-coding-workflow)
- [GPU Architecture for RAID](#gpu-architecture-for-raid)
- [cuSPARSELt Implementation](#cusparselt-implementation)
- [Git Workflow](#git-workflow)
- [Technical Details](#technical-details)
- [Getting Started](#getting-started)

## Repository Structure

```mermaid
graph TD
    A[GRAID_GPT Repository] --> B[README.md]
    A --> C[LICENSE]
    A --> D[cuSPARSELt/]

    D --> E[cusparselt_matmul_basic.cpp]
    D --> F[cusparselt_matrix_descriptor_initialization.cpp]
    D --> G[readme.txt - API Documentation]

    B --> H[Project Documentation]
    E --> I[Basic Matrix Multiplication Example]
    F --> J[Descriptor Initialization Example]
    G --> K[cuSPARSELt API Reference]

    style A fill:#e1f5ff
    style D fill:#ffe1f5
    style B fill:#f5ffe1
```

## Project Architecture

```mermaid
graph TB
    subgraph "GRAID System Architecture"
        A[Data Input] --> B[Reed-Solomon Encoder]
        B --> C[GPU Processing Layer]
        C --> D[cuSPARSELt Operations]
        D --> E[Sparse Matrix Multiplication]
        E --> F[RAID Storage System]
        F --> G[Data Recovery Module]
        G --> H[Reed-Solomon Decoder]
        H --> I[Data Output]
    end

    subgraph "GPU Acceleration"
        C --> J[Multi-threaded Cores]
        J --> K[Shared Memory]
        K --> L[Constant Memory Cache]
    end

    subgraph "Error Correction"
        B --> M[k Data Buffers]
        B --> N[m Coding Buffers]
        M --> O[Redundancy Generation]
        N --> O
    end

    style A fill:#90EE90
    style I fill:#90EE90
    style C fill:#FFD700
    style F fill:#87CEEB
```

## Reed-Solomon Coding Workflow

```mermaid
flowchart TD
    A[Start: Data Stream] --> B{Split into k buffers}
    B --> C[Buffer 0: First s bytes]
    B --> D[Buffer 1: Next s bytes]
    B --> E[Buffer k-1: Last data bytes]

    C --> F[Information Dispersal Matrix A = I F]
    D --> F
    E --> F

    F --> G[Matrix Multiplication: c = F × d]
    G --> H[Generate m coding buffers]

    H --> I[Combined Buffer Space: e = d c]

    I --> J{Any k elements sufficient}
    J --> K[Data Recovery Possible]

    L[Finite Field Operations] --> M[XOR for Addition]
    L --> N[LFSR for Multiplication by 2]
    L --> O[Lookup Tables: exp and log]

    M --> G
    N --> G
    O --> G

    K --> P[Decoding: Recover missing data]
    P --> Q[Output: Recovered Data]

    style A fill:#90EE90
    style Q fill:#90EE90
    style F fill:#FFD700
    style L fill:#FF6B6B
```

## GPU Architecture for RAID

```mermaid
graph LR
    subgraph "GPU Hardware Architecture"
        A[GPU Device] --> B[240 Processing Cores]
        B --> C[30 Multiprocessors]
        C --> D[8 Cores per Multiprocessor]

        C --> E[Shared Memory Space]
        E --> F[16 Parallel Banks]
        F --> G[~1.82 requests/cycle/unit]

        C --> H[Constant Memory Cache]
        H --> I[Information Dispersal Matrix]
        I --> J[Fast Register-Speed Access]
    end

    subgraph "Memory Bus"
        K[384-bit Wide Bus] --> L[32-bit Parallel Transfers]
        L --> M[4 bytes per thread]
        M --> N[Latency Hiding]
    end

    subgraph "Processing Model"
        O[Parity Generation] --> P[Independent Byte Operations]
        P --> Q[Thread per 4-byte chunk]
        Q --> R[Parallel Table Lookups]
        R --> S[55 requests/cycle across chip]
    end

    B --> K
    D --> O

    style A fill:#4A90E2
    style E fill:#50C878
    style H fill:#FFD700
```

## cuSPARSELt Implementation

### Workflow Diagram

```mermaid
sequenceDiagram
    participant App as Application
    participant Handle as cuSPARSELt Handle
    participant MatDesc as Matrix Descriptors
    participant MulDesc as Matmul Descriptor
    participant AlgSel as Algorithm Selection
    participant Plan as Execution Plan
    participant GPU as GPU Device

    App->>Handle: cusparseLtInit()
    activate Handle

    App->>MatDesc: cusparseLtDenseDescriptorInit()
    App->>MatDesc: cusparseLtStructuredDescriptorInit()
    activate MatDesc

    App->>MulDesc: cusparseLtMatmulDescriptorInit()
    activate MulDesc
    Note over MulDesc: Configure operations<br/>opA, opB, compute type

    App->>AlgSel: cusparseLtMatmulAlgSelectionInit()
    activate AlgSel
    Note over AlgSel: Select algorithm<br/>Configure Split-K

    App->>Plan: cusparseLtMatmulPlanInit()
    activate Plan

    App->>GPU: cusparseLtMatmul()
    Note over GPU: D = α·op(A)·op(B) + β·C
    GPU-->>App: Result

    App->>Plan: cusparseLtMatmulPlanDestroy()
    deactivate Plan

    App->>MulDesc: Destroy descriptors
    deactivate MulDesc

    App->>MatDesc: cusparseLtMatDescriptorDestroy()
    deactivate MatDesc

    App->>Handle: cusparseLtDestroy()
    deactivate Handle
```

### Data Structure Relationships

```mermaid
classDiagram
    class cusparseLtHandle {
        +device properties
        +system information
        +Init()
        +Destroy()
    }

    class cusparseLtMatDescriptor {
        +rows: int64
        +cols: int64
        +leading dimension
        +alignment: uint32
        +valueType: cudaDataType
        +order: cusparseOrder
        +SetAttribute()
        +GetAttribute()
    }

    class cusparseLtMatmulDescriptor {
        +opA: cusparseOperation
        +opB: cusparseOperation
        +computeType: cusparseComputeType
        +activation functions
        +bias pointer
        +SetAttribute()
        +GetAttribute()
    }

    class cusparseLtMatmulAlgSelection {
        +algorithm: cusparseLtMatmulAlg
        +config ID
        +Split-K parameters
        +search iterations
        +SetAttribute()
        +GetAttribute()
    }

    class cusparseLtMatmulPlan {
        +execution plan
        +workspace size
        +Init()
        +Destroy()
    }

    cusparseLtHandle "1" --> "*" cusparseLtMatDescriptor : creates
    cusparseLtHandle "1" --> "*" cusparseLtMatmulDescriptor : creates
    cusparseLtMatDescriptor "3" --> "1" cusparseLtMatmulDescriptor : matA, matB, matC
    cusparseLtMatmulDescriptor "1" --> "1" cusparseLtMatmulAlgSelection : configures
    cusparseLtMatmulAlgSelection "1" --> "1" cusparseLtMatmulPlan : initializes
```

### Sparse Matrix Operations Pipeline

```mermaid
flowchart LR
    A[Dense Matrix Input] --> B[cusparseLtSpMMAPrune]
    B --> C{Pruning Algorithm}

    C --> D[TILE: 2:4 sparsity]
    C --> E[STRIP: directional pruning]

    D --> F[Pruned Matrix]
    E --> F

    F --> G[cusparseLtSpMMAPruneCheck]
    G --> H{Valid Structure?}

    H -->|Yes| I[cusparseLtSpMMACompress]
    H -->|No| J[Error: Invalid Pruning]

    I --> K[Compressed Matrix]
    K --> L[cusparseLtMatmul]
    L --> M[Result Matrix D]

    style A fill:#90EE90
    style M fill:#90EE90
    style J fill:#FF6B6B
    style K fill:#FFD700
```

## Git Workflow

```mermaid
gitGraph
    commit id: "Initial commit"
    commit id: "Add LICENSE"

    branch claude/feature-development
    checkout claude/feature-development
    commit id: "Add cuSPARSELt examples"
    commit id: "Create basic matmul example"
    commit id: "Add descriptor initialization"

    checkout main
    merge claude/feature-development

    branch claude/add-git-mermaid-diagrams-011nX8usJxjEhHyWakMgR3jX
    checkout claude/add-git-mermaid-diagrams-011nX8usJxjEhHyWakMgR3jX
    commit id: "Update readme.txt"
    commit id: "Add cuSPARSELt programs"
    commit id: "Enhance documentation" type: HIGHLIGHT

    checkout main
    commit id: "Continue main development"
```

## Technical Details

### Reed-Solomon Coding for RAID

The primary operation in Reed-Solomon coding involves multiplying **F** (the lower m rows of an information dispersal matrix **A = [I F]**) with a vector of data elements **d**. This results in another vector of redundant elements (the coding vector, **c**).

#### Key Properties

- **Redundancy**: Any k elements of **e = [d c]** may be used to recover **d**, even if some (or all) elements of **d** are not available
- **Finite Field Operations**:
  - XOR for addition
  - Linear Feedback Shift Register (LFSR) for multiplication by two
  - Identity: `x × y = exp(log(x) + log(y))` in finite fields of size 2^w

#### Optimization for RAID Systems

For RAID systems with **w = 8** (8 bits per symbol):
- Pre-calculated lookup tables for `exp` and `log` operations (256 bytes each)
- Multiplication implemented with 3 table lookups + addition modulo 2^w - 1
- Significantly faster than traditional logical operations

### Mapping Reed-Solomon Coding to GPUs

#### Buffer Space Organization

```
Buffer Space Layout:
┌─────────────┬─────────────┬───┬─────────────┬─────────────┬───┬─────────────┐
│  Buffer 0   │  Buffer 1   │...│  Buffer k-1 │  Buffer k   │...│ Buffer k+m-1│
│  (Data)     │  (Data)     │   │  (Data)     │  (Coding)   │   │  (Coding)   │
│  s bytes    │  s bytes    │   │  s bytes    │  s bytes    │   │  s bytes    │
└─────────────┴─────────────┴───┴─────────────┴─────────────┴───┴─────────────┘
     ← k data buffers →              ← m coding buffers →
```

- **Buffer**: A specific slice of data of size **s** bytes
- **k**: Number of data buffers
- **m**: Number of coding (redundancy) buffers
- **Buffer Space**: Continuous memory storing all buffers together

#### GPU Architecture Benefits

**NVIDIA GeForce GTX 285 Specifications:**
- 240 processing cores across 30 multiprocessors
- 8 cores per multiprocessor
- 384-bit memory bus for parallel transfers
- Each thread handles 4 bytes per buffer

**Performance Characteristics:**
- Shared memory: 16 parallel banks
- ~4.4 accesses per bank on average for 32 simultaneous threads
- 4 clock cycles per access
- **55 requests per cycle** across the entire chip

**Constant Memory Optimization:**
- Ideal for storing the information dispersal matrix
- Cached at multiprocessor level
- Access speed comparable to registers
- Perfect for lock-step access patterns

## cuSPARSELt Implementation

### Supported Data Types and Compute Modes

| Input Type | Output Type | Compute Type | Description |
|------------|-------------|--------------|-------------|
| CUDA_R_16F | CUDA_R_16F | CUSPARSE_COMPUTE_16F | 16-bit floating-point |
| CUDA_R_16BF | CUDA_R_16BF | CUSPARSE_COMPUTE_16F | 16-bit bfloat |
| CUDA_R_8I | CUDA_R_8I | CUSPARSE_COMPUTE_32I | 8-bit integer |
| CUDA_R_8I | CUDA_R_16F | CUSPARSE_COMPUTE_32I | Mixed precision |
| CUDA_R_32F | CUDA_R_32F | CUSPARSE_COMPUTE_TF32_FAST | TensorFloat-32 (fast) |
| CUDA_R_32F | CUDA_R_32F | CUSPARSE_COMPUTE_TF32 | TensorFloat-32 (accurate) |

### Sparsity Patterns

The library supports **50% sparsity ratio** (CUSPARSELT_SPARSITY_50_PERCENT):
- **2:4 pattern** for half, bfloat16, int8 (2 non-zeros per 4 elements)
- **1:2 pattern** for tf32 and float (1 non-zero per 2 elements)

### Matrix Operation Formula

The cuSPARSELt library computes:

```
D = Activation(α · op(A) · op(B) + β · C + bias) · scale
```

Where:
- **A, B, C**: Input matrices (A or B is structured/sparse)
- **D**: Output matrix
- **α, β**: Scalars or vectors of scalars
- **op()**: Optional transpose operation
- **Activation**: Optional ReLU or GeLU
- **bias**: Optional bias vector
- **scale**: Optional per-channel scaling

## Getting Started

### Prerequisites

- CUDA Toolkit (11.0 or later)
- NVIDIA GPU with Compute Capability 8.0+ (Ampere architecture or newer)
- cuSPARSELt library
- C++14 compatible compiler

### Building the Examples

```bash
# Navigate to cuSPARSELt directory
cd cuSPARSELt

# Compile basic matrix multiplication example
nvcc -lcusparseLt cusparselt_matmul_basic.cpp -o matmul_basic

# Compile descriptor initialization example
nvcc -lcusparseLt cusparselt_matrix_descriptor_initialization.cpp -o descriptor_init

# Run examples
./matmul_basic
./descriptor_init
```

### Example Usage

See the example programs in the `cuSPARSELt/` directory:

1. **cusparselt_matmul_basic.cpp**: Demonstrates the complete workflow for sparse matrix multiplication
2. **cusparselt_matrix_descriptor_initialization.cpp**: Shows how to initialize and configure matrix descriptors

## Project Roadmap

- [x] Basic cuSPARSELt examples
- [x] Documentation with mermaid diagrams
- [ ] Performance benchmarks
- [ ] Integration with actual RAID systems
- [ ] Python bindings
- [ ] Comparative analysis with CPU implementations

## Contributing

Contributions are welcome! Please ensure:

1. Code follows CUDA best practices
2. Examples are well-documented
3. Performance implications are noted
4. Tests are included where applicable

## References

- [NVIDIA cuSPARSELt Documentation](https://docs.nvidia.com/cuda/cusparselt/)
- Reed-Solomon coding for RAID systems
- GPU architecture optimization techniques

## License

See [LICENSE](LICENSE) file for details.

---

**Note**: This project combines theoretical concepts of Reed-Solomon coding with practical GPU implementations using NVIDIA's cuSPARSELt library for high-performance RAID systems.
