/**
 * Galois Field GF(2^8) Arithmetic Operations
 *
 * Implements finite field arithmetic for Reed-Solomon erasure coding
 * Used in RAID 6 for dual parity (P+Q) calculations
 *
 * GF(2^8) operations with primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1
 *
 * Optimized for NVIDIA GPUs using constant memory for lookup tables
 */

#include <cuda_runtime.h>
#include <stdint.h>

// Constant memory for GF lookup tables (cached on-chip)
__constant__ uint8_t gf_exp[512];  // Exponential table (extended for overflow)
__constant__ uint8_t gf_log[256];  // Logarithm table
__constant__ uint8_t gf_inv[256];  // Multiplicative inverse table

/**
 * Device function: GF(2^8) multiplication using table lookup
 *
 * Identity: x * y = exp(log(x) + log(y)) in GF
 *
 * @param a: First operand
 * @param b: Second operand
 * @return Product in GF(2^8)
 */
__device__ __forceinline__ uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return gf_exp[gf_log[a] + gf_log[b]];
}

/**
 * Device function: GF(2^8) division
 *
 * @param a: Dividend
 * @param b: Divisor
 * @return Quotient in GF(2^8)
 */
__device__ __forceinline__ uint8_t gf_div(uint8_t a, uint8_t b) {
    if (a == 0) {
        return 0;
    }
    if (b == 0) {
        return 0; // Division by zero (should not happen)
    }
    return gf_exp[(gf_log[a] + 255 - gf_log[b]) % 255];
}

/**
 * Device function: GF(2^8) multiplicative inverse
 *
 * @param a: Element to invert
 * @return Multiplicative inverse of a
 */
__device__ __forceinline__ uint8_t gf_inverse(uint8_t a) {
    if (a == 0) {
        return 0;
    }
    return gf_inv[a];
}

/**
 * Device function: GF(2^8) power
 *
 * @param a: Base
 * @param n: Exponent
 * @return a^n in GF(2^8)
 */
__device__ __forceinline__ uint8_t gf_pow(uint8_t a, uint8_t n) {
    if (a == 0) {
        return 0;
    }
    if (n == 0) {
        return 1;
    }
    return gf_exp[(gf_log[a] * n) % 255];
}

/**
 * Vectorized GF(2^8) multiplication kernel
 *
 * Multiplies array by scalar in GF(2^8)
 * Used for encoding matrix multiplication in Reed-Solomon
 *
 * @param input: Input data array
 * @param output: Output array
 * @param scalar: GF(2^8) scalar multiplier
 * @param size: Number of bytes to process
 */
__global__ void gf_mul_scalar_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    uint8_t scalar,
    size_t size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Early exit if scalar is 0 or 1
    if (scalar == 0) {
        for (size_t i = tid; i < size; i += stride) {
            output[i] = 0;
        }
        return;
    }

    if (scalar == 1) {
        for (size_t i = tid; i < size; i += stride) {
            output[i] = input[i];
        }
        return;
    }

    // General case: multiply each element
    uint16_t log_scalar = gf_log[scalar];
    for (size_t i = tid; i < size; i += stride) {
        uint8_t val = input[i];
        if (val == 0) {
            output[i] = 0;
        } else {
            output[i] = gf_exp[gf_log[val] + log_scalar];
        }
    }
}

/**
 * GF(2^8) vector dot product kernel
 *
 * Computes: output[i] = coeffs[0]*input[0][i] + coeffs[1]*input[1][i] + ...
 * Addition is XOR in GF(2^8)
 *
 * Used for Reed-Solomon parity calculation
 *
 * @param input_vectors: Array of pointers to input vectors
 * @param coeffs: GF coefficients for each input
 * @param output: Output vector
 * @param num_vectors: Number of input vectors
 * @param size: Size of each vector in bytes
 */
__global__ void gf_vector_dot_product_kernel(
    const uint8_t** __restrict__ input_vectors,
    const uint8_t* __restrict__ coeffs,
    uint8_t* __restrict__ output,
    int num_vectors,
    size_t size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < size; i += stride) {
        uint8_t result = 0;

        #pragma unroll 4
        for (int v = 0; v < num_vectors; v++) {
            uint8_t coeff = coeffs[v];
            uint8_t value = input_vectors[v][i];

            // GF multiply and XOR (GF addition)
            if (coeff != 0 && value != 0) {
                result ^= gf_exp[gf_log[coeff] + gf_log[value]];
            }
        }

        output[i] = result;
    }
}

/**
 * Optimized GF matrix-vector multiplication
 *
 * Computes: output = matrix * input in GF(2^8)
 * Matrix stored in row-major order
 *
 * @param matrix: Encoding matrix (rows x cols)
 * @param input: Input vector
 * @param output: Output vector
 * @param rows: Number of rows in matrix
 * @param cols: Number of columns in matrix
 * @param vec_size: Size of each vector element block
 */
__global__ void gf_matrix_vector_mul_kernel(
    const uint8_t* __restrict__ matrix,
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows,
    int cols,
    size_t vec_size
) {
    int row = blockIdx.y;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    if (row >= rows) return;

    for (size_t i = tid; i < vec_size; i += stride) {
        uint8_t result = 0;

        for (int col = 0; col < cols; col++) {
            uint8_t coeff = matrix[row * cols + col];
            uint8_t value = input[col * vec_size + i];

            if (coeff != 0 && value != 0) {
                result ^= gf_exp[gf_log[coeff] + gf_log[value]];
            }
        }

        output[row * vec_size + i] = result;
    }
}

/**
 * Shared memory optimized GF matrix-vector multiplication
 *
 * Uses shared memory to cache matrix rows and input chunks
 * Better performance for smaller matrices
 *
 * @param matrix: Encoding matrix
 * @param input: Input vector
 * @param output: Output vector
 * @param rows: Number of matrix rows
 * @param cols: Number of matrix columns
 * @param vec_size: Vector element block size
 */
__global__ void gf_matrix_vector_mul_shared_kernel(
    const uint8_t* __restrict__ matrix,
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows,
    int cols,
    size_t vec_size
) {
    extern __shared__ uint8_t shared_mem[];
    uint8_t* shared_matrix = shared_mem;
    uint8_t* shared_input = shared_mem + cols;

    int row = blockIdx.y;
    size_t local_tid = threadIdx.x;

    if (row >= rows) return;

    // Load matrix row into shared memory
    if (local_tid < cols) {
        shared_matrix[local_tid] = matrix[row * cols + local_tid];
    }
    __syncthreads();

    // Process vector in chunks
    size_t chunk_size = blockDim.x;
    for (size_t base = 0; base < vec_size; base += chunk_size) {
        size_t i = base + local_tid;

        uint8_t result = 0;

        if (i < vec_size) {
            for (int col = 0; col < cols; col++) {
                uint8_t coeff = shared_matrix[col];
                uint8_t value = input[col * vec_size + i];

                if (coeff != 0 && value != 0) {
                    result ^= gf_exp[gf_log[coeff] + gf_log[value]];
                }
            }

            output[row * vec_size + i] = result;
        }

        __syncthreads();
    }
}

/**
 * Host function to initialize GF tables in constant memory
 *
 * Must be called before any GF operations
 * Uses primitive polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D)
 *
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t gf_init_tables() {
    uint8_t host_exp[512];
    uint8_t host_log[256];
    uint8_t host_inv[256];

    // Generate exponential and logarithm tables
    uint8_t x = 1;
    for (int i = 0; i < 255; i++) {
        host_exp[i] = x;
        host_log[x] = i;

        // Multiply by generator (2)
        x = (x << 1) ^ ((x & 0x80) ? 0x1D : 0);
    }

    // Extend exp table for overflow
    for (int i = 255; i < 512; i++) {
        host_exp[i] = host_exp[i - 255];
    }

    host_log[0] = 0; // Undefined, but set to 0

    // Generate inverse table
    host_inv[0] = 0;
    host_inv[1] = 1;
    for (int i = 2; i < 256; i++) {
        host_inv[i] = host_exp[255 - host_log[i]];
    }

    // Copy to constant memory
    cudaError_t err;
    err = cudaMemcpyToSymbol(gf_exp, host_exp, sizeof(host_exp));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(gf_log, host_log, sizeof(host_log));
    if (err != cudaSuccess) return err;

    err = cudaMemcpyToSymbol(gf_inv, host_inv, sizeof(host_inv));
    return err;
}

/**
 * Host function to launch GF scalar multiplication
 *
 * @param input_dev: Device input array
 * @param output_dev: Device output array
 * @param scalar: GF(2^8) scalar
 * @param size: Array size in bytes
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_gf_mul_scalar(
    const uint8_t* input_dev,
    uint8_t* output_dev,
    uint8_t scalar,
    size_t size,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = min(blocks_per_grid, 2048);

    gf_mul_scalar_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input_dev, output_dev, scalar, size
    );

    return cudaGetLastError();
}

/**
 * Host function to launch GF vector dot product
 *
 * @param input_vectors_dev: Device array of pointers to input vectors
 * @param coeffs_dev: Device array of GF coefficients
 * @param output_dev: Device output vector
 * @param num_vectors: Number of input vectors
 * @param size: Size of each vector
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_gf_vector_dot_product(
    const uint8_t** input_vectors_dev,
    const uint8_t* coeffs_dev,
    uint8_t* output_dev,
    int num_vectors,
    size_t size,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = min(blocks_per_grid, 2048);

    gf_vector_dot_product_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        input_vectors_dev, coeffs_dev, output_dev, num_vectors, size
    );

    return cudaGetLastError();
}
