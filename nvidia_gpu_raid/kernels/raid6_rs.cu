/**
 * RAID 6 Reed-Solomon P+Q Parity Computation Kernels
 *
 * Implements dual parity (P and Q) using Reed-Solomon erasure coding
 * Allows recovery from up to 2 simultaneous drive failures
 *
 * P parity: Simple XOR (same as RAID 5)
 * Q parity: Reed-Solomon using Galois Field GF(2^8) arithmetic
 *
 * Optimized for NVIDIA Ampere/Turing architectures
 */

#include <cuda_runtime.h>
#include <stdint.h>

// External constant memory from galois_field.cu
extern __constant__ uint8_t gf_exp[512];
extern __constant__ uint8_t gf_log[256];

/**
 * Combined P+Q parity kernel (optimized)
 *
 * Computes both P and Q parities in a single pass
 * P: Simple XOR of all data blocks
 * Q: Reed-Solomon with generator coefficients g^0, g^1, ..., g^(k-1)
 *
 * @param data_blocks: Array of pointers to k data blocks
 * @param parity_p: Output P parity block
 * @param parity_q: Output Q parity block
 * @param block_size: Size of each block in bytes
 * @param num_blocks: Number of data blocks (k)
 * @param gen_coeffs: Generator coefficients for Q (powers of generator)
 */
__global__ void raid6_encode_kernel(
    const uint8_t** __restrict__ data_blocks,
    uint8_t* __restrict__ parity_p,
    uint8_t* __restrict__ parity_q,
    size_t block_size,
    int num_blocks,
    const uint8_t* __restrict__ gen_coeffs
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < block_size; i += stride) {
        uint8_t p_result = 0;
        uint8_t q_result = 0;

        // Process each data block
        #pragma unroll 4
        for (int b = 0; b < num_blocks; b++) {
            uint8_t data_byte = data_blocks[b][i];
            uint8_t coeff = gen_coeffs[b];

            // P parity: XOR
            p_result ^= data_byte;

            // Q parity: GF(2^8) multiply and XOR
            if (data_byte != 0 && coeff != 0) {
                q_result ^= gf_exp[gf_log[data_byte] + gf_log[coeff]];
            }
        }

        parity_p[i] = p_result;
        parity_q[i] = q_result;
    }
}

/**
 * Vectorized P+Q parity kernel using 128-bit operations
 *
 * Processes 16 bytes at a time for better memory bandwidth utilization
 * Unpacks for GF operations, then repacks for writes
 *
 * @param data_blocks: Array of pointers to k data blocks
 * @param parity_p: Output P parity block
 * @param parity_q: Output Q parity block
 * @param block_size: Size of each block in bytes (must be multiple of 16)
 * @param num_blocks: Number of data blocks
 * @param gen_coeffs: Generator coefficients
 */
__global__ void raid6_encode_vectorized_kernel(
    const uint8_t** __restrict__ data_blocks,
    uint8_t* __restrict__ parity_p,
    uint8_t* __restrict__ parity_q,
    size_t block_size,
    int num_blocks,
    const uint8_t* __restrict__ gen_coeffs
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t num_elements = block_size / sizeof(uint4);

    for (size_t idx = tid; idx < num_elements; idx += stride) {
        // Load 16 bytes at a time
        uint4 p_vec = make_uint4(0, 0, 0, 0);
        uint32_t q_bytes[4] = {0, 0, 0, 0};

        for (int b = 0; b < num_blocks; b++) {
            const uint4* data_ptr = reinterpret_cast<const uint4*>(data_blocks[b]);
            uint4 data_vec = data_ptr[idx];
            uint8_t coeff = gen_coeffs[b];

            // P parity (XOR)
            p_vec.x ^= data_vec.x;
            p_vec.y ^= data_vec.y;
            p_vec.z ^= data_vec.z;
            p_vec.w ^= data_vec.w;

            // Q parity (GF operations on each byte)
            uint8_t* data_bytes = reinterpret_cast<uint8_t*>(&data_vec);
            for (int i = 0; i < 16; i++) {
                uint8_t data_byte = data_bytes[i];
                if (data_byte != 0 && coeff != 0) {
                    uint8_t gf_product = gf_exp[gf_log[data_byte] + gf_log[coeff]];
                    reinterpret_cast<uint8_t*>(q_bytes)[i] ^= gf_product;
                }
            }
        }

        // Write results
        uint4* p_ptr = reinterpret_cast<uint4*>(parity_p);
        p_ptr[idx] = p_vec;

        uint4* q_ptr = reinterpret_cast<uint4*>(parity_q);
        q_ptr[idx] = *reinterpret_cast<uint4*>(q_bytes);
    }
}

/**
 * Incremental P+Q update kernel
 *
 * Efficiently updates P and Q parities when a single data block changes
 * Much faster than full re-encoding
 *
 * Formula:
 *   new_P = old_P XOR old_data XOR new_data
 *   new_Q = old_Q XOR (coeff * old_data) XOR (coeff * new_data)
 *
 * @param old_parity_p: Current P parity
 * @param old_parity_q: Current Q parity
 * @param new_parity_p: Output updated P parity
 * @param new_parity_q: Output updated Q parity
 * @param old_data: Old data block being replaced
 * @param new_data: New data block
 * @param block_index: Index of block being updated (for Q coefficient)
 * @param gen_coeffs: Generator coefficients
 * @param block_size: Size of blocks in bytes
 */
__global__ void raid6_update_kernel(
    const uint8_t* __restrict__ old_parity_p,
    const uint8_t* __restrict__ old_parity_q,
    uint8_t* __restrict__ new_parity_p,
    uint8_t* __restrict__ new_parity_q,
    const uint8_t* __restrict__ old_data,
    const uint8_t* __restrict__ new_data,
    int block_index,
    const uint8_t* __restrict__ gen_coeffs,
    size_t block_size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    uint8_t coeff = gen_coeffs[block_index];

    for (size_t i = tid; i < block_size; i += stride) {
        uint8_t old_p = old_parity_p[i];
        uint8_t old_q = old_parity_q[i];
        uint8_t old_d = old_data[i];
        uint8_t new_d = new_data[i];

        // Update P (simple XOR)
        uint8_t new_p = old_p ^ old_d ^ new_d;

        // Update Q (GF multiply)
        uint8_t old_contrib = 0;
        uint8_t new_contrib = 0;

        if (old_d != 0 && coeff != 0) {
            old_contrib = gf_exp[gf_log[old_d] + gf_log[coeff]];
        }
        if (new_d != 0 && coeff != 0) {
            new_contrib = gf_exp[gf_log[new_d] + gf_log[coeff]];
        }

        uint8_t new_q = old_q ^ old_contrib ^ new_contrib;

        new_parity_p[i] = new_p;
        new_parity_q[i] = new_q;
    }
}

/**
 * Separate P parity kernel (optimized XOR only)
 *
 * When Q parity is not needed, use this faster P-only kernel
 *
 * @param data_blocks: Array of pointers to data blocks
 * @param parity_p: Output P parity block
 * @param block_size: Size of each block in bytes
 * @param num_blocks: Number of data blocks
 */
__global__ void raid6_encode_p_only_kernel(
    const uint8_t** __restrict__ data_blocks,
    uint8_t* __restrict__ parity_p,
    size_t block_size,
    int num_blocks
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t num_elements = block_size / sizeof(uint4);

    for (size_t i = tid; i < num_elements; i += stride) {
        uint4 xor_result = make_uint4(0, 0, 0, 0);

        #pragma unroll 4
        for (int b = 0; b < num_blocks; b++) {
            const uint4* data_ptr = reinterpret_cast<const uint4*>(data_blocks[b]);
            uint4 data = data_ptr[i];

            xor_result.x ^= data.x;
            xor_result.y ^= data.y;
            xor_result.z ^= data.z;
            xor_result.w ^= data.w;
        }

        uint4* parity_ptr = reinterpret_cast<uint4*>(parity_p);
        parity_ptr[i] = xor_result;
    }
}

/**
 * Separate Q parity kernel
 *
 * Computes only Q parity using Reed-Solomon
 * Useful when P is already known/cached
 *
 * @param data_blocks: Array of pointers to data blocks
 * @param parity_q: Output Q parity block
 * @param block_size: Size of each block in bytes
 * @param num_blocks: Number of data blocks
 * @param gen_coeffs: Generator coefficients
 */
__global__ void raid6_encode_q_only_kernel(
    const uint8_t** __restrict__ data_blocks,
    uint8_t* __restrict__ parity_q,
    size_t block_size,
    int num_blocks,
    const uint8_t* __restrict__ gen_coeffs
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = tid; i < block_size; i += stride) {
        uint8_t q_result = 0;

        #pragma unroll 4
        for (int b = 0; b < num_blocks; b++) {
            uint8_t data_byte = data_blocks[b][i];
            uint8_t coeff = gen_coeffs[b];

            if (data_byte != 0 && coeff != 0) {
                q_result ^= gf_exp[gf_log[data_byte] + gf_log[coeff]];
            }
        }

        parity_q[i] = q_result;
    }
}

/**
 * Multi-stream P+Q encoding kernel
 *
 * Processes a portion of data for stream-based parallelism
 *
 * @param data_blocks: Array of pointers to data blocks
 * @param parity_p: Output P parity block
 * @param parity_q: Output Q parity block
 * @param block_size: Total block size
 * @param num_blocks: Number of data blocks
 * @param gen_coeffs: Generator coefficients
 * @param stream_offset: Offset for this stream
 * @param stream_size: Size for this stream
 */
__global__ void raid6_encode_stream_kernel(
    const uint8_t** __restrict__ data_blocks,
    uint8_t* __restrict__ parity_p,
    uint8_t* __restrict__ parity_q,
    size_t block_size,
    int num_blocks,
    const uint8_t* __restrict__ gen_coeffs,
    size_t stream_offset,
    size_t stream_size
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    size_t start = stream_offset;
    size_t end = min(stream_offset + stream_size, block_size);

    for (size_t i = start + tid; i < end; i += stride) {
        uint8_t p_result = 0;
        uint8_t q_result = 0;

        for (int b = 0; b < num_blocks; b++) {
            uint8_t data_byte = data_blocks[b][i];
            uint8_t coeff = gen_coeffs[b];

            p_result ^= data_byte;

            if (data_byte != 0 && coeff != 0) {
                q_result ^= gf_exp[gf_log[data_byte] + gf_log[coeff]];
            }
        }

        parity_p[i] = p_result;
        parity_q[i] = q_result;
    }
}

/**
 * Host function: Generate Vandermonde coefficients for RAID 6
 *
 * Creates generator matrix coefficients: g^0, g^1, g^2, ..., g^(k-1)
 * Where g is the generator element (typically 2 in GF(2^8))
 *
 * @param coeffs_dev: Device array to store coefficients (size: num_blocks)
 * @param num_blocks: Number of data blocks (k)
 * @param generator: Generator element (default: 2)
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t raid6_generate_coefficients(
    uint8_t* coeffs_dev,
    int num_blocks,
    uint8_t generator = 2
) {
    uint8_t* coeffs_host = new uint8_t[num_blocks];

    coeffs_host[0] = 1; // g^0 = 1
    for (int i = 1; i < num_blocks; i++) {
        coeffs_host[i] = (coeffs_host[i-1] << 1) ^ ((coeffs_host[i-1] & 0x80) ? 0x1D : 0);
    }

    cudaError_t err = cudaMemcpy(coeffs_dev, coeffs_host, num_blocks, cudaMemcpyHostToDevice);
    delete[] coeffs_host;

    return err;
}

/**
 * Host function: Launch RAID 6 P+Q encoding kernel
 *
 * @param data_blocks_dev: Device array of pointers to data blocks
 * @param parity_p_dev: Device pointer to P parity output
 * @param parity_q_dev: Device pointer to Q parity output
 * @param block_size: Size of each block in bytes
 * @param num_blocks: Number of data blocks
 * @param gen_coeffs_dev: Device array of generator coefficients
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_raid6_encode(
    const uint8_t** data_blocks_dev,
    uint8_t* parity_p_dev,
    uint8_t* parity_q_dev,
    size_t block_size,
    int num_blocks,
    const uint8_t* gen_coeffs_dev,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (block_size + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = min(blocks_per_grid, 2048);

    raid6_encode_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        data_blocks_dev,
        parity_p_dev,
        parity_q_dev,
        block_size,
        num_blocks,
        gen_coeffs_dev
    );

    return cudaGetLastError();
}

/**
 * Host function: Launch RAID 6 P+Q update kernel
 *
 * @param old_parity_p_dev: Device pointer to old P parity
 * @param old_parity_q_dev: Device pointer to old Q parity
 * @param new_parity_p_dev: Device pointer to new P parity output
 * @param new_parity_q_dev: Device pointer to new Q parity output
 * @param old_data_dev: Device pointer to old data
 * @param new_data_dev: Device pointer to new data
 * @param block_index: Index of block being updated
 * @param gen_coeffs_dev: Device array of generator coefficients
 * @param block_size: Size of blocks
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_raid6_update(
    const uint8_t* old_parity_p_dev,
    const uint8_t* old_parity_q_dev,
    uint8_t* new_parity_p_dev,
    uint8_t* new_parity_q_dev,
    const uint8_t* old_data_dev,
    const uint8_t* new_data_dev,
    int block_index,
    const uint8_t* gen_coeffs_dev,
    size_t block_size,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (block_size + threads_per_block - 1) / threads_per_block;

    raid6_update_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        old_parity_p_dev,
        old_parity_q_dev,
        new_parity_p_dev,
        new_parity_q_dev,
        old_data_dev,
        new_data_dev,
        block_index,
        gen_coeffs_dev,
        block_size
    );

    return cudaGetLastError();
}
