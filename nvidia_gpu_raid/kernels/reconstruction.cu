/**
 * RAID Data Reconstruction Kernels
 *
 * Implements data recovery algorithms for RAID 5/6
 * Supports recovery from single or dual drive failures
 *
 * RAID 5: Single failure recovery using XOR
 * RAID 6: Single or dual failure recovery using Reed-Solomon decoding
 *
 * Optimized for NVIDIA Ampere/Turing architectures
 */

#include <cuda_runtime.h>
#include <stdint.h>

// External constant memory from galois_field.cu
extern __constant__ uint8_t gf_exp[512];
extern __constant__ uint8_t gf_log[256];
extern __constant__ uint8_t gf_inv[256];

/**
 * RAID 5 single block reconstruction kernel
 *
 * Recovers missing data block using XOR of surviving blocks and parity
 * Formula: missing = parity XOR d0 XOR d1 XOR ... XOR d(n-1)
 *
 * @param surviving_blocks: Array of pointers to surviving data blocks
 * @param parity: Parity block
 * @param recovered: Output recovered block
 * @param block_size: Size of each block in bytes
 * @param num_surviving: Number of surviving data blocks
 */
__global__ void raid5_reconstruct_kernel(
    const uint8_t** __restrict__ surviving_blocks,
    const uint8_t* __restrict__ parity,
    uint8_t* __restrict__ recovered,
    size_t block_size,
    int num_surviving
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t num_elements = block_size / sizeof(uint4);

    for (size_t i = tid; i < num_elements; i += stride) {
        // Start with parity
        const uint4* parity_ptr = reinterpret_cast<const uint4*>(parity);
        uint4 xor_result = parity_ptr[i];

        // XOR all surviving blocks
        #pragma unroll 4
        for (int b = 0; b < num_surviving; b++) {
            const uint4* data_ptr = reinterpret_cast<const uint4*>(surviving_blocks[b]);
            uint4 data = data_ptr[i];

            xor_result.x ^= data.x;
            xor_result.y ^= data.y;
            xor_result.z ^= data.z;
            xor_result.w ^= data.w;
        }

        // Write recovered data
        uint4* recovered_ptr = reinterpret_cast<uint4*>(recovered);
        recovered_ptr[i] = xor_result;
    }
}

/**
 * RAID 6 single failure reconstruction using P parity
 *
 * When only one block fails and P parity is intact, use simple XOR
 * This is faster than full Reed-Solomon decoding
 *
 * @param surviving_blocks: Array of pointers to surviving data blocks
 * @param parity_p: P parity block
 * @param recovered: Output recovered block
 * @param block_size: Size of each block in bytes
 * @param num_surviving: Number of surviving data blocks
 */
__global__ void raid6_reconstruct_single_p_kernel(
    const uint8_t** __restrict__ surviving_blocks,
    const uint8_t* __restrict__ parity_p,
    uint8_t* __restrict__ recovered,
    size_t block_size,
    int num_surviving
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t num_elements = block_size / sizeof(uint4);

    for (size_t i = tid; i < num_elements; i += stride) {
        const uint4* parity_ptr = reinterpret_cast<const uint4*>(parity_p);
        uint4 xor_result = parity_ptr[i];

        #pragma unroll 4
        for (int b = 0; b < num_surviving; b++) {
            const uint4* data_ptr = reinterpret_cast<const uint4*>(surviving_blocks[b]);
            uint4 data = data_ptr[i];

            xor_result.x ^= data.x;
            xor_result.y ^= data.y;
            xor_result.z ^= data.z;
            xor_result.w ^= data.w;
        }

        uint4* recovered_ptr = reinterpret_cast<uint4*>(recovered);
        recovered_ptr[i] = xor_result;
    }
}

/**
 * RAID 6 dual failure reconstruction using Reed-Solomon
 *
 * Recovers two missing blocks using P and Q parities
 * Solves the linear system in GF(2^8):
 *   P = sum(Di)
 *   Q = sum(g^i * Di)
 *
 * @param surviving_blocks: Pointers to surviving data blocks
 * @param parity_p: P parity block
 * @param parity_q: Q parity block
 * @param recovered_0: Output first recovered block
 * @param recovered_1: Output second recovered block
 * @param block_size: Size of each block in bytes
 * @param num_surviving: Number of surviving data blocks
 * @param gen_coeffs: Generator coefficients
 * @param failed_indices: Indices of two failed blocks
 */
__global__ void raid6_reconstruct_dual_kernel(
    const uint8_t** __restrict__ surviving_blocks,
    const uint8_t* __restrict__ parity_p,
    const uint8_t* __restrict__ parity_q,
    uint8_t* __restrict__ recovered_0,
    uint8_t* __restrict__ recovered_1,
    size_t block_size,
    int num_surviving,
    const uint8_t* __restrict__ gen_coeffs,
    const int* __restrict__ failed_indices
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    int idx0 = failed_indices[0];
    int idx1 = failed_indices[1];
    uint8_t g0 = gen_coeffs[idx0];
    uint8_t g1 = gen_coeffs[idx1];

    for (size_t i = tid; i < block_size; i += stride) {
        // Calculate syndrome S_P and S_Q
        uint8_t s_p = parity_p[i];
        uint8_t s_q = parity_q[i];

        // Subtract contributions from surviving blocks
        for (int b = 0; b < num_surviving; b++) {
            uint8_t data = surviving_blocks[b][i];
            s_p ^= data;

            // Find original index of this surviving block
            // (Assumes surviving blocks are in order with gaps for failed)
            int orig_idx = b < idx0 ? b : (b < idx1 - 1 ? b + 1 : b + 2);
            uint8_t coeff = gen_coeffs[orig_idx];

            if (data != 0 && coeff != 0) {
                s_q ^= gf_exp[gf_log[data] + gf_log[coeff]];
            }
        }

        // Solve 2x2 system in GF(2^8):
        // D0 + D1 = S_P
        // g0*D0 + g1*D1 = S_Q
        //
        // Solution:
        // D0 = (S_Q - g1*S_P) / (g0 - g1)
        // D1 = S_P - D0

        uint8_t denominator = (g0 == g1) ? 1 : gf_exp[(gf_log[g0] + 255 - gf_log[g1]) % 255];

        uint8_t numerator = s_q;
        if (g1 != 0 && s_p != 0) {
            numerator ^= gf_exp[gf_log[g1] + gf_log[s_p]];
        }

        uint8_t d0 = 0;
        if (numerator != 0 && denominator != 0) {
            d0 = gf_exp[(gf_log[numerator] + gf_log[gf_inv[denominator]]) % 255];
        }

        uint8_t d1 = s_p ^ d0;

        recovered_0[i] = d0;
        recovered_1[i] = d1;
    }
}

/**
 * Generic erasure decoding kernel using Gaussian elimination
 *
 * Solves system of linear equations in GF(2^8) for arbitrary erasures
 * More general but slower than specialized kernels
 *
 * @param surviving_data: Surviving data blocks
 * @param parity_blocks: Parity blocks (P and Q)
 * @param recovered_blocks: Output recovered blocks
 * @param block_size: Size of each block in bytes
 * @param num_surviving: Number of surviving data blocks
 * @param num_failed: Number of failed blocks (1 or 2)
 * @param encoding_matrix: Subset of encoding matrix for surviving+parity
 * @param failed_indices: Indices of failed blocks
 */
__global__ void erasure_decode_kernel(
    const uint8_t** __restrict__ surviving_data,
    const uint8_t** __restrict__ parity_blocks,
    uint8_t** __restrict__ recovered_blocks,
    size_t block_size,
    int num_surviving,
    int num_failed,
    const uint8_t* __restrict__ decoding_matrix,
    const int* __restrict__ failed_indices
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    // Each thread processes one byte position across all blocks
    for (size_t i = tid; i < block_size; i += stride) {
        // Compute syndrome vector
        uint8_t syndrome[2]; // Max 2 failures for RAID 6

        for (int f = 0; f < num_failed; f++) {
            uint8_t result = 0;

            // Contribution from parity blocks
            for (int p = 0; p < num_failed; p++) {
                result ^= parity_blocks[p][i];
            }

            // Subtract contribution from surviving data
            for (int s = 0; s < num_surviving; s++) {
                uint8_t data = surviving_data[s][i];
                uint8_t coeff = decoding_matrix[f * num_surviving + s];

                if (data != 0 && coeff != 0) {
                    result ^= gf_exp[gf_log[data] + gf_log[coeff]];
                }
            }

            syndrome[f] = result;
        }

        // Apply inverse of failed blocks' encoding matrix
        for (int f = 0; f < num_failed; f++) {
            uint8_t recovered_byte = 0;

            for (int s = 0; s < num_failed; s++) {
                uint8_t inv_coeff = decoding_matrix[(num_surviving + f) * num_failed + s];
                uint8_t syndrome_val = syndrome[s];

                if (syndrome_val != 0 && inv_coeff != 0) {
                    recovered_byte ^= gf_exp[gf_log[syndrome_val] + gf_log[inv_coeff]];
                }
            }

            recovered_blocks[f][i] = recovered_byte;
        }
    }
}

/**
 * Fast reconstruction verification kernel
 *
 * Verifies that reconstructed data produces correct parity
 * Used to validate recovery process
 *
 * @param all_blocks: All blocks including reconstructed
 * @param parity_p: P parity block
 * @param parity_q: Q parity block (nullptr if RAID 5)
 * @param gen_coeffs: Generator coefficients
 * @param verification_result: Output 1 if valid, 0 if mismatch
 * @param block_size: Size of each block in bytes
 * @param num_blocks: Total number of data blocks
 */
__global__ void verify_reconstruction_kernel(
    const uint8_t** __restrict__ all_blocks,
    const uint8_t* __restrict__ parity_p,
    const uint8_t* __restrict__ parity_q,
    const uint8_t* __restrict__ gen_coeffs,
    int* __restrict__ verification_result,
    size_t block_size,
    int num_blocks
) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    __shared__ int block_mismatch;
    if (threadIdx.x == 0) {
        block_mismatch = 0;
    }
    __syncthreads();

    for (size_t i = tid; i < block_size; i += stride) {
        uint8_t computed_p = 0;
        uint8_t computed_q = 0;

        for (int b = 0; b < num_blocks; b++) {
            uint8_t data = all_blocks[b][i];
            computed_p ^= data;

            if (parity_q != nullptr) {
                uint8_t coeff = gen_coeffs[b];
                if (data != 0 && coeff != 0) {
                    computed_q ^= gf_exp[gf_log[data] + gf_log[coeff]];
                }
            }
        }

        // Check P parity
        if (computed_p != parity_p[i]) {
            atomicOr(&block_mismatch, 1);
        }

        // Check Q parity if RAID 6
        if (parity_q != nullptr && computed_q != parity_q[i]) {
            atomicOr(&block_mismatch, 1);
        }
    }

    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        verification_result[0] = (block_mismatch == 0) ? 1 : 0;
    }
}

/**
 * Host function: Launch RAID 5 reconstruction
 *
 * @param surviving_blocks_dev: Device array of pointers to surviving blocks
 * @param parity_dev: Device pointer to parity block
 * @param recovered_dev: Device pointer for recovered output
 * @param block_size: Size of blocks in bytes
 * @param num_surviving: Number of surviving blocks
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_raid5_reconstruct(
    const uint8_t** surviving_blocks_dev,
    const uint8_t* parity_dev,
    uint8_t* recovered_dev,
    size_t block_size,
    int num_surviving,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    size_t num_elements = block_size / sizeof(uint4);
    int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = min(blocks_per_grid, 2048);

    raid5_reconstruct_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        surviving_blocks_dev,
        parity_dev,
        recovered_dev,
        block_size,
        num_surviving
    );

    return cudaGetLastError();
}

/**
 * Host function: Launch RAID 6 dual failure reconstruction
 *
 * @param surviving_blocks_dev: Device array of pointers to surviving blocks
 * @param parity_p_dev: Device pointer to P parity
 * @param parity_q_dev: Device pointer to Q parity
 * @param recovered_0_dev: Device pointer for first recovered block
 * @param recovered_1_dev: Device pointer for second recovered block
 * @param block_size: Size of blocks in bytes
 * @param num_surviving: Number of surviving blocks
 * @param gen_coeffs_dev: Device array of generator coefficients
 * @param failed_indices_dev: Device array of failed block indices
 * @param stream: CUDA stream
 * @return cudaError_t: CUDA error code
 */
extern "C" cudaError_t launch_raid6_reconstruct_dual(
    const uint8_t** surviving_blocks_dev,
    const uint8_t* parity_p_dev,
    const uint8_t* parity_q_dev,
    uint8_t* recovered_0_dev,
    uint8_t* recovered_1_dev,
    size_t block_size,
    int num_surviving,
    const uint8_t* gen_coeffs_dev,
    const int* failed_indices_dev,
    cudaStream_t stream = nullptr
) {
    int threads_per_block = 256;
    int blocks_per_grid = (block_size + threads_per_block - 1) / threads_per_block;
    blocks_per_grid = min(blocks_per_grid, 2048);

    raid6_reconstruct_dual_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        surviving_blocks_dev,
        parity_p_dev,
        parity_q_dev,
        recovered_0_dev,
        recovered_1_dev,
        block_size,
        num_surviving,
        gen_coeffs_dev,
        failed_indices_dev
    );

    return cudaGetLastError();
}
