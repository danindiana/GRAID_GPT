/**
 * ZSTD Compression CUDA Kernel (Simplified)
 *
 * GPU-accelerated ZSTD compression for ZFS integration
 * This is a simplified implementation focusing on entropy coding
 * Full ZSTD is complex; this provides basic functionality
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define ZSTD_BLOCK_SIZE_MAX (128 * 1024)
#define ZSTD_WINDOW_LOG_MAX 27

/* ZSTD frame header */
struct zstd_frame_header {
    uint32_t magic;              /* 0xFD2FB528 */
    uint8_t frame_desc;
    uint64_t window_size;
    uint64_t frame_content_size;
    uint32_t dict_id;
};

/* Huffman table */
#define ZSTD_HUFFMAN_SYMBOLS 256

struct huffman_table {
    uint16_t codes[ZSTD_HUFFMAN_SYMBOLS];
    uint8_t lengths[ZSTD_HUFFMAN_SYMBOLS];
};

/**
 * Build Huffman table from frequency counts
 */
__device__ void build_huffman_table(const uint32_t* freqs,
                                   struct huffman_table* table) {
    /* Simplified Huffman construction */
    /* In practice, would use canonical Huffman codes */

    /* For now, use a simple length encoding based on frequency */
    for (int i = 0; i < ZSTD_HUFFMAN_SYMBOLS; i++) {
        if (freqs[i] == 0) {
            table->lengths[i] = 0;
            table->codes[i] = 0;
        } else if (freqs[i] > 1000) {
            table->lengths[i] = 4;
            table->codes[i] = i & 0x0F;
        } else if (freqs[i] > 100) {
            table->lengths[i] = 8;
            table->codes[i] = i;
        } else {
            table->lengths[i] = 12;
            table->codes[i] = (i << 4) | (i & 0x0F);
        }
    }
}

/**
 * Count symbol frequencies
 */
__device__ void count_frequencies(const uint8_t* data, int size,
                                 uint32_t* freqs) {
    for (int i = 0; i < ZSTD_HUFFMAN_SYMBOLS; i++) {
        freqs[i] = 0;
    }

    for (int i = 0; i < size; i++) {
        atomicAdd(&freqs[data[i]], 1);
    }
}

/**
 * Simple run-length encoding (RLE)
 * Used for highly compressible data
 */
__device__ int rle_compress(const uint8_t* src, uint8_t* dst,
                           int src_size, int dst_capacity) {
    if (src_size == 0) return 0;

    int src_pos = 0;
    int dst_pos = 0;

    while (src_pos < src_size && dst_pos < dst_capacity - 2) {
        uint8_t symbol = src[src_pos];
        int run_len = 1;

        while (src_pos + run_len < src_size &&
               src[src_pos + run_len] == symbol &&
               run_len < 255) {
            run_len++;
        }

        if (run_len >= 3) {
            /* Encode as RLE: marker (0xFF) + symbol + length */
            dst[dst_pos++] = 0xFF;
            dst[dst_pos++] = symbol;
            dst[dst_pos++] = (uint8_t)run_len;
        } else {
            /* Just copy the bytes */
            for (int i = 0; i < run_len; i++) {
                if (dst_pos >= dst_capacity) break;
                dst[dst_pos++] = symbol;
            }
        }

        src_pos += run_len;
    }

    return dst_pos;
}

/**
 * Simplified ZSTD compression (device function)
 * Uses RLE + simple entropy coding
 */
__device__ int zstd_compress_simple_device(const uint8_t* src, uint8_t* dst,
                                          int src_size, int dst_capacity) {
    if (src_size > ZSTD_BLOCK_SIZE_MAX) {
        return -1;
    }

    uint32_t freqs[ZSTD_HUFFMAN_SYMBOLS];
    struct huffman_table huffman;

    /* Count symbol frequencies */
    count_frequencies(src, src_size, freqs);

    /* Check if data is highly compressible (lots of repeats) */
    uint32_t max_freq = 0;
    for (int i = 0; i < ZSTD_HUFFMAN_SYMBOLS; i++) {
        if (freqs[i] > max_freq) {
            max_freq = freqs[i];
        }
    }

    /* If one symbol dominates (>50%), use RLE */
    if (max_freq * 2 > src_size) {
        int compressed = rle_compress(src, dst + 4, src_size, dst_capacity - 4);
        if (compressed > 0 && compressed < src_size) {
            /* Write header: type (RLE) + size */
            dst[0] = 1;  /* RLE block */
            dst[1] = (uint8_t)(src_size & 0xFF);
            dst[2] = (uint8_t)((src_size >> 8) & 0xFF);
            dst[3] = (uint8_t)((src_size >> 16) & 0xFF);
            return compressed + 4;
        }
    }

    /* For other data, use raw block (no compression) */
    /* In full ZSTD, would use LZ77 + Huffman/FSE */
    if (src_size + 4 <= dst_capacity) {
        dst[0] = 0;  /* Raw block */
        dst[1] = (uint8_t)(src_size & 0xFF);
        dst[2] = (uint8_t)((src_size >> 8) & 0xFF);
        dst[3] = (uint8_t)((src_size >> 16) & 0xFF);
        memcpy(dst + 4, src, src_size);
        return src_size + 4;
    }

    return 0;  /* Incompressible */
}

/**
 * ZSTD compress kernel
 */
__global__ void zstd_compress_kernel(const uint8_t** src_blocks,
                                    uint8_t** dst_blocks,
                                    const int* src_sizes,
                                    int* dst_sizes,
                                    int dst_capacity,
                                    int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    const uint8_t* src = src_blocks[block_idx];
    uint8_t* dst = dst_blocks[block_idx];
    int src_size = src_sizes[block_idx];

    int compressed_size = zstd_compress_simple_device(src, dst, src_size,
                                                      dst_capacity);

    dst_sizes[block_idx] = compressed_size;
}

/**
 * RLE decompression
 */
__device__ int rle_decompress(const uint8_t* src, uint8_t* dst,
                             int src_size, int dst_capacity) {
    int src_pos = 0;
    int dst_pos = 0;

    while (src_pos < src_size && dst_pos < dst_capacity) {
        if (src[src_pos] == 0xFF && src_pos + 2 < src_size) {
            /* RLE sequence */
            uint8_t symbol = src[src_pos + 1];
            int run_len = src[src_pos + 2];

            for (int i = 0; i < run_len && dst_pos < dst_capacity; i++) {
                dst[dst_pos++] = symbol;
            }

            src_pos += 3;
        } else {
            /* Literal */
            dst[dst_pos++] = src[src_pos++];
        }
    }

    return dst_pos;
}

/**
 * ZSTD decompress (device function)
 */
__device__ int zstd_decompress_simple_device(const uint8_t* src, uint8_t* dst,
                                            int src_size, int dst_capacity) {
    if (src_size < 4) {
        return -1;
    }

    uint8_t block_type = src[0];
    int original_size = src[1] | (src[2] << 8) | (src[3] << 16);

    if (original_size > dst_capacity) {
        return -1;
    }

    if (block_type == 0) {
        /* Raw block */
        if (src_size - 4 != original_size) {
            return -1;
        }
        memcpy(dst, src + 4, original_size);
        return original_size;
    } else if (block_type == 1) {
        /* RLE block */
        return rle_decompress(src + 4, dst, src_size - 4, dst_capacity);
    }

    return -1;  /* Unknown block type */
}

/**
 * ZSTD decompress kernel
 */
__global__ void zstd_decompress_kernel(const uint8_t** src_blocks,
                                      uint8_t** dst_blocks,
                                      const int* src_sizes,
                                      int* dst_sizes,
                                      int dst_capacity,
                                      int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    const uint8_t* src = src_blocks[block_idx];
    uint8_t* dst = dst_blocks[block_idx];
    int src_size = src_sizes[block_idx];

    int decompressed_size = zstd_decompress_simple_device(src, dst, src_size,
                                                          dst_capacity);

    dst_sizes[block_idx] = decompressed_size;
}

/**
 * Host wrapper for ZSTD compression
 */
extern "C" int zstd_compress_gpu(const uint8_t** src_blocks, uint8_t** dst_blocks,
                                const int* src_sizes, int* dst_sizes,
                                int dst_capacity, int num_blocks,
                                cudaStream_t stream) {
    int threads_per_block = 128;
    int num_blocks_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    zstd_compress_kernel<<<num_blocks_grid, threads_per_block, 0, stream>>>(
        src_blocks, dst_blocks, src_sizes, dst_sizes, dst_capacity, num_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ZSTD compress kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Host wrapper for ZSTD decompression
 */
extern "C" int zstd_decompress_gpu(const uint8_t** src_blocks, uint8_t** dst_blocks,
                                  const int* src_sizes, int* dst_sizes,
                                  int dst_capacity, int num_blocks,
                                  cudaStream_t stream) {
    int threads_per_block = 128;
    int num_blocks_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    zstd_decompress_kernel<<<num_blocks_grid, threads_per_block, 0, stream>>>(
        src_blocks, dst_blocks, src_sizes, dst_sizes, dst_capacity, num_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ZSTD decompress kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Estimate ZSTD compression ratio
 */
extern "C" float zstd_estimate_ratio(const uint8_t* data, int size) {
    /* Sample first 4KB for estimate */
    int sample_size = (size > 4096) ? 4096 : size;

    /* Count unique bytes and repeats */
    int freqs[256] = {0};
    int repeats = 0;

    for (int i = 0; i < sample_size; i++) {
        freqs[data[i]]++;
        if (i > 0 && data[i] == data[i-1]) {
            repeats++;
        }
    }

    /* Calculate entropy */
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (freqs[i] > 0) {
            float prob = (float)freqs[i] / sample_size;
            entropy -= prob * log2f(prob);
        }
    }

    /* Estimate based on entropy and repeat ratio */
    float repeat_ratio = (float)repeats / sample_size;
    float entropy_ratio = entropy / 8.0f;  /* Normalized to 0-1 */

    /* High repeats = better compression */
    /* Low entropy = better compression */
    float estimated_ratio = (entropy_ratio * 0.7f) + ((1.0f - repeat_ratio) * 0.3f);

    /* Clamp to reasonable range */
    if (estimated_ratio < 0.3f) estimated_ratio = 0.3f;
    if (estimated_ratio > 1.0f) estimated_ratio = 1.0f;

    return estimated_ratio;
}

/**
 * Check if data is compressible
 */
extern "C" bool zstd_is_compressible(const uint8_t* data, int size) {
    float ratio = zstd_estimate_ratio(data, size);
    return ratio < 0.9f;  /* Worth compressing if we expect >10% reduction */
}
