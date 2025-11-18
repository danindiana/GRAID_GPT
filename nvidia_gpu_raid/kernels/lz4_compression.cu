/**
 * LZ4 Compression CUDA Kernel
 *
 * GPU-accelerated LZ4 compression for ZFS integration
 * Based on LZ4 specification: https://github.com/lz4/lz4/blob/dev/doc/lz4_Block_format.md
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>

#define LZ4_MEMORY_USAGE 14
#define LZ4_HASH_SIZE_U32 (1 << LZ4_MEMORY_USAGE)
#define LZ4_HASH_SIZE (LZ4_HASH_SIZE_U32 * sizeof(uint32_t))

#define LZ4_MIN_MATCH 4
#define LZ4_MAX_DISTANCE 65535
#define LZ4_ML_BITS 4
#define LZ4_ML_MASK ((1U << LZ4_ML_BITS) - 1)
#define LZ4_RUN_BITS (8 - LZ4_ML_BITS)
#define LZ4_RUN_MASK ((1U << LZ4_RUN_BITS) - 1)

/**
 * LZ4 hash function
 */
__device__ __forceinline__ uint32_t lz4_hash(uint32_t value) {
    return (value * 2654435761U) >> (32 - LZ4_MEMORY_USAGE);
}

/**
 * Read 32-bit value from memory (little-endian)
 */
__device__ __forceinline__ uint32_t read32(const uint8_t* ptr) {
    return *((const uint32_t*)ptr);
}

/**
 * Write 16-bit value to memory (little-endian)
 */
__device__ __forceinline__ void write16(uint8_t* ptr, uint16_t value) {
    *((uint16_t*)ptr) = value;
}

/**
 * Count matching bytes
 */
__device__ int count_match(const uint8_t* src, const uint8_t* match,
                          const uint8_t* src_end) {
    int count = 0;
    while (src < src_end && *src == *match) {
        src++;
        match++;
        count++;
    }
    return count;
}

/**
 * LZ4 compress a single block (device function)
 */
__device__ int lz4_compress_block_device(const uint8_t* src, uint8_t* dst,
                                         int src_size, int dst_capacity,
                                         uint32_t* hash_table) {
    const uint8_t* src_base = src;
    const uint8_t* src_end = src + src_size;
    const uint8_t* src_limit = src_end - 12;  /* Minimum match size + safety */

    uint8_t* dst_base = dst;
    uint8_t* dst_end = dst + dst_capacity;

    const uint8_t* anchor = src;
    uint8_t* op = dst;

    /* Initialize hash table */
    for (int i = 0; i < LZ4_HASH_SIZE_U32; i++) {
        hash_table[i] = 0;
    }

    /* First byte never matches */
    src++;

    while (src < src_limit) {
        /* Find match */
        const uint8_t* match;
        uint32_t forward_h;
        int find_match_attempts = 64;

        do {
            uint32_t h = lz4_hash(read32(src));
            uint32_t match_pos = hash_table[h];
            hash_table[h] = (uint32_t)(src - src_base);

            match = src_base + match_pos;

            /* Check if match is valid */
            if (match_pos == 0 ||
                (src - match) > LZ4_MAX_DISTANCE ||
                read32(match) != read32(src)) {
                src++;
                if (--find_match_attempts == 0 || src >= src_limit) {
                    goto _last_literals;
                }
                continue;
            }

            break;
        } while (true);

        /* Found a match - encode literal run */
        int literal_len = (int)(src - anchor);

        if (op + literal_len + 2 + 2 > dst_end) {
            return 0;  /* Not enough space */
        }

        /* Encode token */
        uint8_t* token = op++;

        if (literal_len >= LZ4_RUN_MASK) {
            *token = (LZ4_RUN_MASK << LZ4_ML_BITS);
            int len = literal_len - LZ4_RUN_MASK;
            for (; len >= 255; len -= 255) {
                *op++ = 255;
            }
            *op++ = (uint8_t)len;
        } else {
            *token = (uint8_t)(literal_len << LZ4_ML_BITS);
        }

        /* Copy literals */
        const uint8_t* lit = anchor;
        uint8_t* out_lit = op;
        op += literal_len;

        for (int i = 0; i < literal_len; i++) {
            out_lit[i] = lit[i];
        }

        /* Encode offset */
        write16(op, (uint16_t)(src - match));
        op += 2;

        /* Count match length */
        src += LZ4_MIN_MATCH;
        match += LZ4_MIN_MATCH;
        int match_len = count_match(src, match, src_end);
        src += match_len;
        match_len += LZ4_MIN_MATCH;

        /* Encode match length */
        if (match_len >= LZ4_ML_MASK + LZ4_MIN_MATCH) {
            *token |= LZ4_ML_MASK;
            match_len -= LZ4_ML_MASK + LZ4_MIN_MATCH;
            for (; match_len >= 255; match_len -= 255) {
                *op++ = 255;
            }
            *op++ = (uint8_t)match_len;
        } else {
            *token |= (uint8_t)(match_len - LZ4_MIN_MATCH);
        }

        anchor = src;
    }

_last_literals:
    /* Encode last literals */
    int last_run = (int)(src_end - anchor);
    if (op + last_run + 1 + ((last_run + 255 - LZ4_RUN_MASK) / 255) > dst_end) {
        return 0;
    }

    if (last_run >= LZ4_RUN_MASK) {
        *op++ = (LZ4_RUN_MASK << LZ4_ML_BITS);
        int len = last_run - LZ4_RUN_MASK;
        for (; len >= 255; len -= 255) {
            *op++ = 255;
        }
        *op++ = (uint8_t)len;
    } else {
        *op++ = (uint8_t)(last_run << LZ4_ML_BITS);
    }

    for (int i = 0; i < last_run; i++) {
        op[i] = anchor[i];
    }
    op += last_run;

    return (int)(op - dst_base);
}

/**
 * LZ4 compress kernel - each block processes one input block
 */
__global__ void lz4_compress_kernel(const uint8_t** src_blocks,
                                   uint8_t** dst_blocks,
                                   const int* src_sizes,
                                   int* dst_sizes,
                                   int dst_capacity,
                                   int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (block_idx >= num_blocks) {
        return;
    }

    /* Allocate hash table in shared memory for this block */
    __shared__ uint32_t hash_table[LZ4_HASH_SIZE_U32];

    const uint8_t* src = src_blocks[block_idx];
    uint8_t* dst = dst_blocks[block_idx];
    int src_size = src_sizes[block_idx];

    int compressed_size = lz4_compress_block_device(src, dst, src_size,
                                                    dst_capacity, hash_table);

    dst_sizes[block_idx] = compressed_size;
}

/**
 * LZ4 decompress a single block (device function)
 */
__device__ int lz4_decompress_block_device(const uint8_t* src, uint8_t* dst,
                                          int src_size, int dst_capacity) {
    const uint8_t* src_end = src + src_size;
    uint8_t* dst_base = dst;
    uint8_t* dst_end = dst + dst_capacity;

    while (src < src_end) {
        /* Read token */
        uint8_t token = *src++;

        /* Decode literal length */
        int literal_len = token >> LZ4_ML_BITS;
        if (literal_len == LZ4_RUN_MASK) {
            uint8_t s;
            do {
                s = *src++;
                literal_len += s;
            } while (s == 255);
        }

        /* Copy literals */
        if (dst + literal_len > dst_end) {
            return -1;  /* Output buffer overflow */
        }

        for (int i = 0; i < literal_len; i++) {
            dst[i] = src[i];
        }
        src += literal_len;
        dst += literal_len;

        if (src >= src_end) {
            break;  /* End of compressed block */
        }

        /* Read offset */
        uint16_t offset = *((const uint16_t*)src);
        src += 2;

        /* Decode match length */
        int match_len = (token & LZ4_ML_MASK) + LZ4_MIN_MATCH;
        if ((token & LZ4_ML_MASK) == LZ4_ML_MASK) {
            uint8_t s;
            do {
                s = *src++;
                match_len += s;
            } while (s == 255);
        }

        /* Copy match */
        uint8_t* match = dst - offset;
        if (match < dst_base) {
            return -1;  /* Invalid offset */
        }

        if (dst + match_len > dst_end) {
            return -1;  /* Output buffer overflow */
        }

        /* Handle overlapping matches */
        for (int i = 0; i < match_len; i++) {
            dst[i] = match[i];
        }
        dst += match_len;
    }

    return (int)(dst - dst_base);
}

/**
 * LZ4 decompress kernel
 */
__global__ void lz4_decompress_kernel(const uint8_t** src_blocks,
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

    int decompressed_size = lz4_decompress_block_device(src, dst, src_size, dst_capacity);

    dst_sizes[block_idx] = decompressed_size;
}

/**
 * Host wrapper for LZ4 compression
 */
extern "C" int lz4_compress_gpu(const uint8_t** src_blocks, uint8_t** dst_blocks,
                               const int* src_sizes, int* dst_sizes,
                               int dst_capacity, int num_blocks,
                               cudaStream_t stream) {
    int threads_per_block = 128;
    int num_blocks_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    lz4_compress_kernel<<<num_blocks_grid, threads_per_block, 0, stream>>>(
        src_blocks, dst_blocks, src_sizes, dst_sizes, dst_capacity, num_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "LZ4 compress kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Host wrapper for LZ4 decompression
 */
extern "C" int lz4_decompress_gpu(const uint8_t** src_blocks, uint8_t** dst_blocks,
                                 const int* src_sizes, int* dst_sizes,
                                 int dst_capacity, int num_blocks,
                                 cudaStream_t stream) {
    int threads_per_block = 128;
    int num_blocks_grid = (num_blocks + threads_per_block - 1) / threads_per_block;

    lz4_decompress_kernel<<<num_blocks_grid, threads_per_block, 0, stream>>>(
        src_blocks, dst_blocks, src_sizes, dst_sizes, dst_capacity, num_blocks);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "LZ4 decompress kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

/**
 * Estimate compression ratio
 */
extern "C" float lz4_estimate_ratio(const uint8_t* data, int size) {
    /* Simple heuristic: count unique bytes in sample */
    int sample_size = (size > 4096) ? 4096 : size;
    int unique_count = 0;
    bool seen[256] = {false};

    for (int i = 0; i < sample_size; i++) {
        if (!seen[data[i]]) {
            seen[data[i]] = true;
            unique_count++;
        }
    }

    /* Estimate ratio based on uniqueness */
    float uniqueness = (float)unique_count / 256.0f;

    /* Highly unique data compresses poorly */
    if (uniqueness > 0.8f) {
        return 1.0f;  /* No compression expected */
    }

    /* Rough estimate */
    return 0.5f + (uniqueness * 0.5f);
}
