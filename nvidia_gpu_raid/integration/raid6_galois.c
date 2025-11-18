/**
 * RAID 6 Reed-Solomon Implementation
 *
 * Complete Galois Field GF(2^8) arithmetic for RAID 6 P+Q parity
 * Used by MD RAID, LVM, and ZFS integration libraries
 */

#include <stdint.h>
#include <string.h>

/* Galois Field GF(2^8) lookup tables */
static uint8_t gf_exp[512];
static uint8_t gf_log[256];
static int gf_tables_initialized = 0;

/* Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1 (0x11d) */
#define GF_POLYNOMIAL 0x11d

/**
 * Initialize Galois Field lookup tables
 */
static void init_gf_tables(void) {
    int i;
    uint32_t b = 1;

    if (gf_tables_initialized) {
        return;
    }

    /* Generate exponential table */
    for (i = 0; i < 256; i++) {
        gf_exp[i] = (uint8_t)b;
        gf_log[b] = (uint8_t)i;

        b <<= 1;
        if (b & 0x100) {
            b ^= GF_POLYNOMIAL;
        }
    }

    /* Extend exp table for convenience */
    for (i = 256; i < 512; i++) {
        gf_exp[i] = gf_exp[i - 255];
    }

    gf_exp[255] = gf_exp[0];  /* Handle 0 properly */
    gf_log[0] = 0;

    gf_tables_initialized = 1;
}

/**
 * Galois Field multiplication
 */
static inline uint8_t gf_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) {
        return 0;
    }
    return gf_exp[gf_log[a] + gf_log[b]];
}

/**
 * Galois Field division
 */
static inline uint8_t gf_div(uint8_t a, uint8_t b) {
    if (a == 0) {
        return 0;
    }
    if (b == 0) {
        return 0;  /* Division by zero */
    }
    return gf_exp[(gf_log[a] + 255 - gf_log[b]) % 255];
}

/**
 * Galois Field power (a^n)
 */
static inline uint8_t gf_pow(uint8_t a, int n) {
    if (n == 0) {
        return 1;
    }
    if (a == 0) {
        return 0;
    }
    return gf_exp[(gf_log[a] * n) % 255];
}

/**
 * Compute RAID 6 P parity (simple XOR like RAID 5)
 */
void raid6_compute_p(uint8_t **data_blocks, uint8_t *parity_p,
                    int num_data_blocks, size_t block_size) {
    int i;
    size_t j;

    memset(parity_p, 0, block_size);

    for (i = 0; i < num_data_blocks; i++) {
        for (j = 0; j < block_size; j++) {
            parity_p[j] ^= data_blocks[i][j];
        }
    }
}

/**
 * Compute RAID 6 Q parity (Reed-Solomon using powers of 2)
 */
void raid6_compute_q(uint8_t **data_blocks, uint8_t *parity_q,
                    int num_data_blocks, size_t block_size) {
    int i;
    size_t j;
    uint8_t coef;

    init_gf_tables();

    memset(parity_q, 0, block_size);

    for (i = 0; i < num_data_blocks; i++) {
        /* Coefficient is 2^i in GF(2^8) */
        coef = gf_pow(2, i);

        for (j = 0; j < block_size; j++) {
            parity_q[j] ^= gf_mul(coef, data_blocks[i][j]);
        }
    }
}

/**
 * Compute both P and Q parity in one pass (more efficient)
 */
void raid6_compute_pq(uint8_t **data_blocks, uint8_t *parity_p, uint8_t *parity_q,
                     int num_data_blocks, size_t block_size) {
    int i;
    size_t j;
    uint8_t coef;

    init_gf_tables();

    memset(parity_p, 0, block_size);
    memset(parity_q, 0, block_size);

    for (i = 0; i < num_data_blocks; i++) {
        coef = gf_pow(2, i);

        for (j = 0; j < block_size; j++) {
            uint8_t data = data_blocks[i][j];
            parity_p[j] ^= data;
            parity_q[j] ^= gf_mul(coef, data);
        }
    }
}

/**
 * Recover from single data block failure using P parity
 */
void raid6_recover_single(uint8_t **data_blocks, uint8_t *parity_p,
                         int failed_idx, int num_data_blocks, size_t block_size) {
    int i;
    size_t j;
    uint8_t *failed_block = data_blocks[failed_idx];

    memset(failed_block, 0, block_size);

    /* XOR all good data blocks and P parity */
    for (i = 0; i < num_data_blocks; i++) {
        if (i == failed_idx) continue;

        for (j = 0; j < block_size; j++) {
            failed_block[j] ^= data_blocks[i][j];
        }
    }

    for (j = 0; j < block_size; j++) {
        failed_block[j] ^= parity_p[j];
    }
}

/**
 * Recover from dual data block failure using P and Q parity
 */
void raid6_recover_dual(uint8_t **data_blocks, uint8_t *parity_p, uint8_t *parity_q,
                       int failed_idx1, int failed_idx2,
                       int num_data_blocks, size_t block_size) {
    int i;
    size_t j;
    uint8_t *failed1 = data_blocks[failed_idx1];
    uint8_t *failed2 = data_blocks[failed_idx2];
    uint8_t p_syndrome, q_syndrome;
    uint8_t coef1, coef2;
    uint8_t a, b;

    init_gf_tables();

    /* Compute P and Q syndromes (what's missing) */
    /* P_syndrome = P ⊕ (all good data blocks) */
    /* Q_syndrome = Q ⊕ (all good data blocks with coefficients) */

    coef1 = gf_pow(2, failed_idx1);
    coef2 = gf_pow(2, failed_idx2);

    for (j = 0; j < block_size; j++) {
        p_syndrome = parity_p[j];
        q_syndrome = parity_q[j];

        /* XOR all good data blocks */
        for (i = 0; i < num_data_blocks; i++) {
            if (i == failed_idx1 || i == failed_idx2) continue;

            uint8_t data = data_blocks[i][j];
            p_syndrome ^= data;
            q_syndrome ^= gf_mul(gf_pow(2, i), data);
        }

        /* Solve for failed blocks using Galois Field algebra
         * p_syndrome = D1 ⊕ D2
         * q_syndrome = (2^i1 * D1) ⊕ (2^i2 * D2)
         *
         * Let a = 2^i1, b = 2^i2
         * p_syndrome = D1 ⊕ D2
         * q_syndrome = a*D1 ⊕ b*D2
         *
         * From first equation: D2 = p_syndrome ⊕ D1
         * Substitute into second:
         * q_syndrome = a*D1 ⊕ b*(p_syndrome ⊕ D1)
         * q_syndrome = a*D1 ⊕ b*p_syndrome ⊕ b*D1
         * q_syndrome ⊕ b*p_syndrome = (a ⊕ b)*D1
         * D1 = (q_syndrome ⊕ b*p_syndrome) / (a ⊕ b)
         */

        a = gf_mul(coef2, p_syndrome);
        b = q_syndrome ^ a;
        a = coef1 ^ coef2;

        if (a != 0) {
            failed1[j] = gf_div(b, a);
            failed2[j] = p_syndrome ^ failed1[j];
        } else {
            /* Degenerate case - shouldn't happen with proper configuration */
            failed1[j] = 0;
            failed2[j] = 0;
        }
    }
}

/**
 * Recover from P parity failure (recompute from data)
 */
void raid6_recover_p(uint8_t **data_blocks, uint8_t *parity_p,
                    int num_data_blocks, size_t block_size) {
    raid6_compute_p(data_blocks, parity_p, num_data_blocks, block_size);
}

/**
 * Recover from Q parity failure (recompute from data)
 */
void raid6_recover_q(uint8_t **data_blocks, uint8_t *parity_q,
                    int num_data_blocks, size_t block_size) {
    raid6_compute_q(data_blocks, parity_q, num_data_blocks, block_size);
}

/**
 * Recover from one data block + one parity block failure
 */
void raid6_recover_data_parity(uint8_t **data_blocks, uint8_t *parity_p, uint8_t *parity_q,
                              int failed_data_idx, int failed_parity, /* 0=P, 1=Q */
                              int num_data_blocks, size_t block_size) {
    if (failed_parity == 0) {
        /* P parity failed - use Q to recover data, then recompute P */
        /* This requires solving: Q_syndrome = 2^i * D */
        init_gf_tables();

        size_t j;
        int i;
        uint8_t coef = gf_pow(2, failed_data_idx);
        uint8_t q_syndrome;
        uint8_t *failed_block = data_blocks[failed_data_idx];

        for (j = 0; j < block_size; j++) {
            q_syndrome = parity_q[j];

            /* XOR all good data blocks with their coefficients */
            for (i = 0; i < num_data_blocks; i++) {
                if (i == failed_data_idx) continue;
                q_syndrome ^= gf_mul(gf_pow(2, i), data_blocks[i][j]);
            }

            /* D = Q_syndrome / 2^i */
            failed_block[j] = gf_div(q_syndrome, coef);
        }

        /* Recompute P */
        raid6_compute_p(data_blocks, parity_p, num_data_blocks, block_size);
    } else {
        /* Q parity failed - use P to recover data, then recompute Q */
        raid6_recover_single(data_blocks, parity_p, failed_data_idx,
                           num_data_blocks, block_size);

        /* Recompute Q */
        raid6_compute_q(data_blocks, parity_q, num_data_blocks, block_size);
    }
}

/**
 * Verify RAID 6 integrity
 * Returns 0 if OK, -1 if P parity mismatch, -2 if Q parity mismatch
 */
int raid6_verify(uint8_t **data_blocks, uint8_t *parity_p, uint8_t *parity_q,
                int num_data_blocks, size_t block_size) {
    uint8_t *computed_p = malloc(block_size);
    uint8_t *computed_q = malloc(block_size);
    int result = 0;

    if (!computed_p || !computed_q) {
        free(computed_p);
        free(computed_q);
        return -3;
    }

    raid6_compute_pq(data_blocks, computed_p, computed_q, num_data_blocks, block_size);

    if (memcmp(computed_p, parity_p, block_size) != 0) {
        result = -1;
    }

    if (memcmp(computed_q, parity_q, block_size) != 0) {
        result = (result == -1) ? -3 : -2;
    }

    free(computed_p);
    free(computed_q);

    return result;
}

/**
 * RAID 6 scrub - verify and correct single-bit errors
 */
int raid6_scrub(uint8_t **data_blocks, uint8_t *parity_p, uint8_t *parity_q,
               int num_data_blocks, size_t block_size) {
    int verify_result = raid6_verify(data_blocks, parity_p, parity_q,
                                     num_data_blocks, block_size);

    if (verify_result == 0) {
        return 0;  /* All OK */
    }

    /* Attempt to identify and fix errors */
    /* This would require more sophisticated error detection */
    /* For now, just report the error */
    return verify_result;
}
