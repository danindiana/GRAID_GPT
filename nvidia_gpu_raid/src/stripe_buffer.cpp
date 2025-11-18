/**
 * Stripe Buffer Manager
 *
 * Manages stripe-level buffering for efficient RAID operations
 */

#include "../include/raid_types.h"
#include <stdlib.h>
#include <string.h>

#define MAX_STRIPES_IN_FLIGHT 16

typedef struct {
    stripe_info_t stripes[MAX_STRIPES_IN_FLIGHT];
    int num_active;
    uint32_t next_stripe_id;
} stripe_buffer_manager_t;

static stripe_buffer_manager_t g_stripe_mgr = {0};

/**
 * Initialize stripe buffer manager
 */
void stripe_buffer_init() {
    memset(&g_stripe_mgr, 0, sizeof(stripe_buffer_manager_t));
}

/**
 * Allocate a new stripe
 */
stripe_info_t* stripe_buffer_alloc(uint32_t num_blocks, size_t block_size) {
    if (g_stripe_mgr.num_active >= MAX_STRIPES_IN_FLIGHT) {
        return NULL;  // No free stripes
    }

    // Find free slot
    for (int i = 0; i < MAX_STRIPES_IN_FLIGHT; i++) {
        if (g_stripe_mgr.stripes[i].stripe_id == 0) {
            stripe_info_t* stripe = &g_stripe_mgr.stripes[i];
            stripe->stripe_id = ++g_stripe_mgr.next_stripe_id;
            stripe->num_blocks = num_blocks;
            stripe->block_size = block_size;

            stripe->data_blocks_host = (uint8_t**)calloc(num_blocks, sizeof(uint8_t*));
            stripe->data_blocks_device = (uint8_t**)calloc(num_blocks, sizeof(uint8_t*));

            g_stripe_mgr.num_active++;
            return stripe;
        }
    }

    return NULL;
}

/**
 * Free a stripe
 */
void stripe_buffer_free(stripe_info_t* stripe) {
    if (!stripe) return;

    if (stripe->data_blocks_host) {
        free(stripe->data_blocks_host);
    }
    if (stripe->data_blocks_device) {
        free(stripe->data_blocks_device);
    }

    memset(stripe, 0, sizeof(stripe_info_t));
    g_stripe_mgr.num_active--;
}

/**
 * Get stripe by ID
 */
stripe_info_t* stripe_buffer_get(uint32_t stripe_id) {
    for (int i = 0; i < MAX_STRIPES_IN_FLIGHT; i++) {
        if (g_stripe_mgr.stripes[i].stripe_id == stripe_id) {
            return &g_stripe_mgr.stripes[i];
        }
    }
    return NULL;
}

/**
 * Get number of active stripes
 */
int stripe_buffer_get_active_count() {
    return g_stripe_mgr.num_active;
}

/**
 * Cleanup all stripes
 */
void stripe_buffer_cleanup() {
    for (int i = 0; i < MAX_STRIPES_IN_FLIGHT; i++) {
        if (g_stripe_mgr.stripes[i].stripe_id != 0) {
            stripe_buffer_free(&g_stripe_mgr.stripes[i]);
        }
    }
}
