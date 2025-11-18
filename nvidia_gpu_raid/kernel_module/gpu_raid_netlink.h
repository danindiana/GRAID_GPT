/*
 * GPU RAID Netlink Protocol
 *
 * Defines netlink protocol for communication between kernel module
 * and userspace GPU daemon
 */

#ifndef GPU_RAID_NETLINK_H
#define GPU_RAID_NETLINK_H

#include <linux/types.h>

/* Netlink protocol number (use generic netlink) */
#define NETLINK_GPU_RAID 31

/* Message types */
enum gpu_raid_cmd {
    GPU_RAID_CMD_UNSPEC = 0,
    GPU_RAID_CMD_ENCODE,        /* Request parity encoding */
    GPU_RAID_CMD_DECODE,        /* Request data reconstruction */
    GPU_RAID_CMD_REGISTER,      /* Register GPU daemon */
    GPU_RAID_CMD_UNREGISTER,    /* Unregister GPU daemon */
    GPU_RAID_CMD_PING,          /* Keepalive ping */
    GPU_RAID_CMD_STATUS,        /* Query GPU status */
    __GPU_RAID_CMD_MAX,
};

#define GPU_RAID_CMD_MAX (__GPU_RAID_CMD_MAX - 1)

/* Attribute types */
enum gpu_raid_attr {
    GPU_RAID_ATTR_UNSPEC = 0,
    GPU_RAID_ATTR_RAID_LEVEL,       /* u8: RAID level (5 or 6) */
    GPU_RAID_ATTR_NUM_DATA_BLOCKS,  /* u32: Number of data blocks */
    GPU_RAID_ATTR_NUM_PARITY_BLOCKS,/* u32: Number of parity blocks */
    GPU_RAID_ATTR_BLOCK_SIZE,       /* u64: Block size in bytes */
    GPU_RAID_ATTR_DMA_HANDLE,       /* u64: DMA buffer handle */
    GPU_RAID_ATTR_REQUEST_ID,       /* u64: Unique request ID */
    GPU_RAID_ATTR_STATUS,           /* u32: Operation status */
    GPU_RAID_ATTR_ERROR_CODE,       /* s32: Error code if failed */
    GPU_RAID_ATTR_GPU_DEVICE_ID,    /* u32: GPU device ID */
    GPU_RAID_ATTR_FAILED_INDICES,   /* Binary: Array of failed block indices */
    __GPU_RAID_ATTR_MAX,
};

#define GPU_RAID_ATTR_MAX (__GPU_RAID_ATTR_MAX - 1)

/* Operation status codes */
enum gpu_raid_status {
    GPU_RAID_STATUS_SUCCESS = 0,
    GPU_RAID_STATUS_PENDING,
    GPU_RAID_STATUS_ERROR,
    GPU_RAID_STATUS_GPU_UNAVAILABLE,
    GPU_RAID_STATUS_TIMEOUT,
};

/* DMA buffer descriptor */
struct gpu_raid_dma_desc {
    __u64 handle;           /* Unique handle for this DMA buffer */
    __u64 bus_addr;         /* DMA bus address */
    __u64 size;             /* Buffer size */
    __u32 num_blocks;       /* Number of blocks */
    __u32 block_size;       /* Size of each block */
};

/* Request descriptor for netlink communication */
struct gpu_raid_request_desc {
    __u64 request_id;       /* Unique request identifier */
    __u8 raid_level;        /* 5 or 6 */
    __u8 operation;         /* 0=encode, 1=decode */
    __u16 reserved;
    __u32 num_data_blocks;
    __u32 num_parity_blocks;
    __u64 block_size;
    __u64 dma_handle;       /* Handle to DMA buffer */
    __u32 failed_indices[16]; /* For decode: which blocks failed */
    __u32 num_failed;
};

/* Response descriptor */
struct gpu_raid_response_desc {
    __u64 request_id;
    __u32 status;           /* gpu_raid_status */
    __s32 error_code;       /* errno if failed */
    __u64 completion_time_ns; /* Time taken on GPU */
};

#endif /* GPU_RAID_NETLINK_H */
