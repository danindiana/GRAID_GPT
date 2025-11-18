/**
 * GPU RAID Daemon
 *
 * Userspace daemon that handles GPU RAID operations requested by kernel module
 * Communicates via netlink, performs actual GPU computation
 */

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <map>
#include <atomic>
#include <csignal>
#include <cstring>
#include <unistd.h>
#include <syslog.h>
#include <sys/socket.h>
#include <linux/netlink.h>

#include "../include/gpu_raid.h"
#include "../kernel_module/gpu_raid_netlink.h"

class GPURaidDaemon {
public:
    GPURaidDaemon(int gpu_device_id = 0)
        : gpu_device_id_(gpu_device_id)
        , running_(true)
        , netlink_fd_(-1)
        , next_request_id_(1)
    {
    }

    ~GPURaidDaemon() {
        stop();
    }

    bool initialize() {
        openlog("gpu_raid_daemon", LOG_PID | LOG_NDELAY, LOG_DAEMON);
        syslog(LOG_INFO, "Initializing GPU RAID daemon (GPU %d)", gpu_device_id_);

        // Initialize GPU RAID library
        gpu_raid_config_t config = {
            .raid_level = GPU_RAID_LEVEL_5,  // Default, will be overridden per request
            .num_data_drives = 4,
            .stripe_size_kb = 256,
            .gpu_device_id = gpu_device_id_,
            .device_type = GPU_RAID_DEVICE_AUTO,
            .memory_pool_size_mb = 4096,
            .num_streams = 4,
            .enable_profiling = false
        };

        gpu_raid_error_t err = gpu_raid_init(&config, &gpu_handle_);
        if (err != GPU_RAID_SUCCESS) {
            syslog(LOG_ERR, "Failed to initialize GPU RAID: %s",
                   gpu_raid_get_error_string(err));
            return false;
        }

        // Initialize netlink socket
        if (!init_netlink()) {
            syslog(LOG_ERR, "Failed to initialize netlink");
            gpu_raid_destroy(gpu_handle_);
            return false;
        }

        syslog(LOG_INFO, "GPU RAID daemon initialized successfully");
        return true;
    }

    void run() {
        // Start worker threads
        std::thread netlink_thread(&GPURaidDaemon::netlink_worker, this);
        std::thread gpu_thread(&GPURaidDaemon::gpu_worker, this);

        // Register with kernel
        send_register_message();

        // Wait for shutdown
        netlink_thread.join();
        gpu_thread.join();

        syslog(LOG_INFO, "GPU RAID daemon stopped");
        closelog();
    }

    void stop() {
        running_ = false;
        request_cv_.notify_all();

        if (netlink_fd_ >= 0) {
            close(netlink_fd_);
            netlink_fd_ = -1;
        }

        if (gpu_handle_) {
            gpu_raid_destroy(gpu_handle_);
            gpu_handle_ = nullptr;
        }
    }

private:
    int gpu_device_id_;
    std::atomic<bool> running_;
    int netlink_fd_;
    gpu_raid_handle_t gpu_handle_;

    // Request queue
    std::queue<gpu_raid_request_desc> request_queue_;
    std::mutex queue_mutex_;
    std::condition_variable request_cv_;

    // DMA buffers
    std::map<uint64_t, void*> dma_buffers_;
    std::mutex dma_mutex_;

    std::atomic<uint64_t> next_request_id_;

    // Statistics
    std::atomic<uint64_t> requests_processed_;
    std::atomic<uint64_t> requests_failed_;

    bool init_netlink() {
        struct sockaddr_nl src_addr;

        netlink_fd_ = socket(AF_NETLINK, SOCK_RAW, NETLINK_GPU_RAID);
        if (netlink_fd_ < 0) {
            syslog(LOG_ERR, "Failed to create netlink socket: %s", strerror(errno));
            return false;
        }

        memset(&src_addr, 0, sizeof(src_addr));
        src_addr.nl_family = AF_NETLINK;
        src_addr.nl_pid = getpid();

        if (bind(netlink_fd_, (struct sockaddr*)&src_addr, sizeof(src_addr)) < 0) {
            syslog(LOG_ERR, "Failed to bind netlink socket: %s", strerror(errno));
            close(netlink_fd_);
            netlink_fd_ = -1;
            return false;
        }

        syslog(LOG_INFO, "Netlink socket initialized (fd=%d, pid=%d)",
               netlink_fd_, getpid());
        return true;
    }

    void send_register_message() {
        syslog(LOG_INFO, "Registering with kernel module");
        // TODO: Send netlink registration message to kernel
    }

    void netlink_worker() {
        struct sockaddr_nl dest_addr;
        struct nlmsghdr *nlh;
        struct iovec iov;
        struct msghdr msg;
        char buffer[8192];

        syslog(LOG_INFO, "Netlink worker started");

        while (running_) {
            memset(&dest_addr, 0, sizeof(dest_addr));
            memset(&msg, 0, sizeof(msg));

            nlh = (struct nlmsghdr *)buffer;
            iov.iov_base = (void *)nlh;
            iov.iov_len = sizeof(buffer);

            msg.msg_name = (void *)&dest_addr;
            msg.msg_namelen = sizeof(dest_addr);
            msg.msg_iov = &iov;
            msg.msg_iovlen = 1;

            // Receive message from kernel
            ssize_t len = recvmsg(netlink_fd_, &msg, 0);
            if (len < 0) {
                if (errno == EINTR || errno == EAGAIN) {
                    continue;
                }
                syslog(LOG_ERR, "Netlink receive error: %s", strerror(errno));
                break;
            }

            if (len == 0) {
                continue;
            }

            // Process netlink message
            process_netlink_message(nlh, len);
        }

        syslog(LOG_INFO, "Netlink worker stopped");
    }

    void process_netlink_message(struct nlmsghdr *nlh, size_t len) {
        // Parse netlink message and extract request
        gpu_raid_request_desc *req = (gpu_raid_request_desc *)NLMSG_DATA(nlh);

        syslog(LOG_DEBUG, "Received request: id=%lu, raid=%d, op=%d, blocks=%d",
               req->request_id, req->raid_level, req->operation, req->num_data_blocks);

        // Queue request for GPU processing
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            request_queue_.push(*req);
        }
        request_cv_.notify_one();
    }

    void gpu_worker() {
        syslog(LOG_INFO, "GPU worker started");

        while (running_) {
            gpu_raid_request_desc req;

            // Wait for request
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                request_cv_.wait(lock, [this] {
                    return !request_queue_.empty() || !running_;
                });

                if (!running_) {
                    break;
                }

                req = request_queue_.front();
                request_queue_.pop();
            }

            // Process request on GPU
            process_gpu_request(req);
        }

        syslog(LOG_INFO, "GPU worker stopped (processed=%lu, failed=%lu)",
               requests_processed_.load(), requests_failed_.load());
    }

    void process_gpu_request(const gpu_raid_request_desc& req) {
        auto start = std::chrono::high_resolution_clock::now();

        gpu_raid_response_desc response;
        response.request_id = req.request_id;
        response.status = GPU_RAID_STATUS_SUCCESS;
        response.error_code = 0;

        // Get DMA buffer
        void *dma_buffer = nullptr;
        {
            std::lock_guard<std::mutex> lock(dma_mutex_);
            auto it = dma_buffers_.find(req.dma_handle);
            if (it != dma_buffers_.end()) {
                dma_buffer = it->second;
            }
        }

        if (!dma_buffer) {
            syslog(LOG_ERR, "Invalid DMA handle: %lu", req.dma_handle);
            response.status = GPU_RAID_STATUS_ERROR;
            response.error_code = -EINVAL;
            send_response(response);
            requests_failed_++;
            return;
        }

        // Prepare block pointers
        std::vector<const uint8_t*> data_blocks(req.num_data_blocks);
        std::vector<uint8_t*> parity_blocks(req.num_parity_blocks);

        uint8_t *base = (uint8_t*)dma_buffer;
        for (uint32_t i = 0; i < req.num_data_blocks; i++) {
            data_blocks[i] = base + (i * req.block_size);
        }
        for (uint32_t i = 0; i < req.num_parity_blocks; i++) {
            parity_blocks[i] = base + ((req.num_data_blocks + i) * req.block_size);
        }

        // Perform GPU operation
        gpu_raid_error_t err = GPU_RAID_SUCCESS;

        if (req.operation == 0) {
            // Encode (compute parity)
            err = gpu_raid_encode(gpu_handle_, data_blocks.data(), parity_blocks.data(),
                                 req.num_data_blocks, req.block_size);
        } else {
            // Decode (reconstruct)
            std::vector<uint8_t*> output_blocks(req.num_failed);
            for (uint32_t i = 0; i < req.num_failed; i++) {
                output_blocks[i] = base + (req.failed_indices[i] * req.block_size);
            }

            std::vector<const uint8_t*> parity_const(req.num_parity_blocks);
            for (uint32_t i = 0; i < req.num_parity_blocks; i++) {
                parity_const[i] = parity_blocks[i];
            }

            err = gpu_raid_reconstruct(gpu_handle_, data_blocks.data(), parity_const.data(),
                                      req.failed_indices, req.num_failed,
                                      output_blocks.data(), req.num_data_blocks, req.block_size);
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

        if (err != GPU_RAID_SUCCESS) {
            syslog(LOG_ERR, "GPU operation failed: %s", gpu_raid_get_error_string(err));
            response.status = GPU_RAID_STATUS_ERROR;
            response.error_code = -EIO;
            requests_failed_++;
        } else {
            response.completion_time_ns = duration.count();
            requests_processed_++;
        }

        // Send response back to kernel
        send_response(response);
    }

    void send_response(const gpu_raid_response_desc& response) {
        struct sockaddr_nl dest_addr;
        struct nlmsghdr *nlh;
        struct iovec iov;
        struct msghdr msg;
        char buffer[1024];

        memset(&dest_addr, 0, sizeof(dest_addr));
        dest_addr.nl_family = AF_NETLINK;
        dest_addr.nl_pid = 0;  // Kernel

        nlh = (struct nlmsghdr *)buffer;
        nlh->nlmsg_len = NLMSG_SPACE(sizeof(response));
        nlh->nlmsg_pid = getpid();
        nlh->nlmsg_flags = 0;

        memcpy(NLMSG_DATA(nlh), &response, sizeof(response));

        iov.iov_base = (void *)nlh;
        iov.iov_len = nlh->nlmsg_len;

        memset(&msg, 0, sizeof(msg));
        msg.msg_name = (void *)&dest_addr;
        msg.msg_namelen = sizeof(dest_addr);
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;

        if (sendmsg(netlink_fd_, &msg, 0) < 0) {
            syslog(LOG_ERR, "Failed to send response: %s", strerror(errno));
        }
    }

    // DMA buffer management
public:
    uint64_t register_dma_buffer(void *buffer, size_t size) {
        std::lock_guard<std::mutex> lock(dma_mutex_);
        uint64_t handle = next_request_id_++;
        dma_buffers_[handle] = buffer;
        return handle;
    }

    void unregister_dma_buffer(uint64_t handle) {
        std::lock_guard<std::mutex> lock(dma_mutex_);
        dma_buffers_.erase(handle);
    }
};

static GPURaidDaemon *g_daemon = nullptr;

static void signal_handler(int sig) {
    if (g_daemon) {
        syslog(LOG_INFO, "Received signal %d, shutting down", sig);
        g_daemon->stop();
    }
}

int main(int argc, char **argv) {
    int gpu_device_id = 0;
    bool daemonize = false;

    // Parse arguments
    int opt;
    while ((opt = getopt(argc, argv, "g:dh")) != -1) {
        switch (opt) {
        case 'g':
            gpu_device_id = atoi(optarg);
            break;
        case 'd':
            daemonize = true;
            break;
        case 'h':
            printf("Usage: %s [-g gpu_id] [-d] [-h]\n", argv[0]);
            printf("  -g GPU_ID   GPU device ID (default: 0)\n");
            printf("  -d          Run as daemon\n");
            printf("  -h          Show this help\n");
            return 0;
        default:
            return 1;
        }
    }

    // Daemonize if requested
    if (daemonize) {
        pid_t pid = fork();
        if (pid < 0) {
            perror("fork");
            return 1;
        }
        if (pid > 0) {
            // Parent exits
            return 0;
        }

        // Child continues as daemon
        setsid();
        chdir("/");
        close(STDIN_FILENO);
        close(STDOUT_FILENO);
        close(STDERR_FILENO);
    }

    // Create daemon instance
    g_daemon = new GPURaidDaemon(gpu_device_id);

    // Setup signal handlers
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    // Initialize and run
    if (!g_daemon->initialize()) {
        delete g_daemon;
        return 1;
    }

    g_daemon->run();

    delete g_daemon;
    g_daemon = nullptr;

    return 0;
}
