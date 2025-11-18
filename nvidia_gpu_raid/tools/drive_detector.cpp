/**
 * GPU RAID Drive Detection Tool
 *
 * Automatically detects storage devices and recommends optimal GPU RAID configurations
 * based on drive characteristics (NVMe, SATA SSD, HDD, etc.)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <unistd.h>

namespace fs = std::filesystem;

enum class DriveType {
    NVME_PCIE4,
    NVME_PCIE3,
    SATA_SSD,
    HDD_7200,
    HDD_5400,
    UNKNOWN
};

struct DriveInfo {
    std::string device_name;       // e.g., "nvme0n1", "sda"
    std::string model;             // Drive model string
    DriveType type;                // Detected drive type
    std::string profile_path;      // Recommended profile JSON
    uint64_t size_bytes;           // Total capacity
    bool rotational;               // true for HDDs
    uint32_t queue_depth;          // I/O queue depth
    std::string transport;         // "nvme", "sata", "usb", etc.
    uint32_t pcie_gen;             // PCIe generation (3 or 4) for NVMe
    uint32_t rpm;                  // RPM for HDDs (5400, 7200, etc.)
    std::string scheduler;         // Current I/O scheduler
};

class DriveDetector {
public:
    DriveDetector() : sysfs_block_path_("/sys/block") {}

    std::vector<DriveInfo> detect_all_drives() {
        std::vector<DriveInfo> drives;

        if (!fs::exists(sysfs_block_path_)) {
            std::cerr << "Error: /sys/block not found. Are you running on Linux?\n";
            return drives;
        }

        for (const auto& entry : fs::directory_iterator(sysfs_block_path_)) {
            std::string device_name = entry.path().filename().string();

            // Skip loop devices, ram disks, etc.
            if (device_name.find("loop") == 0 ||
                device_name.find("ram") == 0 ||
                device_name.find("dm-") == 0) {
                continue;
            }

            DriveInfo info;
            if (probe_drive(device_name, info)) {
                drives.push_back(info);
            }
        }

        return drives;
    }

    void print_drive_info(const DriveInfo& drive) const {
        std::cout << "\nDevice: /dev/" << drive.device_name << "\n";
        std::cout << "  Model: " << drive.model << "\n";
        std::cout << "  Type: " << drive_type_to_string(drive.type) << "\n";
        std::cout << "  Size: " << format_size(drive.size_bytes) << "\n";
        std::cout << "  Transport: " << drive.transport << "\n";

        if (drive.rotational) {
            std::cout << "  Rotational: Yes (" << drive.rpm << " RPM)\n";
        } else {
            std::cout << "  Rotational: No (SSD)\n";
        }

        std::cout << "  Queue Depth: " << drive.queue_depth << "\n";

        if (drive.transport == "nvme") {
            std::cout << "  PCIe Gen: " << drive.pcie_gen << "\n";
        }

        std::cout << "  I/O Scheduler: " << drive.scheduler << "\n";
        std::cout << "  Recommended Profile: " << drive.profile_path << "\n";
    }

private:
    std::string sysfs_block_path_;

    bool probe_drive(const std::string& device, DriveInfo& info) {
        info.device_name = device;
        std::string device_path = sysfs_block_path_ + "/" + device;

        // Read size
        info.size_bytes = read_uint64(device_path + "/size") * 512;  // sectors to bytes
        if (info.size_bytes == 0) {
            return false;  // Invalid device
        }

        // Read rotational status
        info.rotational = read_uint64(device_path + "/queue/rotational") != 0;

        // Read queue depth
        info.queue_depth = read_uint64(device_path + "/queue/nr_requests");

        // Read I/O scheduler
        info.scheduler = read_scheduler(device_path + "/queue/scheduler");

        // Read model
        info.model = read_string(device_path + "/device/model");
        trim(info.model);

        // Detect transport type
        if (device.find("nvme") == 0) {
            info.transport = "nvme";
            info.pcie_gen = detect_pcie_gen(device);
            info.type = (info.pcie_gen >= 4) ? DriveType::NVME_PCIE4 : DriveType::NVME_PCIE3;
            info.profile_path = (info.pcie_gen >= 4) ?
                "storage_profiles/nvme_pcie4_profile.json" :
                "storage_profiles/nvme_pcie3_profile.json";
        } else if (device.find("sd") == 0) {
            // SATA or SCSI
            info.transport = detect_transport_sd(device);

            if (info.rotational) {
                // HDD - try to detect RPM
                info.rpm = detect_rpm(info.model);
                info.type = (info.rpm >= 7200) ? DriveType::HDD_7200 : DriveType::HDD_5400;
                info.profile_path = "storage_profiles/hdd_7200rpm_profile.json";
            } else {
                // SSD
                info.type = DriveType::SATA_SSD;
                info.profile_path = "storage_profiles/sata_ssd_profile.json";
            }
        } else {
            info.transport = "unknown";
            info.type = DriveType::UNKNOWN;
            info.profile_path = "none";
        }

        return true;
    }

    uint64_t read_uint64(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return 0;

        uint64_t value;
        file >> value;
        return value;
    }

    std::string read_string(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return "Unknown";

        std::string value;
        std::getline(file, value);
        return value;
    }

    std::string read_scheduler(const std::string& path) {
        std::string content = read_string(path);

        // Scheduler format: "mq-deadline [none] kyber"
        // Extract the one in brackets
        size_t start = content.find('[');
        size_t end = content.find(']');

        if (start != std::string::npos && end != std::string::npos) {
            return content.substr(start + 1, end - start - 1);
        }

        return content;
    }

    uint32_t detect_pcie_gen(const std::string& device) {
        // Try to read PCIe link speed from sysfs
        // NVMe devices: /sys/class/nvme/nvme0/device/current_link_speed
        std::string nvme_base = device.substr(0, device.find('n'));  // "nvme0n1" -> "nvme0"
        std::string speed_path = "/sys/class/nvme/" + nvme_base + "/device/current_link_speed";

        std::string speed = read_string(speed_path);

        // Speed string format: "8.0 GT/s PCIe" (Gen 3) or "16.0 GT/s PCIe" (Gen 4)
        if (speed.find("16.0") != std::string::npos || speed.find("32.0") != std::string::npos) {
            return 4;
        } else if (speed.find("8.0") != std::string::npos) {
            return 3;
        }

        // Default to Gen 3 if unknown
        return 3;
    }

    std::string detect_transport_sd(const std::string& device) {
        std::string device_path = sysfs_block_path_ + "/" + device;

        // Check if it's USB
        std::string subsystem = fs::read_symlink(device_path + "/device/subsystem").filename();
        if (subsystem == "usb") {
            return "usb";
        }

        // Most likely SATA
        return "sata";
    }

    uint32_t detect_rpm(const std::string& model) {
        std::string model_lower = model;
        std::transform(model_lower.begin(), model_lower.end(), model_lower.begin(), ::tolower);

        // Check for common RPM indicators in model string
        if (model_lower.find("5400") != std::string::npos) return 5400;
        if (model_lower.find("7200") != std::string::npos) return 7200;
        if (model_lower.find("10000") != std::string::npos) return 10000;
        if (model_lower.find("10k") != std::string::npos) return 10000;
        if (model_lower.find("15000") != std::string::npos) return 15000;
        if (model_lower.find("15k") != std::string::npos) return 15000;

        // Check for enterprise drive indicators (usually 7200 RPM)
        if (model_lower.find("enterprise") != std::string::npos ||
            model_lower.find("ironwolf") != std::string::npos ||
            model_lower.find("red") != std::string::npos ||
            model_lower.find("ultrastar") != std::string::npos) {
            return 7200;
        }

        // Default assumption for HDDs
        return 7200;
    }

    std::string drive_type_to_string(DriveType type) const {
        switch (type) {
            case DriveType::NVME_PCIE4: return "NVMe PCIe 4.0";
            case DriveType::NVME_PCIE3: return "NVMe PCIe 3.0";
            case DriveType::SATA_SSD: return "SATA SSD";
            case DriveType::HDD_7200: return "HDD 7200 RPM";
            case DriveType::HDD_5400: return "HDD 5400 RPM";
            default: return "Unknown";
        }
    }

    std::string format_size(uint64_t bytes) const {
        const char* units[] = {"B", "KB", "MB", "GB", "TB"};
        int unit_idx = 0;
        double size = bytes;

        while (size >= 1024 && unit_idx < 4) {
            size /= 1024;
            unit_idx++;
        }

        char buffer[64];
        snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit_idx]);
        return std::string(buffer);
    }

    void trim(std::string& s) {
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
        }));
        s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
            return !std::isspace(ch);
        }).base(), s.end());
    }
};

class ConfigRecommender {
public:
    void recommend_raid_config(const std::vector<DriveInfo>& drives) {
        if (drives.empty()) {
            std::cout << "\nNo drives detected.\n";
            return;
        }

        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          GPU RAID Configuration Recommendations            ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        // Group drives by type
        std::map<DriveType, std::vector<DriveInfo>> grouped;
        for (const auto& drive : drives) {
            grouped[drive.type].push_back(drive);
        }

        // Recommend configuration for each drive type
        for (const auto& [type, type_drives] : grouped) {
            if (type == DriveType::UNKNOWN) continue;

            std::cout << "\n" << drive_type_to_string(type) << " (" << type_drives.size() << " drives):\n";

            // Recommend RAID level
            int raid_level = 5;
            if (type == DriveType::HDD_7200 || type == DriveType::HDD_5400) {
                raid_level = 6;  // Dual parity for HDDs (higher failure rate)
            } else if (type_drives.size() >= 6) {
                raid_level = 6;  // Recommend RAID 6 for larger arrays
            }

            std::cout << "  Recommended RAID Level: RAID " << raid_level << "\n";
            std::cout << "  Recommended Drives: " << get_optimal_drive_count(type, type_drives.size()) << "\n";
            std::cout << "  Profile: " << type_drives[0].profile_path << "\n";

            // Calculate expected performance
            auto perf = estimate_performance(type, type_drives.size(), raid_level);
            std::cout << "  Expected Throughput: " << perf.first << " GB/s (encode)\n";
            std::cout << "  Expected Rebuild Time (8TB): " << perf.second << " hours\n";

            // Specific recommendations
            std::cout << "  Notes:\n";
            if (type == DriveType::NVME_PCIE4) {
                std::cout << "    - Use large stripe sizes (256KB+) for maximum throughput\n";
                std::cout << "    - Enable async GPU operations for best performance\n";
                std::cout << "    - Monitor drive temperature (<70°C)\n";
            } else if (type == DriveType::NVME_PCIE3) {
                std::cout << "    - Use medium stripe sizes (128KB) for balanced performance\n";
                std::cout << "    - Good for most workloads\n";
            } else if (type == DriveType::SATA_SSD) {
                std::cout << "    - SATA bandwidth-limited (600MB/s per drive)\n";
                std::cout << "    - Use more drives to aggregate bandwidth\n";
                std::cout << "    - Enable TRIM/discard for longevity\n";
            } else if (type == DriveType::HDD_7200 || type == DriveType::HDD_5400) {
                std::cout << "    - Use RAID 6 for reliability (HDDs have higher failure rates)\n";
                std::cout << "    - Large stripe sizes (1MB+) minimize seeks\n";
                std::cout << "    - GPU helps with parity, but HDD mechanics are bottleneck\n";
                std::cout << "    - Best for sequential workloads (backups, archives)\n";
            }
        }

        // Check for mixed drive types
        if (grouped.size() > 1) {
            std::cout << "\n⚠ WARNING: Mixed drive types detected!\n";
            std::cout << "  Recommendation: Create separate RAID arrays for each drive type\n";
            std::cout << "  Mixing drive types will limit performance to slowest drive\n";
        }

        // Minimum drive recommendations
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   Quick Start Command                      ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        // Generate example command for first group
        auto first_group = grouped.begin();
        if (first_group != grouped.end() && first_group->first != DriveType::UNKNOWN) {
            const auto& example_drives = first_group->second;
            int raid_level = (first_group->first == DriveType::HDD_7200 ||
                            first_group->first == DriveType::HDD_5400) ? 6 : 5;

            std::cout << "\n# Create RAID " << raid_level << " array:\n";
            std::cout << "./gpu_raid_cli create --raid=" << raid_level
                     << " --drives=" << std::min(6, (int)example_drives.size());

            for (size_t i = 0; i < std::min((size_t)6, example_drives.size()); i++) {
                std::cout << " /dev/" << example_drives[i].device_name;
            }

            std::cout << " \\\n                       --profile=" << example_drives[0].profile_path << "\n";
        }
    }

private:
    std::string drive_type_to_string(DriveType type) const {
        switch (type) {
            case DriveType::NVME_PCIE4: return "NVMe PCIe 4.0";
            case DriveType::NVME_PCIE3: return "NVMe PCIe 3.0";
            case DriveType::SATA_SSD: return "SATA SSD";
            case DriveType::HDD_7200: return "HDD 7200 RPM";
            case DriveType::HDD_5400: return "HDD 5400 RPM";
            default: return "Unknown";
        }
    }

    int get_optimal_drive_count(DriveType type, int available) const {
        switch (type) {
            case DriveType::NVME_PCIE4: return std::min(6, available);
            case DriveType::NVME_PCIE3: return std::min(4, available);
            case DriveType::SATA_SSD: return std::min(6, available);
            case DriveType::HDD_7200:
            case DriveType::HDD_5400: return std::min(8, available);
            default: return 4;
        }
    }

    std::pair<std::string, double> estimate_performance(DriveType type, int num_drives, int raid_level) const {
        // Returns (throughput string, rebuild time in hours for 8TB)
        switch (type) {
            case DriveType::NVME_PCIE4:
                if (raid_level == 5) {
                    return {"20-28", 0.75};
                } else {
                    return {"15-22", 1.0};
                }
            case DriveType::NVME_PCIE3:
                if (raid_level == 5) {
                    return {"12-18", 1.5};
                } else {
                    return {"10-14", 2.0};
                }
            case DriveType::SATA_SSD:
                if (raid_level == 5) {
                    return {"2.5-3.2", 4.5};
                } else {
                    return {"2.0-2.8", 5.5};
                }
            case DriveType::HDD_7200:
                if (raid_level == 5) {
                    return {"1.0-1.4", 12};
                } else {
                    return {"0.8-1.2", 14};
                }
            case DriveType::HDD_5400:
                if (raid_level == 5) {
                    return {"0.6-0.9", 18};
                } else {
                    return {"0.5-0.8", 22};
                }
            default:
                return {"unknown", 0};
        }
    }
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n";
    std::cout << "GPU RAID Drive Detection Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  -l, --list        List all detected drives\n";
    std::cout << "  -r, --recommend   Recommend RAID configuration\n";
    std::cout << "  -d, --device DEV  Show details for specific device\n";
    std::cout << "  -h, --help        Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program << " --list                # List all drives\n";
    std::cout << "  " << program << " --recommend           # Get RAID recommendations\n";
    std::cout << "  " << program << " --device nvme0n1      # Show details for nvme0n1\n\n";
}

int main(int argc, char** argv) {
    if (geteuid() != 0) {
        std::cout << "Note: Running without root. Some drive details may be unavailable.\n\n";
    }

    bool list_drives = false;
    bool recommend = false;
    std::string specific_device;

    // Simple argument parsing
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-l" || arg == "--list") {
            list_drives = true;
        } else if (arg == "-r" || arg == "--recommend") {
            recommend = true;
        } else if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                specific_device = argv[++i];
            }
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    // Default to list and recommend if no args
    if (!list_drives && !recommend && specific_device.empty()) {
        list_drives = true;
        recommend = true;
    }

    DriveDetector detector;
    auto drives = detector.detect_all_drives();

    if (drives.empty()) {
        std::cout << "No storage drives detected.\n";
        return 1;
    }

    if (list_drives) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║              Detected Storage Drives                       ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";

        for (const auto& drive : drives) {
            detector.print_drive_info(drive);
        }

        std::cout << "\nTotal drives detected: " << drives.size() << "\n";
    }

    if (!specific_device.empty()) {
        auto it = std::find_if(drives.begin(), drives.end(),
            [&specific_device](const DriveInfo& d) { return d.device_name == specific_device; });

        if (it != drives.end()) {
            detector.print_drive_info(*it);
        } else {
            std::cout << "Device " << specific_device << " not found.\n";
        }
    }

    if (recommend) {
        ConfigRecommender recommender;
        recommender.recommend_raid_config(drives);
    }

    std::cout << "\n";
    return 0;
}
