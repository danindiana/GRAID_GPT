/**
 * GPU RAID SMART Monitoring Tool
 *
 * Monitors drive health using SMART data
 * Provides early warning for drive failures in RAID arrays
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <unistd.h>

struct SmartAttribute {
    int id;
    std::string name;
    int value;
    int worst;
    int threshold;
    std::string raw_value;
    bool failing;
};

struct DriveHealth {
    std::string device;
    std::string model;
    std::string serial;
    std::string firmware;
    bool smart_enabled;
    bool smart_healthy;
    uint64_t power_on_hours;
    uint64_t power_cycles;
    float temperature_c;
    uint64_t total_bytes_written;
    uint64_t total_bytes_read;
    int wear_leveling_count;
    std::vector<SmartAttribute> attributes;
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
};

class SmartMonitor {
public:
    DriveHealth get_drive_health(const std::string& device) {
        DriveHealth health;
        health.device = device;
        health.smart_enabled = false;
        health.smart_healthy = true;

        // Run smartctl to get SMART data
        std::string cmd = "smartctl -a /dev/" + device + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");

        if (!pipe) {
            health.errors.push_back("Failed to execute smartctl");
            return health;
        }

        char buffer[256];
        std::string output;

        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            output += buffer;
        }

        int ret = pclose(pipe);

        if (ret != 0 && ret != 256) {  // 256 = smartctl info, not error
            health.warnings.push_back("smartctl returned non-zero exit code");
        }

        // Parse output
        parse_smart_output(output, health);

        return health;
    }

    void monitor_array(const std::vector<std::string>& devices, bool continuous = false) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          GPU RAID SMART Health Monitor                     ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

        do {
            time_t now = time(nullptr);
            std::cout << "Scan Time: " << ctime(&now);

            std::vector<DriveHealth> health_reports;
            int healthy_count = 0;
            int warning_count = 0;
            int failing_count = 0;

            for (const auto& device : devices) {
                auto health = get_drive_health(device);
                health_reports.push_back(health);

                if (!health.smart_healthy || !health.errors.empty()) {
                    failing_count++;
                } else if (!health.warnings.empty()) {
                    warning_count++;
                } else {
                    healthy_count++;
                }
            }

            // Summary table
            std::cout << "\n╔══════════════╦══════════════════════════╦═══════╦═══════╦════════╗\n";
            std::cout << "║    Device    ║         Model            ║ Temp  ║ Hours ║ Status ║\n";
            std::cout << "╠══════════════╬══════════════════════════╬═══════╬═══════╬════════╣\n";

            for (const auto& health : health_reports) {
                std::string status;
                if (!health.smart_healthy || !health.errors.empty()) {
                    status = "FAIL";
                } else if (!health.warnings.empty()) {
                    status = "WARN";
                } else {
                    status = "OK";
                }

                printf("║ %-12s ║ %-24s ║ %5.1f ║ %5lu ║ %-6s ║\n",
                       health.device.c_str(),
                       health.model.substr(0, 24).c_str(),
                       health.temperature_c,
                       health.power_on_hours,
                       status.c_str());
            }

            std::cout << "╚══════════════╩══════════════════════════╩═══════╩═══════╩════════╝\n\n";

            // Overall status
            std::cout << "Summary:\n";
            std::cout << "  Healthy: " << healthy_count << "\n";
            std::cout << "  Warnings: " << warning_count << "\n";
            std::cout << "  Failing: " << failing_count << "\n\n";

            // Detailed warnings and errors
            for (const auto& health : health_reports) {
                if (!health.warnings.empty() || !health.errors.empty()) {
                    std::cout << "Device /dev/" << health.device << ":\n";

                    for (const auto& error : health.errors) {
                        std::cout << "  ✗ ERROR: " << error << "\n";
                    }

                    for (const auto& warning : health.warnings) {
                        std::cout << "  ⚠ WARNING: " << warning << "\n";
                    }

                    std::cout << "\n";
                }
            }

            // RAID reliability estimate
            estimate_raid_reliability(health_reports);

            if (continuous) {
                std::cout << "Next scan in 300 seconds (5 minutes)...\n";
                std::cout << "Press Ctrl+C to stop.\n\n";
                sleep(300);
            }

        } while (continuous);
    }

    void print_detailed_report(const std::string& device) {
        auto health = get_drive_health(device);

        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║          SMART Detailed Report                             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";

        std::cout << "Device: /dev/" << health.device << "\n";
        std::cout << "Model: " << health.model << "\n";
        std::cout << "Serial: " << health.serial << "\n";
        std::cout << "Firmware: " << health.firmware << "\n";
        std::cout << "SMART Enabled: " << (health.smart_enabled ? "Yes" : "No") << "\n";
        std::cout << "SMART Health: " << (health.smart_healthy ? "PASSED" : "FAILED") << "\n\n";

        std::cout << "Key Metrics:\n";
        std::cout << "  Temperature: " << health.temperature_c << "°C\n";
        std::cout << "  Power On Hours: " << health.power_on_hours << " hours ("
                  << (health.power_on_hours / 24 / 365) << " years)\n";
        std::cout << "  Power Cycles: " << health.power_cycles << "\n";

        if (health.total_bytes_written > 0) {
            std::cout << "  Total Bytes Written: "
                     << (health.total_bytes_written / (1024.0 * 1024.0 * 1024.0 * 1024.0))
                     << " TB\n";
        }

        if (health.wear_leveling_count >= 0) {
            std::cout << "  Wear Leveling: " << health.wear_leveling_count
                     << "% remaining\n";
        }

        std::cout << "\nSMART Attributes:\n";
        std::cout << "╔════╦══════════════════════════════╦═══════╦═══════╦═══════╦═════════════╗\n";
        std::cout << "║ ID ║           Name               ║ Value ║ Worst ║ Thresh║  Raw Value  ║\n";
        std::cout << "╠════╬══════════════════════════════╬═══════╬═══════╬═══════╬═════════════╣\n";

        for (const auto& attr : health.attributes) {
            printf("║%3d ║ %-28s ║  %3d  ║  %3d  ║  %3d  ║ %-11s ║%s\n",
                   attr.id,
                   attr.name.c_str(),
                   attr.value,
                   attr.worst,
                   attr.threshold,
                   attr.raw_value.c_str(),
                   attr.failing ? " ⚠" : "");
        }

        std::cout << "╚════╩══════════════════════════════╩═══════╩═══════╩═══════╩═════════════╝\n\n";

        if (!health.errors.empty()) {
            std::cout << "ERRORS:\n";
            for (const auto& error : health.errors) {
                std::cout << "  ✗ " << error << "\n";
            }
            std::cout << "\n";
        }

        if (!health.warnings.empty()) {
            std::cout << "WARNINGS:\n";
            for (const auto& warning : health.warnings) {
                std::cout << "  ⚠ " << warning << "\n";
            }
            std::cout << "\n";
        }
    }

private:
    void parse_smart_output(const std::string& output, DriveHealth& health) {
        std::istringstream iss(output);
        std::string line;

        while (std::getline(iss, line)) {
            // Parse model
            if (line.find("Device Model:") != std::string::npos ||
                line.find("Model Number:") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    health.model = trim(line.substr(pos + 1));
                }
            }

            // Parse serial
            else if (line.find("Serial Number:") != std::string::npos ||
                    line.find("Serial number:") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    health.serial = trim(line.substr(pos + 1));
                }
            }

            // Parse firmware
            else if (line.find("Firmware Version:") != std::string::npos) {
                size_t pos = line.find(':');
                if (pos != std::string::npos) {
                    health.firmware = trim(line.substr(pos + 1));
                }
            }

            // Parse SMART enabled
            else if (line.find("SMART support is: Enabled") != std::string::npos) {
                health.smart_enabled = true;
            }

            // Parse SMART health
            else if (line.find("SMART overall-health") != std::string::npos) {
                health.smart_healthy = (line.find("PASSED") != std::string::npos);
            }

            // Parse temperature
            else if (line.find("Temperature") != std::string::npos ||
                    line.find("Current Drive Temperature") != std::string::npos) {
                std::istringstream line_stream(line);
                std::string token;
                while (line_stream >> token) {
                    if (std::isdigit(token[0])) {
                        health.temperature_c = std::stof(token);
                        break;
                    }
                }
            }

            // Parse power on hours
            else if (line.find("Power_On_Hours") != std::string::npos ||
                    line.find("Power On Hours") != std::string::npos) {
                std::istringstream line_stream(line);
                std::string token;
                std::vector<std::string> tokens;
                while (line_stream >> token) {
                    tokens.push_back(token);
                }
                if (!tokens.empty()) {
                    health.power_on_hours = std::stoull(tokens.back());
                }
            }

            // Parse SMART attributes (table format)
            else if (std::isdigit(line[0]) && line.find("0x") != std::string::npos) {
                SmartAttribute attr;
                std::istringstream line_stream(line);

                line_stream >> attr.id;
                line_stream >> attr.name;

                // Skip flag
                std::string flag;
                line_stream >> flag;

                line_stream >> attr.value >> attr.worst >> attr.threshold;

                // Skip type
                std::string type;
                line_stream >> type;

                // Skip updated
                std::string updated;
                line_stream >> updated;

                // Skip when_failed
                std::string when_failed;
                line_stream >> when_failed;

                // Rest is raw value
                std::string raw;
                std::getline(line_stream, raw);
                attr.raw_value = trim(raw);

                attr.failing = (attr.value <= attr.threshold);

                health.attributes.push_back(attr);

                // Check for specific concerning attributes
                if (attr.id == 5 && std::stoull(attr.raw_value) > 0) {
                    health.warnings.push_back("Reallocated sectors detected: " + attr.raw_value);
                }
                else if (attr.id == 187 && std::stoull(attr.raw_value) > 0) {
                    health.warnings.push_back("Uncorrectable errors detected: " + attr.raw_value);
                }
                else if (attr.id == 188 && std::stoull(attr.raw_value) > 0) {
                    health.warnings.push_back("Command timeout detected: " + attr.raw_value);
                }
                else if (attr.id == 197 && std::stoull(attr.raw_value) > 0) {
                    health.warnings.push_back("Current pending sectors: " + attr.raw_value);
                }
                else if (attr.id == 198 && std::stoull(attr.raw_value) > 0) {
                    health.errors.push_back("Uncorrectable sector count: " + attr.raw_value);
                }

                // Temperature check
                if (attr.id == 194 && std::stoi(attr.raw_value) > 60) {
                    health.warnings.push_back("High temperature: " + attr.raw_value + "°C");
                }

                // Wear leveling for SSDs
                if (attr.id == 177) {
                    health.wear_leveling_count = attr.value;
                    if (attr.value < 10) {
                        health.errors.push_back("SSD wear level critical: " + std::to_string(attr.value) + "%");
                    } else if (attr.value < 30) {
                        health.warnings.push_back("SSD wear level low: " + std::to_string(attr.value) + "%");
                    }
                }
            }
        }
    }

    void estimate_raid_reliability(const std::vector<DriveHealth>& drives) {
        std::cout << "RAID Array Reliability:\n";

        int critical_count = 0;
        int warning_count = 0;

        for (const auto& drive : drives) {
            if (!drive.errors.empty() || !drive.smart_healthy) {
                critical_count++;
            } else if (!drive.warnings.empty()) {
                warning_count++;
            }
        }

        if (critical_count >= 2) {
            std::cout << "  ✗ CRITICAL: Multiple drives failing! RAID 6 protection may be insufficient!\n";
            std::cout << "    IMMEDIATE ACTION REQUIRED: Replace failing drives and rebuild\n";
        } else if (critical_count == 1) {
            std::cout << "  ⚠ WARNING: One drive failing. RAID 5 arrays are vulnerable!\n";
            std::cout << "    ACTION: Replace failing drive soon\n";
        } else if (warning_count > 0) {
            std::cout << "  ⚠ CAUTION: " << warning_count << " drive(s) showing warnings\n";
            std::cout << "    ACTION: Monitor closely, plan replacements\n";
        } else {
            std::cout << "  ✓ All drives healthy\n";
        }

        std::cout << "\n";
    }

    std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t\n\r");
        if (first == std::string::npos) return "";

        size_t last = str.find_last_not_of(" \t\n\r");
        return str.substr(first, last - first + 1);
    }
};

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n";
    std::cout << "GPU RAID SMART Health Monitoring Tool\n\n";
    std::cout << "Options:\n";
    std::cout << "  -d, --device DEV      Show detailed report for device\n";
    std::cout << "  -a, --array DEV...    Monitor array of devices\n";
    std::cout << "  -c, --continuous      Continuous monitoring (5 min intervals)\n";
    std::cout << "  -h, --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Detailed report for one drive:\n";
    std::cout << "  " << program << " --device nvme0n1\n\n";
    std::cout << "  # Monitor RAID array:\n";
    std::cout << "  " << program << " --array sda sdb sdc sdd sde sdf\n\n";
    std::cout << "  # Continuous monitoring:\n";
    std::cout << "  " << program << " --array sda sdb sdc --continuous\n\n";
    std::cout << "Requirements:\n";
    std::cout << "  - smartmontools package (smartctl command)\n";
    std::cout << "  - Root/sudo access for SMART queries\n\n";
}

int main(int argc, char** argv) {
    if (geteuid() != 0) {
        std::cerr << "Warning: Not running as root. SMART queries may fail.\n";
        std::cerr << "Try: sudo " << argv[0] << " ...\n\n";
    }

    std::string single_device;
    std::vector<std::string> array_devices;
    bool continuous = false;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-d" || arg == "--device") {
            if (i + 1 < argc) {
                single_device = argv[++i];
            }
        } else if (arg == "-a" || arg == "--array") {
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                array_devices.push_back(argv[++i]);
            }
        } else if (arg == "-c" || arg == "--continuous") {
            continuous = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    SmartMonitor monitor;

    if (!single_device.empty()) {
        monitor.print_detailed_report(single_device);
    } else if (!array_devices.empty()) {
        monitor.monitor_array(array_devices, continuous);
    } else {
        std::cout << "No device specified.\n";
        std::cout << "Run with --help for usage information\n\n";
        return 1;
    }

    return 0;
}
