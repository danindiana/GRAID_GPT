# GPU RAID Tools

This directory contains command-line tools for managing, monitoring, and benchmarking GPU-accelerated RAID arrays.

## Tools Overview

| Tool | Purpose | Requires Root | Requires GPU |
|------|---------|---------------|--------------|
| `gpu_raid_cli` | Main RAID management tool | No (Yes for create) | Yes |
| `drive_detector` | Auto-detect drives and recommend configs | No (Yes for full info) | No |
| `drive_benchmark` | Benchmark GPU RAID performance | No | Yes |
| `smart_monitor` | Monitor drive health via SMART | Yes | No |

---

## gpu_raid_cli

Main command-line interface for GPU RAID operations.

### Features
- Create RAID 5/6 arrays
- Encode and verify parity
- Rebuild failed drives
- Benchmark performance
- Query GPU and RAID statistics

### Usage

```bash
# Show GPU information
./gpu_raid_cli info

# Encode data to RAID 5
./gpu_raid_cli encode \
  --raid=5 \
  --data data1.bin data2.bin data3.bin \
  --parity parity.bin

# Rebuild failed drive
./gpu_raid_cli rebuild \
  --raid=5 \
  --data data1.bin data2.bin data3.bin \
  --parity parity.bin \
  --failed 1 \
  --output recovered.bin

# Verify parity
./gpu_raid_cli verify \
  --raid=5 \
  --data data1.bin data2.bin data3.bin \
  --parity parity.bin

# Benchmark
./gpu_raid_cli bench \
  --raid=5 \
  --drives=6 \
  --block-size=1048576
```

### Options

```
-r, --raid LEVEL        RAID level (5 or 6)
-d, --data FILE...      Data block files
-p, --parity FILE...    Parity block files
-f, --failed INDEX      Failed drive index for rebuild
-o, --output FILE       Output file for rebuild
-b, --block-size SIZE   Block size in bytes
-g, --gpu ID            GPU device ID (default: 0)
-v, --verbose           Verbose output
-h, --help              Show help
```

### Examples

**RAID 5 with 4 drives:**
```bash
# Create test data
dd if=/dev/urandom of=data0.bin bs=1M count=100
dd if=/dev/urandom of=data1.bin bs=1M count=100
dd if=/dev/urandom of=data2.bin bs=1M count=100

# Encode parity
./gpu_raid_cli encode --raid=5 \
  --data data0.bin data1.bin data2.bin \
  --parity parity.bin

# Simulate drive failure
rm data1.bin

# Rebuild
./gpu_raid_cli rebuild --raid=5 \
  --data data0.bin NULL data2.bin \
  --parity parity.bin \
  --failed 1 \
  --output data1_recovered.bin

# Verify
./gpu_raid_cli verify --raid=5 \
  --data data0.bin data1_recovered.bin data2.bin \
  --parity parity.bin
```

**RAID 6 with dual failure:**
```bash
# Encode with 2 parities
./gpu_raid_cli encode --raid=6 \
  --data data0.bin data1.bin data2.bin data3.bin \
  --parity parity_p.bin parity_q.bin

# Simulate dual failure
rm data1.bin data3.bin

# Rebuild both
./gpu_raid_cli rebuild --raid=6 \
  --data data0.bin NULL data2.bin NULL \
  --parity parity_p.bin parity_q.bin \
  --failed 1 3 \
  --output data1_rec.bin data3_rec.bin
```

---

## drive_detector

Automatically detects storage devices and recommends optimal GPU RAID configurations.

### Features
- Scans Linux `/sys/block/` for drives
- Identifies drive type (NVMe PCIe 4.0/3.0, SATA SSD, HDD)
- Reads drive properties (size, model, queue depth, etc.)
- Recommends storage profile
- Suggests RAID level and configuration
- Warns about mixed drive types

### Usage

```bash
# List all detected drives
sudo ./drive_detector --list

# Get RAID recommendations
sudo ./drive_detector --recommend

# Show details for specific device
sudo ./drive_detector --device nvme0n1

# List and recommend (default)
sudo ./drive_detector
```

### Example Output

```
╔════════════════════════════════════════════════════════════╗
║              Detected Storage Drives                       ║
╚════════════════════════════════════════════════════════════╝

Device: /dev/nvme0n1
  Model: Samsung SSD 980 PRO 1TB
  Type: NVMe PCIe 4.0
  Size: 953.87 GB
  Transport: nvme
  Rotational: No (SSD)
  Queue Depth: 1024
  PCIe Gen: 4
  I/O Scheduler: none
  Recommended Profile: storage_profiles/nvme_pcie4_profile.json

Device: /dev/nvme1n1
  Model: Samsung SSD 980 PRO 1TB
  Type: NVMe PCIe 4.0
  Size: 953.87 GB
  Transport: nvme
  Rotational: No (SSD)
  Queue Depth: 1024
  PCIe Gen: 4
  I/O Scheduler: none
  Recommended Profile: storage_profiles/nvme_pcie4_profile.json

[... more drives ...]

╔════════════════════════════════════════════════════════════╗
║          GPU RAID Configuration Recommendations            ║
╚════════════════════════════════════════════════════════════╝

NVMe PCIe 4.0 (6 drives):
  Recommended RAID Level: RAID 5
  Recommended Drives: 6
  Profile: storage_profiles/nvme_pcie4_profile.json
  Expected Throughput: 20-28 GB/s (encode)
  Expected Rebuild Time (8TB): 0.75 hours
  Notes:
    - Use large stripe sizes (256KB+) for maximum throughput
    - Enable async GPU operations for best performance
    - Monitor drive temperature (<70°C)

╔════════════════════════════════════════════════════════════╗
║                   Quick Start Command                      ║
╚════════════════════════════════════════════════════════════╝

# Create RAID 5 array:
./gpu_raid_cli create --raid=5 --drives=6 /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1 \
                       --profile=storage_profiles/nvme_pcie4_profile.json
```

### Detection Logic

**NVMe Detection:**
- Identifies PCIe generation from `/sys/class/nvme/nvmeX/device/current_link_speed`
- PCIe 4.0: 16.0 GT/s
- PCIe 3.0: 8.0 GT/s

**SATA/HDD Detection:**
- Reads `/sys/block/sdX/queue/rotational`
- `0` = SSD, `1` = HDD
- For HDDs, attempts to detect RPM from model string

**Drive Properties:**
- Size: From `/sys/block/DEV/size` (512-byte sectors)
- Queue depth: From `/sys/block/DEV/queue/nr_requests`
- Scheduler: From `/sys/block/DEV/queue/scheduler`

---

## drive_benchmark

Comprehensive GPU RAID performance benchmarking tool.

### Features
- Tests various block sizes (64KB - 2MB)
- Measures encode/decode throughput and latency
- Monitors GPU utilization and temperature
- Finds optimal block size for your configuration
- Can test real drives (DESTRUCTIVE!)

### Usage

```bash
# Comprehensive benchmark (all block sizes)
./drive_benchmark --comprehensive --raid=5 --drives=6

# Single block size test
./drive_benchmark --block=512 --raid=6 --drives=4

# Specify GPU
./drive_benchmark --comprehensive --raid=5 --drives=6 --gpu=1

# Test real drives (DANGEROUS - DESTROYS DATA!)
./drive_benchmark --devices /dev/sdb /dev/sdc /dev/sdd /dev/sde
```

### Example Output

```
╔════════════════════════════════════════════════════════════╗
║          GPU RAID Comprehensive Benchmark                  ║
╚════════════════════════════════════════════════════════════╝

Configuration:
  RAID Level: RAID 5
  Number of Drives: 6

Running benchmarks (this may take a few minutes)...

╔════════════╦═══════════╦═══════════╦═══════════╦═══════════╦═══════╗
║ Block Size ║  Encode   ║  Decode   ║   Encode  ║   Decode  ║  GPU  ║
║    (KB)    ║   (GB/s)  ║   (GB/s)  ║    (ms)   ║    (ms)   ║ Temp  ║
╠════════════╬═══════════╬═══════════╬═══════════╬═══════════╬═══════╣
║         64 ║     12.34 ║     10.12 ║     0.312 ║     0.380 ║  58.0 ║
║        128 ║     18.56 ║     15.23 ║     0.415 ║     0.506 ║  60.5 ║
║        256 ║     24.78 ║     20.34 ║     0.623 ║     0.759 ║  62.0 ║
║        512 ║     26.45 ║     21.89 ║     1.166 ║     1.409 ║  64.0 ║
║       1024 ║     25.12 ║     20.56 ║     2.456 ║     3.002 ║  65.5 ║
║       2048 ║     23.89 ║     19.45 ║     5.168 ║     6.341 ║  66.0 ║
╚════════════╩═══════════╩═══════════╩═══════════╩═══════════╩═══════╝

Optimal Block Size: 512 KB
  Peak Encode: 26.45 GB/s
  Peak Decode: 21.89 GB/s
```

### Options

```
-r, --raid LEVEL          RAID level (5 or 6, default: 5)
-n, --drives COUNT        Number of drives (default: 4)
-s, --stripe SIZE         Stripe size in KB (default: 256)
-b, --block SIZE          Block size in KB for single test
-c, --comprehensive       Run comprehensive benchmark
-d, --devices DEV...      Test real drives (DESTRUCTIVE!)
-g, --gpu ID              GPU device ID (default: 0)
-h, --help                Show help
```

### Real Drive Testing

**⚠️ WARNING:** Real drive testing is **DESTRUCTIVE** and will **ERASE ALL DATA**!

```bash
# Test actual write performance to drives
./drive_benchmark --devices /dev/sdb /dev/sdc /dev/sdd /dev/sde

⚠ WARNING: This test will WRITE to the specified drives!
           All data on these drives may be LOST!

Drives to test:
  - /dev/sdb
  - /dev/sdc
  - /dev/sdd
  - /dev/sde

Type 'YES' to confirm: YES

Running real drive I/O test...

Write Performance:
  Total Data: 4.00 GB
  Time: 2.145 seconds
  Throughput: 1.86 GB/s

⚠ Remember to recreate filesystems on these drives!
```

---

## smart_monitor

Monitors drive health using S.M.A.R.T. (Self-Monitoring, Analysis, and Reporting Technology).

### Features
- Reads SMART attributes from drives
- Detects failing drives early
- Monitors temperature, power-on hours, wear leveling
- Warns about reallocated sectors, pending sectors, errors
- Estimates RAID array reliability
- Continuous monitoring mode (5-minute intervals)

### Requirements
- `smartmontools` package (`smartctl` command)
- Root/sudo access

### Usage

```bash
# Detailed report for one drive
sudo ./smart_monitor --device nvme0n1

# Monitor RAID array
sudo ./smart_monitor --array sda sdb sdc sdd sde sdf

# Continuous monitoring (updates every 5 minutes)
sudo ./smart_monitor --array sda sdb sdc --continuous
```

### Example Output

**Summary View:**
```
╔════════════════════════════════════════════════════════════╗
║          GPU RAID SMART Health Monitor                     ║
╚════════════════════════════════════════════════════════════╝

Scan Time: Mon Nov 18 10:30:00 2025

╔══════════════╦══════════════════════════╦═══════╦═══════╦════════╗
║    Device    ║         Model            ║ Temp  ║ Hours ║ Status ║
╠══════════════╬══════════════════════════╬═══════╬═══════╬════════╣
║ nvme0n1      ║ Samsung SSD 980 PRO 1TB  ║  45.0 ║  8760 ║ OK     ║
║ nvme1n1      ║ Samsung SSD 980 PRO 1TB  ║  47.0 ║  8755 ║ OK     ║
║ nvme2n1      ║ Samsung SSD 980 PRO 1TB  ║  46.0 ║  8762 ║ OK     ║
║ nvme3n1      ║ Samsung SSD 980 PRO 1TB  ║  48.0 ║  8758 ║ WARN   ║
║ nvme4n1      ║ Samsung SSD 980 PRO 1TB  ║  44.0 ║  8761 ║ OK     ║
║ nvme5n1      ║ Samsung SSD 980 PRO 1TB  ║  46.0 ║  8759 ║ OK     ║
╚══════════════╩══════════════════════════╩═══════╩═══════╩════════╝

Summary:
  Healthy: 5
  Warnings: 1
  Failing: 0

Device /dev/nvme3n1:
  ⚠ WARNING: High temperature: 62°C

RAID Array Reliability:
  ⚠ CAUTION: 1 drive(s) showing warnings
    ACTION: Monitor closely, plan replacements
```

**Detailed Report:**
```
╔════════════════════════════════════════════════════════════╗
║          SMART Detailed Report                             ║
╚════════════════════════════════════════════════════════════╝

Device: /dev/sda
Model: WD Red Plus WD80EFZZ
Serial: WD-ABC123456789
Firmware: 81.00A81
SMART Enabled: Yes
SMART Health: PASSED

Key Metrics:
  Temperature: 42°C
  Power On Hours: 26280 hours (3 years)
  Power Cycles: 187
  Total Bytes Written: 15.6 TB
  Wear Leveling: 92% remaining

SMART Attributes:
╔════╦══════════════════════════════╦═══════╦═══════╦═══════╦═════════════╗
║ ID ║           Name               ║ Value ║ Worst ║ Thresh║  Raw Value  ║
╠════╬══════════════════════════════╬═══════╬═══════╬═══════╬═════════════╣
║  1 ║ Raw_Read_Error_Rate          ║  200  ║  200  ║   51  ║ 0           ║
║  3 ║ Spin_Up_Time                 ║  253  ║  253  ║   21  ║ 4333        ║
║  4 ║ Start_Stop_Count             ║  100  ║  100  ║    0  ║ 187         ║
║  5 ║ Reallocated_Sector_Ct        ║  200  ║  200  ║  140  ║ 0           ║
║  7 ║ Seek_Error_Rate              ║  200  ║  200  ║   51  ║ 0           ║
║  9 ║ Power_On_Hours               ║   85  ║   85  ║    0  ║ 26280       ║
║ 10 ║ Spin_Retry_Count             ║  100  ║  100  ║   51  ║ 0           ║
║ 12 ║ Power_Cycle_Count            ║  100  ║  100  ║    0  ║ 187         ║
║194 ║ Temperature_Celsius          ║  115  ║  109  ║    0  ║ 42          ║
║197 ║ Current_Pending_Sector       ║  200  ║  200  ║    0  ║ 0           ║
║198 ║ Offline_Uncorrectable        ║  200  ║  200  ║    0  ║ 0           ║
║199 ║ UDMA_CRC_Error_Count         ║  200  ║  200  ║    0  ║ 0           ║
╚════╩══════════════════════════════╩═══════╩═══════╩═══════╩═════════════╝
```

### SMART Attribute Warnings

The tool automatically warns about concerning SMART attributes:

- **ID 5 (Reallocated Sectors):** Bad sectors remapped. >0 = drive degrading
- **ID 187 (Uncorrectable Errors):** Read errors that couldn't be corrected
- **ID 188 (Command Timeout):** Drive not responding in time
- **ID 197 (Current Pending Sectors):** Sectors waiting to be remapped
- **ID 198 (Offline Uncorrectable):** Sectors that couldn't be read - **CRITICAL**
- **ID 194 (Temperature):** >60°C triggers warning
- **ID 177 (Wear Leveling):** SSD lifespan - <30% triggers warning, <10% critical

### Reliability Estimates

The tool estimates RAID array risk:

- **✓ All drives healthy:** No action needed
- **⚠ CAUTION:** 1+ drives showing warnings → Monitor closely
- **⚠ WARNING:** 1 drive failing → RAID 5 vulnerable, replace soon
- **✗ CRITICAL:** 2+ drives failing → RAID 6 at risk, **IMMEDIATE ACTION**

---

## Building the Tools

All tools are built automatically with CMake:

```bash
cd nvidia_gpu_raid
mkdir build && cd build
cmake ..
make -j$(nproc)

# Tools will be in: build/
ls -l gpu_raid_cli drive_detector drive_benchmark smart_monitor
```

## Installation

Install to system directories:

```bash
cd build
sudo make install

# Tools installed to: /usr/local/bin/
# Profiles installed to: /usr/local/etc/gpu_raid/storage_profiles/
```

## Integration Example

Typical workflow combining all tools:

```bash
# 1. Detect drives and get recommendations
sudo ./drive_detector --recommend > raid_plan.txt

# 2. Check drive health before creating array
sudo ./smart_monitor --array sda sdb sdc sdd sde sdf

# 3. Benchmark configuration
./drive_benchmark --comprehensive --raid=5 --drives=6 > benchmark.txt

# 4. Create RAID array (using recommended profile)
./gpu_raid_cli create --raid=5 --drives=6 \
  --profile=/usr/local/etc/gpu_raid/storage_profiles/nvme_pcie4_profile.json \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 \
  /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1

# 5. Monitor array health continuously
sudo ./smart_monitor --array nvme0n1 nvme1n1 nvme2n1 nvme3n1 nvme4n1 nvme5n1 --continuous
```

## Troubleshooting

### drive_detector not finding drives
```bash
# Check /sys/block exists
ls /sys/block/

# Run with root for full info
sudo ./drive_detector --list
```

### drive_benchmark shows poor performance
```bash
# Check GPU is detected
./gpu_raid_cli info

# Try different block sizes
./drive_benchmark --comprehensive --raid=5 --drives=4

# Check GPU isn't throttling
nvidia-smi
```

### smart_monitor fails
```bash
# Install smartmontools
sudo apt install smartmontools   # Debian/Ubuntu
sudo yum install smartmontools   # RHEL/CentOS

# Check smartctl works
sudo smartctl -a /dev/sda

# Run with sudo
sudo ./smart_monitor --device sda
```

## See Also

- [Storage Profiles](../storage_profiles/README.md)
- [API Documentation](../docs/API.md)
- [Performance Tuning](../docs/PERFORMANCE.md)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
