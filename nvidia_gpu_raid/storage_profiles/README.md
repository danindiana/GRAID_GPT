# GPU RAID Storage Profiles

This directory contains optimized configuration profiles for different storage device types. Each profile is tailored to the specific characteristics and performance capabilities of different drive technologies.

## Available Profiles

### NVMe PCIe 4.0 (`nvme_pcie4_profile.json`)
**Best for:** High-end NVMe SSDs on PCIe 4.0 x4 lanes

**Characteristics:**
- Sequential Read: Up to 7000 MB/s
- Sequential Write: Up to 5000 MB/s
- Random IOPS: 1M+ IOPS
- Latency: ~10 μs

**Recommended Drives:**
- Samsung 980 Pro
- WD Black SN850
- Corsair MP600 Pro
- SK Hynix Platinum P41
- Seagate FireCuda 530

**Expected GPU RAID Performance:**
- RAID 5 Encode: 20-28 GB/s
- RAID 6 Encode: 15-22 GB/s
- Rebuild Time (8TB): <1 hour

**Optimal Configuration:**
- Stripe size: 256 KB
- Chunk size: 64 KB
- RAID level: 5 or 6
- Optimal drive count: 6

---

### NVMe PCIe 3.0 (`nvme_pcie3_profile.json`)
**Best for:** Mainstream NVMe SSDs on PCIe 3.0 x4 lanes

**Characteristics:**
- Sequential Read: Up to 3500 MB/s
- Sequential Write: Up to 3000 MB/s
- Random IOPS: 500K+ IOPS
- Latency: ~15 μs

**Recommended Drives:**
- Samsung 970 EVO Plus
- WD Blue SN570
- Crucial P3 Plus
- Kingston NV2
- Intel 670p

**Expected GPU RAID Performance:**
- RAID 5 Encode: 12-18 GB/s
- RAID 6 Encode: 10-14 GB/s
- Rebuild Time (8TB): 1.5-2 hours

**Optimal Configuration:**
- Stripe size: 128 KB
- Chunk size: 64 KB
- RAID level: 5 (or 6 for critical data)
- Optimal drive count: 4

---

### SATA SSD (`sata_ssd_profile.json`)
**Best for:** SATA III SSDs (cost-effective option)

**Characteristics:**
- Sequential Read: Up to 550 MB/s
- Sequential Write: Up to 520 MB/s
- Random IOPS: 98K IOPS
- Interface: SATA 6Gb/s (bandwidth limited)

**Recommended Drives:**
- Samsung 870 EVO
- Samsung 870 QVO
- Crucial MX500
- WD Blue 3D NAND
- SanDisk Ultra 3D

**Expected GPU RAID Performance:**
- RAID 5 Encode: 2.5-3.2 GB/s
- RAID 6 Encode: 2.0-2.8 GB/s
- Rebuild Time (8TB): 4.5-5.5 hours

**Optimal Configuration:**
- Stripe size: 64 KB (smaller for bandwidth constraints)
- Chunk size: 32 KB
- RAID level: 5
- Optimal drive count: 6-8 (to aggregate bandwidth)

**Limitations:**
- SATA interface caps throughput at ~600 MB/s per drive
- GPU RAID helps with parity but total throughput is SATA-limited
- Best for read-heavy workloads

---

### HDD 7200 RPM (`hdd_7200rpm_profile.json`)
**Best for:** Enterprise HDDs (bulk storage, archives, backups)

**Characteristics:**
- Sequential Read: Up to 180 MB/s
- Sequential Write: Up to 180 MB/s
- Random IOPS: 120 IOPS
- Rotational latency: 4.16 ms (7200 RPM)
- Seek time: ~8.5 ms

**Recommended Drives:**
- WD Red Plus (NAS)
- Seagate IronWolf
- Toshiba N300
- HGST Ultrastar (Enterprise)
- WD Gold (Enterprise)

**Expected GPU RAID Performance:**
- RAID 5 Encode: 1.0-1.4 GB/s
- RAID 6 Encode: 0.8-1.2 GB/s
- Rebuild Time (8TB): 12-14 hours

**Optimal Configuration:**
- Stripe size: 1024 KB (large to minimize seeks)
- Chunk size: 256 KB
- RAID level: **6** (dual parity for reliability)
- Optimal drive count: 8+ (to aggregate throughput)

**Important Notes:**
- GPU helps with parity calculations, but **HDD mechanics are the bottleneck**
- RAID 6 strongly recommended due to higher HDD failure rates
- Use for sequential workloads only (backups, archives, media storage)
- Not suitable for databases, VMs, or random I/O workloads
- Enable write cache and disable power management (APM) for performance
- Monitor SMART: reallocated sectors, pending sectors, CRC errors
- Keep drives cool (below 45°C)

**⚠️ SMR Drives Warning:**
- Shingled Magnetic Recording (SMR) drives **NOT recommended** for RAID
- SMR causes severe performance degradation during rebuild
- Only use CMR (Conventional Magnetic Recording) drives

---

## Using Profiles

### Manual Configuration

Load a profile and apply settings to your RAID array:

```bash
# Create RAID 5 array with NVMe PCIe 4.0 profile
./gpu_raid_cli create \
  --raid=5 \
  --drives=6 \
  --profile=storage_profiles/nvme_pcie4_profile.json \
  /dev/nvme0n1 /dev/nvme1n1 /dev/nvme2n1 /dev/nvme3n1 /dev/nvme4n1 /dev/nvme5n1
```

### Automatic Detection

Use the drive detector to automatically recommend the best profile:

```bash
# Detect drives and get recommendations
sudo ./drive_detector --recommend

# Apply recommended configuration
./drive_detector --list | grep nvme | xargs ./gpu_raid_cli create --auto
```

### Benchmark Your Configuration

Test actual performance with your drives:

```bash
# Comprehensive benchmark with different block sizes
./drive_benchmark --comprehensive --raid=5 --drives=6

# Single block size test
./drive_benchmark --block=512 --raid=6 --drives=4
```

## Profile Structure

Each JSON profile contains:

1. **Drive Characteristics** - Expected performance metrics
2. **RAID Configuration** - Recommended RAID level, drive counts, stripe sizes
3. **GPU RAID Settings** - Memory pool, streams, batch sizes
4. **I/O Scheduler** - Linux kernel scheduler recommendations
5. **Filesystem Recommendations** - Mount options for XFS, ext4, btrfs
6. **Performance Tuning** - VM settings, drive-specific parameters
7. **Expected Performance** - Throughput and rebuild time estimates
8. **Best Practices** - Deployment recommendations
9. **Compatibility** - Example drives and tested GPUs

## Applying System Tuning

Many profiles recommend system-level optimizations. Apply them with:

```bash
# Example for NVMe drives (from nvme_pcie4_profile.json)
echo none > /sys/block/nvme0n1/queue/scheduler
echo 256 > /sys/block/nvme0n1/queue/nr_requests
echo 256 > /sys/block/nvme0n1/queue/read_ahead_kb

# VM settings
sysctl -w vm.dirty_ratio=40
sysctl -w vm.dirty_background_ratio=10
sysctl -w vm.swappiness=10
```

**⚠️ Warning:** System tuning requires root access and affects all I/O. Test in non-production first!

## Mixed Drive Arrays

**Not Recommended!** Mixing drive types (e.g., NVMe + SATA) will limit performance to the slowest drive.

If you must mix drives:
- Create separate RAID arrays for each drive type
- Use tiered storage (SSD for hot data, HDD for cold)
- Never mix HDD and SSD in same array

## Custom Profiles

You can create custom profiles by copying and modifying existing ones:

```bash
cp nvme_pcie4_profile.json my_custom_profile.json
# Edit my_custom_profile.json with your settings
./gpu_raid_cli create --profile=my_custom_profile.json ...
```

Key parameters to adjust:
- `stripe_size_kb` - Larger = better sequential, smaller = better random
- `chunk_size_kb` - Typically stripe_size / num_drives
- `memory_pool_size_mb` - More = better performance, but uses GPU VRAM
- `num_streams` - More = better async performance (2-4 recommended)
- `batch_size_mb` - Larger = better throughput, more latency

## Validation

After applying a profile, validate performance:

```bash
# Check actual throughput
./drive_benchmark --comprehensive --raid=5 --drives=6

# Monitor GPU RAID stats
./gpu_raid_cli info

# Monitor drive health
sudo ./smart_monitor --array nvme0n1 nvme1n1 nvme2n1 nvme3n1
```

## References

- [API Documentation](../docs/API.md)
- [Performance Tuning Guide](../docs/PERFORMANCE.md)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)
- [Quick Start Guide](../docs/QUICK_START.md)

## Support

For drive-specific questions or to report issues:
- GitHub Issues: https://github.com/yourusername/GRAID_GPT/issues
- Include: drive model, GPU model, profile used, benchmark results
