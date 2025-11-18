# Makefile for GPU RAID Integration Components

CC = gcc
CFLAGS = -Wall -O2 -fPIC
LDFLAGS = -shared

# Targets
TARGETS = libmd_raid_gpu.so liblvm_gpu.so libzfs_gpu.so smart_daemon

all: $(TARGETS)

# MD RAID integration (shared library for LD_PRELOAD)
libmd_raid_gpu.so: md_raid_integration.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# LVM integration (shared library for LD_PRELOAD)
liblvm_gpu.so: lvm_integration.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# ZFS integration (shared library for LD_PRELOAD)
libzfs_gpu.so: zfs_hooks.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

# SMART monitoring daemon
smart_daemon: smart_daemon.c
	$(CC) $(CFLAGS) -o gpu_raid_smart $<

clean:
	rm -f $(TARGETS) gpu_raid_smart *.o

install: all
	# Install shared libraries
	install -m 755 libmd_raid_gpu.so /usr/local/lib/
	install -m 755 liblvm_gpu.so /usr/local/lib/
	install -m 755 libzfs_gpu.so /usr/local/lib/
	ldconfig

	# Install SMART daemon
	install -m 755 gpu_raid_smart /usr/local/bin/

	# Install configuration
	mkdir -p /etc/gpu_raid
	install -m 644 smart_daemon.conf /etc/gpu_raid/

	# Install systemd service
	install -m 644 systemd/gpu-raid-smart.service /etc/systemd/system/
	systemctl daemon-reload

	@echo ""
	@echo "Installation complete!"
	@echo ""
	@echo "To enable SMART monitoring:"
	@echo "  sudo systemctl enable gpu-raid-smart"
	@echo "  sudo systemctl start gpu-raid-smart"
	@echo ""
	@echo "To use MD RAID integration:"
	@echo "  export LD_PRELOAD=/usr/local/lib/libmd_raid_gpu.so"
	@echo "  mdadm ..."
	@echo ""

uninstall:
	rm -f /usr/local/lib/libmd_raid_gpu.so
	rm -f /usr/local/lib/liblvm_gpu.so
	rm -f /usr/local/lib/libzfs_gpu.so
	rm -f /usr/local/bin/gpu_raid_smart
	rm -f /etc/systemd/system/gpu-raid-smart.service
	systemctl daemon-reload
	@echo "Uninstallation complete"

.PHONY: all clean install uninstall
