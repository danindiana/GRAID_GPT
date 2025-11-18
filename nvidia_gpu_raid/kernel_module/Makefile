# Makefile for GPU RAID kernel module

obj-m += gpu_raid_kernel.o

KERNEL_VERSION ?= $(shell uname -r)
KERNEL_DIR ?= /lib/modules/$(KERNEL_VERSION)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KERNEL_DIR) M=$(PWD) modules_install
	depmod -a

load:
	sudo insmod gpu_raid_kernel.ko

unload:
	sudo rmmod gpu_raid_kernel

reload: unload load

.PHONY: all clean install load unload reload
