/**
 * Auto-Failover Module for SMART Daemon
 *
 * Automatically detects drive failures and initiates replacement/rebuild
 * Integrates with MD RAID, LVM, and ZFS
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <syslog.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#define MAX_SPARES 8
#define MAX_ARRAYS 16

/* Drive state */
enum drive_state {
    DRIVE_HEALTHY = 0,
    DRIVE_WARNING,       /* Showing early warning signs */
    DRIVE_DEGRADED,      /* Significant issues */
    DRIVE_FAILED,        /* Complete failure */
    DRIVE_REPLACED       /* Being replaced */
};

/* RAID array type */
enum array_type {
    ARRAY_TYPE_MD,
    ARRAY_TYPE_LVM,
    ARRAY_TYPE_ZFS
};

/* Drive info */
struct drive_info {
    char device[64];
    enum drive_state state;
    time_t last_check;
    int warning_count;
    int error_count;

    /* SMART attributes */
    int temperature;
    int reallocated_sectors;
    int pending_sectors;
    int offline_uncorrectable;
    int crc_errors;
};

/* RAID array info */
struct array_info {
    char name[128];
    enum array_type type;
    int num_devices;
    struct drive_info devices[32];
    int num_spares;
    char spares[MAX_SPARES][64];
    bool auto_rebuild;
};

static struct array_info arrays[MAX_ARRAYS];
static int num_arrays = 0;

/**
 * Check if drive has failed
 */
static bool is_drive_failed(const struct drive_info* drive) {
    /* Critical SMART failures */
    if (drive->offline_uncorrectable > 0) {
        return true;
    }

    /* Too many reallocated sectors */
    if (drive->reallocated_sectors > 100) {
        return true;
    }

    /* Consistent high error rate */
    if (drive->error_count > 10) {
        return true;
    }

    return false;
}

/**
 * Check if drive is degraded
 */
static bool is_drive_degraded(const struct drive_info* drive) {
    if (drive->reallocated_sectors > 10) {
        return true;
    }

    if (drive->pending_sectors > 0) {
        return true;
    }

    if (drive->crc_errors > 100) {
        return true;
    }

    if (drive->warning_count > 5) {
        return true;
    }

    return false;
}

/**
 * Select best spare drive
 */
static const char* select_spare_drive(struct array_info* array) {
    if (array->num_spares == 0) {
        return NULL;
    }

    /* For now, just return first spare */
    /* In production, would check spare drive health, size, etc. */
    return array->spares[0];
}

/**
 * Initiate MD RAID rebuild
 */
static int md_raid_replace_drive(const char* array_name,
                                 const char* failed_device,
                                 const char* spare_device) {
    char cmd[512];
    int ret;

    syslog(LOG_INFO, "MD RAID: Replacing %s with %s in %s",
           failed_device, spare_device, array_name);

    /* Mark drive as failed */
    snprintf(cmd, sizeof(cmd), "mdadm %s --fail %s", array_name, failed_device);
    ret = system(cmd);
    if (ret != 0) {
        syslog(LOG_ERR, "Failed to mark drive as failed: %s", failed_device);
        return -1;
    }

    /* Remove failed drive */
    snprintf(cmd, sizeof(cmd), "mdadm %s --remove %s", array_name, failed_device);
    ret = system(cmd);
    if (ret != 0) {
        syslog(LOG_WARNING, "Failed to remove drive (may already be removed): %s",
               failed_device);
    }

    /* Add spare drive */
    snprintf(cmd, sizeof(cmd), "mdadm %s --add %s", array_name, spare_device);
    ret = system(cmd);
    if (ret != 0) {
        syslog(LOG_ERR, "Failed to add spare drive: %s", spare_device);
        return -1;
    }

    syslog(LOG_INFO, "MD RAID: Successfully initiated rebuild on %s", array_name);

    /* Send email notification */
    send_failover_notification("MD RAID", array_name, failed_device, spare_device);

    return 0;
}

/**
 * Initiate LVM RAID rebuild
 */
static int lvm_raid_replace_drive(const char* lv_name,
                                 const char* failed_device,
                                 const char* spare_device) {
    char cmd[512];
    int ret;

    syslog(LOG_INFO, "LVM RAID: Replacing %s with %s in %s",
           failed_device, spare_device, lv_name);

    /* Use lvconvert to replace failed drive */
    snprintf(cmd, sizeof(cmd),
             "lvconvert --repair --use-policies %s --yes",
             lv_name);
    ret = system(cmd);

    if (ret != 0) {
        syslog(LOG_ERR, "LVM auto-repair failed for %s", lv_name);

        /* Try manual replacement */
        snprintf(cmd, sizeof(cmd),
                 "lvconvert --replace %s %s %s --yes",
                 failed_device, lv_name, spare_device);
        ret = system(cmd);

        if (ret != 0) {
            syslog(LOG_ERR, "LVM manual replacement failed");
            return -1;
        }
    }

    syslog(LOG_INFO, "LVM RAID: Successfully initiated rebuild on %s", lv_name);
    send_failover_notification("LVM RAID", lv_name, failed_device, spare_device);

    return 0;
}

/**
 * Initiate ZFS resilver
 */
static int zfs_replace_drive(const char* pool_name,
                            const char* failed_device,
                            const char* spare_device) {
    char cmd[512];
    int ret;

    syslog(LOG_INFO, "ZFS: Replacing %s with %s in pool %s",
           failed_device, spare_device, pool_name);

    /* ZFS replace command */
    snprintf(cmd, sizeof(cmd),
             "zpool replace %s %s %s",
             pool_name, failed_device, spare_device);
    ret = system(cmd);

    if (ret != 0) {
        syslog(LOG_ERR, "ZFS replace failed for pool %s", pool_name);
        return -1;
    }

    syslog(LOG_INFO, "ZFS: Successfully initiated resilver on %s", pool_name);
    send_failover_notification("ZFS", pool_name, failed_device, spare_device);

    return 0;
}

/**
 * Process drive failure and initiate replacement
 */
static void handle_drive_failure(struct array_info* array, int drive_idx) {
    struct drive_info* drive = &array->devices[drive_idx];

    if (drive->state == DRIVE_FAILED || drive->state == DRIVE_REPLACED) {
        return;  /* Already handling */
    }

    syslog(LOG_CRIT, "CRITICAL: Drive %s has FAILED in array %s",
           drive->device, array->name);

    /* Update state */
    drive->state = DRIVE_FAILED;

    /* Check if auto-rebuild is enabled */
    if (!array->auto_rebuild) {
        syslog(LOG_WARNING, "Auto-rebuild disabled for %s - manual intervention required",
               array->name);
        send_admin_alert("Drive Failure - Manual Action Required",
                        array->name, drive->device);
        return;
    }

    /* Select spare drive */
    const char* spare = select_spare_drive(array);
    if (!spare) {
        syslog(LOG_ERR, "No spare drives available for %s", array->name);
        send_admin_alert("Drive Failure - No Spares Available",
                        array->name, drive->device);
        return;
    }

    /* Initiate rebuild based on array type */
    int ret = -1;

    switch (array->type) {
    case ARRAY_TYPE_MD:
        ret = md_raid_replace_drive(array->name, drive->device, spare);
        break;

    case ARRAY_TYPE_LVM:
        ret = lvm_raid_replace_drive(array->name, drive->device, spare);
        break;

    case ARRAY_TYPE_ZFS:
        ret = zfs_replace_drive(array->name, drive->device, spare);
        break;
    }

    if (ret == 0) {
        drive->state = DRIVE_REPLACED;

        /* Remove spare from available list */
        for (int i = 0; i < array->num_spares - 1; i++) {
            if (strcmp(array->spares[i], spare) == 0) {
                memmove(&array->spares[i], &array->spares[i + 1],
                       (array->num_spares - i - 1) * sizeof(array->spares[0]));
                array->num_spares--;
                break;
            }
        }
    } else {
        syslog(LOG_ERR, "Failed to initiate auto-rebuild for %s", array->name);
        send_admin_alert("Auto-Rebuild Failed", array->name, drive->device);
    }
}

/**
 * Handle degraded drive (early warning)
 */
static void handle_degraded_drive(struct array_info* array, int drive_idx) {
    struct drive_info* drive = &array->devices[drive_idx];

    if (drive->state >= DRIVE_DEGRADED) {
        return;  /* Already marked */
    }

    syslog(LOG_WARNING, "WARNING: Drive %s is DEGRADED in array %s",
           drive->device, array->name);

    drive->state = DRIVE_DEGRADED;

    /* Send early warning to admin */
    send_admin_alert("Drive Degraded - Plan Replacement",
                    array->name, drive->device);

    /* If we have spares and proactive replacement is enabled, consider replacing */
    /* This would be configurable behavior */
}

/**
 * Update drive status from SMART data
 */
void update_drive_status(const char* device, const char* array_name,
                        int temperature, int reallocated, int pending,
                        int uncorrectable, int crc_errors) {
    /* Find array */
    struct array_info* array = NULL;
    int drive_idx = -1;

    for (int i = 0; i < num_arrays; i++) {
        if (strcmp(arrays[i].name, array_name) == 0) {
            array = &arrays[i];

            /* Find drive in array */
            for (int j = 0; j < array->num_devices; j++) {
                if (strcmp(array->devices[j].device, device) == 0) {
                    drive_idx = j;
                    break;
                }
            }
            break;
        }
    }

    if (!array || drive_idx < 0) {
        return;  /* Unknown array or drive */
    }

    struct drive_info* drive = &array->devices[drive_idx];

    /* Update SMART data */
    drive->temperature = temperature;
    drive->reallocated_sectors = reallocated;
    drive->pending_sectors = pending;
    drive->offline_uncorrectable = uncorrectable;
    drive->crc_errors = crc_errors;
    drive->last_check = time(NULL);

    /* Count warnings and errors */
    if (reallocated > 0 || pending > 0) {
        drive->warning_count++;
    }

    if (uncorrectable > 0 || crc_errors > 50) {
        drive->error_count++;
    }

    /* Check drive status */
    if (is_drive_failed(drive)) {
        handle_drive_failure(array, drive_idx);
    } else if (is_drive_degraded(drive)) {
        handle_degraded_drive(array, drive_idx);
    } else if (drive->state != DRIVE_HEALTHY) {
        /* Drive recovered? */
        if (drive->warning_count == 0 && drive->error_count == 0) {
            syslog(LOG_INFO, "Drive %s in %s appears to have recovered",
                   drive->device, array->name);
            drive->state = DRIVE_HEALTHY;
        }
    }
}

/**
 * Register RAID array for monitoring
 */
int register_array(const char* name, enum array_type type,
                  const char** devices, int num_devices,
                  const char** spares, int num_spares,
                  bool auto_rebuild) {
    if (num_arrays >= MAX_ARRAYS) {
        return -1;
    }

    struct array_info* array = &arrays[num_arrays++];

    strncpy(array->name, name, sizeof(array->name) - 1);
    array->type = type;
    array->num_devices = num_devices;
    array->num_spares = num_spares;
    array->auto_rebuild = auto_rebuild;

    for (int i = 0; i < num_devices; i++) {
        strncpy(array->devices[i].device, devices[i],
               sizeof(array->devices[i].device) - 1);
        array->devices[i].state = DRIVE_HEALTHY;
        array->devices[i].last_check = 0;
        array->devices[i].warning_count = 0;
        array->devices[i].error_count = 0;
    }

    for (int i = 0; i < num_spares && i < MAX_SPARES; i++) {
        strncpy(array->spares[i], spares[i], sizeof(array->spares[i]) - 1);
    }

    syslog(LOG_INFO, "Registered array %s for monitoring (%d devices, %d spares)",
           name, num_devices, num_spares);

    return 0;
}

/**
 * Send notification
 */
static void send_failover_notification(const char* type, const char* array,
                                       const char* failed, const char* spare) {
    char subject[256];
    char body[1024];

    snprintf(subject, sizeof(subject),
             "Auto-Failover: %s drive replaced in %s", type, array);

    snprintf(body, sizeof(body),
             "GPU RAID Auto-Failover Notification\n\n"
             "Array Type: %s\n"
             "Array Name: %s\n"
             "Failed Drive: %s\n"
             "Replacement Drive: %s\n"
             "Time: %s\n"
             "Status: Rebuild in progress\n\n"
             "Monitor rebuild progress with:\n"
             "  MD RAID: cat /proc/mdstat\n"
             "  LVM: lvs -a\n"
             "  ZFS: zpool status\n",
             type, array, failed, spare, ctime(&(time_t){time(NULL)}));

    /* Send via configured method (email, syslog, etc.) */
    syslog(LOG_NOTICE, "%s", body);

    /* If email configured, send it */
    send_email_alert(subject, body);
}

/**
 * Send admin alert
 */
static void send_admin_alert(const char* reason, const char* array,
                            const char* device) {
    char subject[256];
    char body[1024];

    snprintf(subject, sizeof(subject),
             "Admin Action Required: %s", reason);

    snprintf(body, sizeof(body),
             "GPU RAID Admin Alert\n\n"
             "Reason: %s\n"
             "Array: %s\n"
             "Device: %s\n"
             "Time: %s\n"
             "Action: Manual intervention required\n",
             reason, array, device, ctime(&(time_t){time(NULL)}));

    syslog(LOG_ALERT, "%s", body);
    send_email_alert(subject, body);
}

/**
 * Get array statistics
 */
void get_array_stats(char* buffer, size_t size) {
    int offset = 0;

    offset += snprintf(buffer + offset, size - offset,
                      "Auto-Failover Status:\n\n");

    for (int i = 0; i < num_arrays && offset < size; i++) {
        struct array_info* array = &arrays[i];

        offset += snprintf(buffer + offset, size - offset,
                          "Array: %s\n", array->name);
        offset += snprintf(buffer + offset, size - offset,
                          "  Type: %s\n",
                          array->type == ARRAY_TYPE_MD ? "MD RAID" :
                          array->type == ARRAY_TYPE_LVM ? "LVM" : "ZFS");
        offset += snprintf(buffer + offset, size - offset,
                          "  Auto-rebuild: %s\n",
                          array->auto_rebuild ? "Enabled" : "Disabled");
        offset += snprintf(buffer + offset, size - offset,
                          "  Spares: %d\n", array->num_spares);

        int healthy = 0, warning = 0, degraded = 0, failed = 0;
        for (int j = 0; j < array->num_devices; j++) {
            switch (array->devices[j].state) {
            case DRIVE_HEALTHY: healthy++; break;
            case DRIVE_WARNING: warning++; break;
            case DRIVE_DEGRADED: degraded++; break;
            case DRIVE_FAILED: failed++; break;
            case DRIVE_REPLACED: failed++; break;
            }
        }

        offset += snprintf(buffer + offset, size - offset,
                          "  Drives: %d healthy, %d warning, %d degraded, %d failed\n",
                          healthy, warning, degraded, failed);
        offset += snprintf(buffer + offset, size - offset, "\n");
    }
}
