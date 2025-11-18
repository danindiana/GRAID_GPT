/**
 * GPU RAID SMART Monitoring Daemon
 *
 * Continuously monitors drive health using S.M.A.R.T. data
 * Logs warnings and critical events
 * Can trigger email alerts or shutdown on critical failures
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <syslog.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>

#define DAEMON_NAME "gpu_raid_smart"
#define PID_FILE "/var/run/gpu_raid_smart.pid"
#define CONFIG_FILE "/etc/gpu_raid/smart_daemon.conf"

/* Default configuration */
static int scan_interval = 300;  /* 5 minutes */
static int temp_warning = 55;    /* °C */
static int temp_critical = 65;   /* °C */
static int enable_email_alerts = 0;
static char email_address[256] = "";
static char monitored_devices[1024] = "";

static volatile sig_atomic_t running = 1;

/**
 * Signal handler
 */
static void signal_handler(int sig)
{
    switch (sig) {
    case SIGTERM:
    case SIGINT:
        syslog(LOG_INFO, "Received signal %d, shutting down", sig);
        running = 0;
        break;
    case SIGHUP:
        syslog(LOG_INFO, "Received SIGHUP, reloading configuration");
        /* TODO: Reload config */
        break;
    }
}

/**
 * Daemonize process
 */
static int daemonize(void)
{
    pid_t pid, sid;
    FILE *pidfile;

    /* Already a daemon */
    if (getppid() == 1)
        return 0;

    /* Fork off parent */
    pid = fork();
    if (pid < 0) {
        return -1;
    }

    /* Exit parent */
    if (pid > 0) {
        exit(EXIT_SUCCESS);
    }

    /* Create new session */
    sid = setsid();
    if (sid < 0) {
        return -1;
    }

    /* Change working directory */
    if (chdir("/") < 0) {
        return -1;
    }

    /* Close standard file descriptors */
    close(STDIN_FILENO);
    close(STDOUT_FILENO);
    close(STDERR_FILENO);

    /* Redirect to /dev/null */
    open("/dev/null", O_RDONLY);
    open("/dev/null", O_WRONLY);
    open("/dev/null", O_WRONLY);

    /* Write PID file */
    pidfile = fopen(PID_FILE, "w");
    if (pidfile) {
        fprintf(pidfile, "%d\n", getpid());
        fclose(pidfile);
    }

    /* Set file creation mask */
    umask(027);

    return 0;
}

/**
 * Load configuration
 */
static int load_config(void)
{
    FILE *f;
    char line[512];
    char key[256], value[256];

    f = fopen(CONFIG_FILE, "r");
    if (!f) {
        syslog(LOG_WARNING, "Cannot open config file %s, using defaults", CONFIG_FILE);
        return -1;
    }

    while (fgets(line, sizeof(line), f)) {
        /* Skip comments and empty lines */
        if (line[0] == '#' || line[0] == '\n')
            continue;

        if (sscanf(line, "%255s = %255s", key, value) == 2) {
            if (strcmp(key, "scan_interval") == 0) {
                scan_interval = atoi(value);
            } else if (strcmp(key, "temp_warning") == 0) {
                temp_warning = atoi(value);
            } else if (strcmp(key, "temp_critical") == 0) {
                temp_critical = atoi(value);
            } else if (strcmp(key, "enable_email") == 0) {
                enable_email_alerts = atoi(value);
            } else if (strcmp(key, "email_address") == 0) {
                strncpy(email_address, value, sizeof(email_address) - 1);
            } else if (strcmp(key, "devices") == 0) {
                strncpy(monitored_devices, value, sizeof(monitored_devices) - 1);
            }
        }
    }

    fclose(f);

    syslog(LOG_INFO, "Configuration loaded: scan_interval=%d, temp_warning=%d, temp_critical=%d",
           scan_interval, temp_warning, temp_critical);

    return 0;
}

/**
 * Parse SMART output for a device
 */
static int check_device_smart(const char *device)
{
    char cmd[512];
    char line[512];
    FILE *pipe;
    int temperature = 0;
    int reallocated_sectors = 0;
    int pending_sectors = 0;
    int offline_uncorrectable = 0;
    int health_passed = 1;

    snprintf(cmd, sizeof(cmd), "smartctl -A /dev/%s 2>/dev/null", device);

    pipe = popen(cmd, "r");
    if (!pipe) {
        syslog(LOG_ERR, "Failed to run smartctl for %s", device);
        return -1;
    }

    while (fgets(line, sizeof(line), pipe)) {
        int id, value, worst, thresh;
        char attr_name[64], raw_value[64];

        /* Parse SMART attribute line */
        if (sscanf(line, "%d %s %*s %d %d %d %*s %*s %*s %s",
                   &id, attr_name, &value, &worst, &thresh, raw_value) == 6) {

            /* Temperature (ID 194) */
            if (id == 194) {
                temperature = atoi(raw_value);
                if (temperature >= temp_critical) {
                    syslog(LOG_CRIT, "CRITICAL: %s temperature %d°C >= %d°C",
                           device, temperature, temp_critical);
                    /* TODO: Send email alert */
                } else if (temperature >= temp_warning) {
                    syslog(LOG_WARNING, "WARNING: %s temperature %d°C >= %d°C",
                           device, temperature, temp_warning);
                }
            }

            /* Reallocated Sectors (ID 5) */
            if (id == 5) {
                reallocated_sectors = atoi(raw_value);
                if (reallocated_sectors > 0) {
                    syslog(LOG_WARNING, "WARNING: %s has %d reallocated sectors",
                           device, reallocated_sectors);
                }
            }

            /* Current Pending Sectors (ID 197) */
            if (id == 197) {
                pending_sectors = atoi(raw_value);
                if (pending_sectors > 0) {
                    syslog(LOG_WARNING, "WARNING: %s has %d pending sectors",
                           device, pending_sectors);
                }
            }

            /* Offline Uncorrectable (ID 198) */
            if (id == 198) {
                offline_uncorrectable = atoi(raw_value);
                if (offline_uncorrectable > 0) {
                    syslog(LOG_CRIT, "CRITICAL: %s has %d offline uncorrectable sectors",
                           device, offline_uncorrectable);
                    health_passed = 0;
                }
            }

            /* Check if attribute is failing */
            if (value <= thresh && thresh > 0) {
                syslog(LOG_CRIT, "CRITICAL: %s attribute %d (%s) failing: %d <= %d",
                       device, id, attr_name, value, thresh);
                health_passed = 0;
            }
        }

        /* Check for overall health assessment */
        if (strstr(line, "SMART overall-health") && strstr(line, "FAILED")) {
            syslog(LOG_CRIT, "CRITICAL: %s overall SMART health FAILED", device);
            health_passed = 0;
        }
    }

    pclose(pipe);

    if (health_passed) {
        syslog(LOG_DEBUG, "Device %s healthy (temp=%d°C)", device, temperature);
    }

    return health_passed ? 0 : -1;
}

/**
 * Main monitoring loop
 */
static void monitoring_loop(void)
{
    char *devices_copy;
    char *device;
    char *saveptr;
    int failed_count;

    while (running) {
        failed_count = 0;

        syslog(LOG_DEBUG, "Starting SMART scan");

        /* Parse monitored devices (comma-separated) */
        devices_copy = strdup(monitored_devices);
        if (!devices_copy) {
            syslog(LOG_ERR, "Memory allocation failed");
            sleep(scan_interval);
            continue;
        }

        device = strtok_r(devices_copy, ",", &saveptr);
        while (device) {
            /* Trim whitespace */
            while (*device == ' ') device++;

            if (check_device_smart(device) < 0) {
                failed_count++;
            }

            device = strtok_r(NULL, ",", &saveptr);
        }

        free(devices_copy);

        if (failed_count > 0) {
            syslog(LOG_WARNING, "SMART scan completed: %d device(s) with issues", failed_count);
        } else {
            syslog(LOG_INFO, "SMART scan completed: all devices healthy");
        }

        /* Sleep until next scan */
        sleep(scan_interval);
    }
}

/**
 * Send email alert
 */
static void send_email_alert(const char *subject, const char *body)
{
    char cmd[1024];

    if (!enable_email_alerts || email_address[0] == '\0') {
        return;
    }

    snprintf(cmd, sizeof(cmd),
             "echo '%s' | mail -s '%s' %s",
             body, subject, email_address);

    if (system(cmd) != 0) {
        syslog(LOG_ERR, "Failed to send email alert");
    } else {
        syslog(LOG_INFO, "Email alert sent to %s", email_address);
    }
}

/**
 * Main
 */
int main(int argc, char **argv)
{
    int foreground = 0;
    int opt;

    /* Parse command line */
    while ((opt = getopt(argc, argv, "fh")) != -1) {
        switch (opt) {
        case 'f':
            foreground = 1;
            break;
        case 'h':
            printf("Usage: %s [-f] [-h]\n", argv[0]);
            printf("  -f  Run in foreground (don't daemonize)\n");
            printf("  -h  Show this help\n");
            return 0;
        default:
            return 1;
        }
    }

    /* Open syslog */
    openlog(DAEMON_NAME, LOG_PID | (foreground ? LOG_PERROR : 0), LOG_DAEMON);

    /* Load configuration */
    load_config();

    /* Daemonize unless in foreground mode */
    if (!foreground) {
        if (daemonize() < 0) {
            syslog(LOG_ERR, "Failed to daemonize");
            return 1;
        }
    }

    /* Setup signal handlers */
    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);
    signal(SIGHUP, signal_handler);

    syslog(LOG_INFO, "GPU RAID SMART daemon started (PID %d)", getpid());

    /* Main monitoring loop */
    monitoring_loop();

    /* Cleanup */
    syslog(LOG_INFO, "GPU RAID SMART daemon stopped");
    closelog();

    if (!foreground) {
        unlink(PID_FILE);
    }

    return 0;
}
