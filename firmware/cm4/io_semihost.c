/* Scenario I/O on the chip: Arm semihosting (bkpt 0xAB), served by the emulator or a debug probe. */
#include <stdint.h>

#include "../io.h"

enum { SYS_OPEN = 0x01, SYS_CLOSE = 0x02, SYS_WRITE0 = 0x04, SYS_WRITE = 0x05, SYS_READ = 0x06,
       SYS_EXIT_EXTENDED = 0x20 };
#define ADP_STOPPED_APPLICATION_EXIT 0x20026

static int sh_call(int op, const void *arg) {
    register int r0 __asm__("r0") = op;
    register const void *r1 __asm__("r1") = arg;
    __asm__ volatile("bkpt 0xAB" : "+r"(r0) : "r"(r1) : "memory");
    return r0;
}

void sh_write0(const char *s) { sh_call(SYS_WRITE0, s); }

void sh_exit(int status) {
    uint32_t args[2] = {ADP_STOPPED_APPLICATION_EXIT, (uint32_t)status};
    sh_call(SYS_EXIT_EXTENDED, args);
    for (;;) {}
}

static int in_fd = -1, out_fd = -1;

static int sh_open(const char *path, int mode) {
    int len = 0;
    while (path[len]) len++;
    uint32_t args[3] = {(uint32_t)(uintptr_t)path, (uint32_t)mode, (uint32_t)len};
    return sh_call(SYS_OPEN, args);
}

void io_fail(const char *msg) {
    sh_write0(msg);
    sh_write0("\n");
    sh_exit(2);
}

void io_open(const char *in_path, const char *out_path) {
    in_fd = sh_open(in_path, 1);            /* "rb" */
    out_fd = sh_open(out_path, 5);          /* "wb" */
    if (in_fd < 0 || out_fd < 0) io_fail("cannot open scenario or output");
}

void io_read(void *p, size_t n) {
    uint32_t args[3] = {(uint32_t)in_fd, (uint32_t)(uintptr_t)p, (uint32_t)n};
    if (n && sh_call(SYS_READ, args) != 0) io_fail("scenario truncated");
}

void io_write(const void *p, size_t n) {
    uint32_t args[3] = {(uint32_t)out_fd, (uint32_t)(uintptr_t)p, (uint32_t)n};
    if (n && sh_call(SYS_WRITE, args) != 0) io_fail("cannot write output");
}

void io_close(void) {
    uint32_t a = (uint32_t)in_fd, b = (uint32_t)out_fd;
    sh_call(SYS_CLOSE, &a);
    sh_call(SYS_CLOSE, &b);
}
