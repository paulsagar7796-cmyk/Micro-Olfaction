/*
 * Cortex-M4F start-up for the STM32F405 memory map (stm32f405.ld): vector table, .data/.bss set-up,
 * FPU enable, and stack painting. After main returns, the deepest stack use is reported through
 * semihosting ("stack_max_bytes=N") and the program exits with main's status.
 */
#include <stdint.h>
#include <string.h>

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
int main(int argc, char **argv);
void sh_write0(const char *s);
void sh_exit(int status);

#define PAINT 0xA5A5A5A5u

void *memcpy(void *restrict dst, const void *restrict src, size_t n) {
    uint8_t *d = dst;
    const uint8_t *s = src;
    if ((((uintptr_t)d | (uintptr_t)s) & 3u) == 0) {
        for (; n >= 4; n -= 4, d += 4, s += 4) *(uint32_t *)(void *)d = *(const uint32_t *)(const void *)s;
    }
    while (n--) *d++ = *s++;
    return dst;
}

void *memset(void *dst, int c, size_t n) {
    uint8_t *d = dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

static void put_uint(const char *label, uint32_t v) {
    char buf[48], digits[12];
    int n = 0, k = 0;
    while (label[n]) { buf[n] = label[n]; n++; }
    do { digits[k++] = (char)('0' + v % 10u); v /= 10u; } while (v);
    while (k) buf[n++] = digits[--k];
    buf[n++] = '\n';
    buf[n] = 0;
    sh_write0(buf);
}

static void fault_handler(void) {
    sh_write0("fault\n");
    sh_exit(3);
}

void reset_handler(void) {
    uint32_t *src = &_sidata, *dst = &_sdata;
    while (dst < &_edata) *dst++ = *src++;
    for (dst = &_sbss; dst < &_ebss;) *dst++ = 0;
    *(volatile uint32_t *)0xE000ED88u |= 0xFu << 20;          /* CPACR: full access to CP10, CP11 (FPU) */
    __asm volatile("dsb\n\tisb" ::: "memory");
    uint32_t sp;
    __asm volatile("mov %0, sp" : "=r"(sp));
    for (uint32_t *p = &_ebss; p < (uint32_t *)(sp - 64u); p++) *p = PAINT;
    int status = main(0, 0);
    uint32_t *p = &_ebss;
    while (p < &_estack && *p == PAINT) p++;
    put_uint("stack_max_bytes=", (uint32_t)((uintptr_t)&_estack - (uintptr_t)p));
    put_uint("static_ram_bytes=", (uint32_t)((uintptr_t)&_ebss - (uintptr_t)&_sdata));
    sh_exit(status);
}

__attribute__((section(".isr_vector"), used)) static void (*const vectors[16])(void) = {
    (void (*)(void))(uintptr_t)&_estack, reset_handler, fault_handler, fault_handler, fault_handler,
    fault_handler, fault_handler, 0, 0, 0, 0, fault_handler, fault_handler, 0, fault_handler, fault_handler,
};
