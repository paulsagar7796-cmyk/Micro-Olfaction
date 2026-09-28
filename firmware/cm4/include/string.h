/* Freestanding build: the two string functions the firmware uses (defined in startup.c). */
#ifndef S2P_STRING_H
#define S2P_STRING_H

#include <stddef.h>

void *memcpy(void *restrict dst, const void *restrict src, size_t n);
void *memset(void *dst, int c, size_t n);

#endif
