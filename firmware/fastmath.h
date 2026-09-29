/*
 * Inline versions of s2p.c's Cephes expf / logf and bit helpers, for the optimised kernels (s2p_opt.c,
 * elm_opt.c). Same operations in the same order as s2p_expf / s2p_logf, with the polynomial loops
 * unrolled, so the results are bit-identical; only call overhead and coefficient loads go away.
 */
#ifndef S2P_FASTMATH_H
#define S2P_FASTMATH_H

#include <stdint.h>

#define S2P_PRAGMA(x) _Pragma(#x)
#if defined(__clang__)
#define S2P_UNROLL(n) S2P_PRAGMA(clang loop unroll_count(n))
#elif defined(__GNUC__)
#define S2P_UNROLL(n) S2P_PRAGMA(GCC unroll n)
#else
#define S2P_UNROLL(n)
#endif

static inline uint32_t fm_bits(float x) { uint32_t u; __builtin_memcpy(&u, &x, 4); return u; }
static inline float fm_float(uint32_t u) { float x; __builtin_memcpy(&x, &u, 4); return x; }
static inline float fm_pow2(int n) { return fm_float((uint32_t)(n + 127) << 23); }   /* normal range only */
static inline float fm_floor_small(float x) {                                       /* |x| < 2^23 */
    float i = (float)(int32_t)x;
    return i > x ? i - 1.0f : i;
}

#define FM_C1 0.693359375f
#define FM_C2 (-2.12194440e-4f)

static inline float fm_expf(float x) {
    if (x < -87.3365478515625f) x = -87.3365478515625f;
    if (x > 88.0f) x = 88.0f;
    float fx = fm_floor_small(x * 1.44269504088896341f + 0.5f);
    x = x - fx * FM_C1;
    x = x - fx * FM_C2;
    float z = x * x;
    float y = 1.9875691500e-4f;
    y = y * x + 1.3981999507e-3f;
    y = y * x + 8.3334519073e-3f;
    y = y * x + 4.1665795894e-2f;
    y = y * x + 1.6666665459e-1f;
    y = y * x + 5.0000001201e-1f;
    y = y * z + x + 1.0f;
    return y * fm_pow2((int)fx);
}

static inline float fm_logf(float x) {                       /* x > 0, normal */
    uint32_t u = fm_bits(x);
    float e = (float)((int)((u >> 23) & 0xffu) - 126);
    float m = fm_float((u & 0x807fffffu) | (126u << 23));
    if (m < 0.707106781186547524f) { e = e - 1.0f; m = m + m - 1.0f; }
    else m = m - 1.0f;
    float z = m * m;
    float y = 7.0376836292e-2f;
    y = y * m + -1.1514610310e-1f;
    y = y * m + 1.1676998740e-1f;
    y = y * m + -1.2420140846e-1f;
    y = y * m + 1.4249322787e-1f;
    y = y * m + -1.6668057665e-1f;
    y = y * m + 2.0000714765e-1f;
    y = y * m + -2.4999993993e-1f;
    y = y * m + 3.3333331174e-1f;
    y = y * m * z;
    y = y + FM_C2 * e;
    y = y + -0.5f * z;
    m = m + y;
    return m + FM_C1 * e;
}

#endif
