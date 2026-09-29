/*
 * System 2+ for a Cortex-M4F, optimised: the same API (s2p.h) and the same arithmetic as the portable
 * reference s2p.c. Every sum keeps its order and every operation its rounding, so results are
 * bit-identical (check_firmware.py --variant opt); the speed comes from
 *   - the six one-vs-rest scores computed in one pass over the features, eight at a time (each unit's
 *     sum still accumulated left to right),
 *   - dot products and weight updates unrolled by eight,
 *   - the alignment constants held in registers instead of reloaded per feature,
 *   - exp and log inlined (fastmath.h).
 * Compile without floating-point contraction (-ffp-contract=off), as s2p.c.
 */
#include "s2p.h"

#include <string.h>

#include "fastmath.h"

float s2p_expf(float x) { return fm_expf(x); }
float s2p_logf(float x) { return fm_logf(x); }

/* ── scikit-learn's xorshift (our_rand_r) and Fisher–Yates shuffle, as s2p.c ── */
static inline uint32_t xorshift(uint32_t *s) {
    if (*s == 0) *s = 1;
    *s ^= *s << 13;
    *s ^= *s >> 17;
    *s ^= *s << 5;
    return *s % 0x80000000u;
}

static void fisher_yates(uint8_t *ind, int n, uint32_t *state) {
    for (int i = 0; i < n; i++) ind[i] = (uint8_t)i;
    for (unsigned i = 0; i + 1 < (unsigned)n; i++) {
        unsigned j = i + xorshift(state) % ((unsigned)n - i);
        uint8_t tmp = ind[i]; ind[i] = ind[j]; ind[j] = tmp;
    }
}

/* left-to-right dot product of 256, eight terms per iteration */
static inline float dot256(const float *restrict w, const float *restrict x) {
    float acc = 0.0f;
    for (int j = 0; j < S2P_FEAT; j += 8) {
        acc = acc + w[j] * x[j];
        acc = acc + w[j + 1] * x[j + 1];
        acc = acc + w[j + 2] * x[j + 2];
        acc = acc + w[j + 3] * x[j + 3];
        acc = acc + w[j + 4] * x[j + 4];
        acc = acc + w[j + 5] * x[j + 5];
        acc = acc + w[j + 6] * x[j + 6];
        acc = acc + w[j + 7] * x[j + 7];
    }
    return acc;
}

/* w = w + x·c and w = w·s, element by element */
static inline void axpy256(float *restrict w, const float *restrict x, float c) {
    S2P_UNROLL(8)
    for (int j = 0; j < S2P_FEAT; j++) w[j] = w[j] + x[j] * c;
}

static inline void scale256(float *restrict w, float s) {
    S2P_UNROLL(8)
    for (int j = 0; j < S2P_FEAT; j++) w[j] = w[j] * s;
}

/* ── per sample ── */
void s2p_features(const s2p_device *d, const float raw[S2P_RAW], float f[S2P_FEAT]) {
    for (int g = 0; g < S2P_TYPES; g++) {
        float *restrict x = f + g * S2P_SENSORS, *restrict v = f + S2P_RAW + g * S2P_SENSORS;
        const float fl = d->floor_[g], sign = g < 5 ? 1.0f : -1.0f;      /* EMAd transients flipped */
        float sum = 0.0f;
        S2P_UNROLL(4)
        for (int s = 0; s < S2P_SENSORS; s++) {
            float xs = raw[s * S2P_TYPES + g] * sign;
            x[s] = xs;
            float vs = fm_logf(xs > fl ? xs : fl);
            v[s] = vs;
            sum = sum + vs;
        }
        const float mean = sum / (float)S2P_SENSORS;
        S2P_UNROLL(8)
        for (int s = 0; s < S2P_SENSORS; s++) v[s] = v[s] - mean;
    }
}

void s2p_align(s2p_device *d, const float f[S2P_FEAT], float y[S2P_FEAT]) {
    const float a = d->ema_a, oma = d->ema_oma, eps = d->eps;
    float *restrict mu = d->mu, *restrict var = d->var;
    const float *restrict fr = f;
    float *restrict yr = y;
    S2P_UNROLL(4)
    for (int j = 0; j < S2P_FEAT; j++) {
        const float fj = fr[j];
        const float m = a * mu[j] + oma * fj;
        const float dev = fj - m;
        const float v = a * var[j] + oma * (dev * dev);
        mu[j] = m;
        var[j] = v;
        yr[j] = dev / __builtin_sqrtf(v + eps);
    }
}

/* All six scores in one pass; units beyond n_classes hold zero weights and are not used by callers. */
void s2p_decision(const s2p_readout *r, const float y[S2P_FEAT], float dec[S2P_MAX_CLASSES]) {
    const float *restrict w0 = r->w[0], *restrict w1 = r->w[1], *restrict w2 = r->w[2];
    const float *restrict w3 = r->w[3], *restrict w4 = r->w[4], *restrict w5 = r->w[5];
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f, a5 = 0.0f;
    for (int j = 0; j < S2P_FEAT; j += 4) {
        const float x0 = y[j], x1 = y[j + 1], x2 = y[j + 2], x3 = y[j + 3];
        a0 = a0 + w0[j] * x0; a0 = a0 + w0[j + 1] * x1; a0 = a0 + w0[j + 2] * x2; a0 = a0 + w0[j + 3] * x3;
        a1 = a1 + w1[j] * x0; a1 = a1 + w1[j + 1] * x1; a1 = a1 + w1[j + 2] * x2; a1 = a1 + w1[j + 3] * x3;
        a2 = a2 + w2[j] * x0; a2 = a2 + w2[j + 1] * x1; a2 = a2 + w2[j + 2] * x2; a2 = a2 + w2[j + 3] * x3;
        a3 = a3 + w3[j] * x0; a3 = a3 + w3[j + 1] * x1; a3 = a3 + w3[j + 2] * x2; a3 = a3 + w3[j + 3] * x3;
        a4 = a4 + w4[j] * x0; a4 = a4 + w4[j + 1] * x1; a4 = a4 + w4[j + 2] * x2; a4 = a4 + w4[j + 3] * x3;
        a5 = a5 + w5[j] * x0; a5 = a5 + w5[j + 1] * x1; a5 = a5 + w5[j + 2] * x2; a5 = a5 + w5[j + 3] * x3;
    }
    dec[0] = a0 + r->b[0]; dec[1] = a1 + r->b[1]; dec[2] = a2 + r->b[2];
    dec[3] = a3 + r->b[3]; dec[4] = a4 + r->b[4]; dec[5] = a5 + r->b[5];
}

static inline int argmax(const float *v, int n) {
    int k = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[k]) k = i;
    return k;
}

/* cgradient_half_binomial, y in {0, 1} */
static inline float gradient(float y, float p) {
    if (p > -37.0f) {
        float e = fm_expf(-p);
        return ((1.0f - y) - y * e) / (1.0f + e);
    }
    return fm_expf(p) - y;
}

void s2p_partial_fit(s2p_readout *r, const float *const *rows, const int *cls, const float *weight, int n) {
    uint8_t order[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    const float alpha = r->alpha, opt_init = r->opt_init;
    for (int u = 0; u < r->n_classes; u++) {
        uint32_t seed = r->unit_seed[u];
        fisher_yates(order, n, &seed);
        float *w = r->w[u], b = r->b[u], t = r->t, scale = 1.0f;
        const float pos = r->pos_weight[u];
        for (int s = 0; s < n; s++) {
            int i = order[s];
            const float *x = rows[i];
            float y = cls[i] == u ? 1.0f : 0.0f;
            float p = dot256(w, x) * scale + b;
            float eta = 1.0f / (alpha * (opt_init + t - 1.0f));
            float dloss = gradient(y, p);
            if (dloss < -1e12f) dloss = -1e12f;
            else if (dloss > 1e12f) dloss = 1e12f;
            float update = -eta * dloss;
            update = update * ((y > 0.0f ? pos : 1.0f) * weight[i]);
            float decay = 1.0f - eta * alpha;
            scale = scale * (decay > 0.0f ? decay : 0.0f);
            if (scale < 1e-6f) {
                scale256(w, scale);
                scale = 1.0f;
            }
            if (update != 0.0f) {
                axpy256(w, x, update / scale);
                b = b + update;
            }
            t = t + 1.0f;
        }
        scale256(w, scale);
        r->b[u] = b;
    }
    r->t = r->t + (float)n;
}

/* ── per visit ── */
void s2p_visit_begin(s2p_device *d) {
    d->n_lab = 0;
    d->updates = 0;
    for (int u = 0; u < S2P_MAX_CLASSES; u++) d->used[u] = 0;
}

int s2p_add_labelled(s2p_device *d, const float y[S2P_FEAT], int cls) {
    if (d->n_lab >= S2P_MAX_LABELLED) return -1;
    memcpy(d->lab_y[d->n_lab], y, sizeof(float) * S2P_FEAT);
    d->lab_class[d->n_lab++] = cls;
    return 0;
}

void s2p_calibrate(s2p_device *d, const uint8_t *perms) {
    const float *rows[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    int cls[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    float weight[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    uint8_t own[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    int n = d->n_lab;
    if (n == 0) return;
    for (int p = 0; p < S2P_PASSES; p++) {
        const uint8_t *perm = perms ? perms + p * n : own;
        if (!perms) fisher_yates(own, n, &d->rng);
        for (int s = 0; s < n; s++) {
            rows[s] = d->lab_y[perm[s]];
            cls[s] = d->lab_class[perm[s]];
            weight[s] = d->k_weight;
        }
        s2p_partial_fit(&d->r, rows, cls, weight, n);
    }
}

/* Pseudo-label confidence test, as s2p.c (underflow-safe: 1 / Σ σ(dᵢ)/σ(m) ≥ τ) */
#define EXPIT64_ZERO (-709.782712893384f)
static inline int confident(const float *dec, int n, int k, float tau) {
    float m = dec[k], sum = 0.0f;
    if (m < EXPIT64_ZERO) return 0;
    if (m >= 0.0f) {
        float num = 1.0f + fm_expf(-m);
        for (int i = 0; i < n; i++) sum += num / (1.0f + fm_expf(-dec[i]));
    } else {
        float em = fm_expf(m);
        for (int i = 0; i < n; i++) {
            float a = fm_expf(dec[i] - m);
            sum += a * (1.0f + em) / (1.0f + a * em);
        }
    }
    return 1.0f / sum >= tau;
}

int s2p_classify(s2p_device *d, const float y[S2P_FEAT]) {
    float dec[S2P_MAX_CLASSES];
    s2p_decision(&d->r, y, dec);
    int k = argmax(dec, d->r.n_classes);
    if (!d->pseudo) return k;
    if (confident(dec, d->r.n_classes, k, d->tau) && d->used[k] < d->cap) {
        const float *row = y;
        float weight = d->frac;
        s2p_partial_fit(&d->r, &row, &k, &weight, 1);
        d->used[k]++;
        d->updates++;
    }
    return k;
}
