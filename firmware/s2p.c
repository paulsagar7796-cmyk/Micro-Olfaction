/* System 2+ on a microcontroller; see s2p.h. Each function mirrors device_reference.py. */
#include "s2p.h"

#include <string.h>

/* ── bit-level helpers (exact, instead of libm's frexpf / ldexpf / floorf); __builtin_memcpy so that a
 * freestanding build still compiles these to a register move rather than a call ── */
static inline uint32_t as_bits(float x) { uint32_t u; __builtin_memcpy(&u, &x, 4); return u; }
static inline float as_float(uint32_t u) { float x; __builtin_memcpy(&x, &u, 4); return x; }
static inline float pow2(int n) { return as_float((uint32_t)(n + 127) << 23); }   /* normal range only */
static inline float floor_small(float x) {                                       /* |x| < 2^23 */
    float i = (float)(int32_t)x;
    return i > x ? i - 1.0f : i;
}

/* ── Cephes expf / logf, as device_reference.expf / logf ── */
static const float EXP_P[6] = {1.9875691500e-4f, 1.3981999507e-3f, 8.3334519073e-3f,
                               4.1665795894e-2f, 1.6666665459e-1f, 5.0000001201e-1f};
static const float LOG_P[9] = {7.0376836292e-2f, -1.1514610310e-1f, 1.1676998740e-1f,
                               -1.2420140846e-1f, 1.4249322787e-1f, -1.6668057665e-1f,
                               2.0000714765e-1f, -2.4999993993e-1f, 3.3333331174e-1f};
#define C1 0.693359375f
#define C2 (-2.12194440e-4f)

float s2p_expf(float x) {
    if (x < -87.3365478515625f) x = -87.3365478515625f;
    if (x > 88.0f) x = 88.0f;
    float fx = floor_small(x * 1.44269504088896341f + 0.5f);
    x = x - fx * C1;
    x = x - fx * C2;
    float z = x * x, y = EXP_P[0];
    for (int i = 1; i < 6; i++) y = y * x + EXP_P[i];
    y = y * z + x + 1.0f;
    return y * pow2((int)fx);
}

float s2p_logf(float x) {                                   /* x > 0, normal */
    uint32_t u = as_bits(x);
    float e = (float)((int)((u >> 23) & 0xffu) - 126);
    float m = as_float((u & 0x807fffffu) | (126u << 23));   /* x = m · 2^e, m in [0.5, 1) */
    if (m < 0.707106781186547524f) { e = e - 1.0f; m = m + m - 1.0f; }
    else m = m - 1.0f;
    float z = m * m, y = LOG_P[0];
    for (int i = 1; i < 9; i++) y = y * m + LOG_P[i];
    y = y * m * z;
    y = y + C2 * e;
    y = y + -0.5f * z;
    m = m + y;
    return m + C1 * e;
}

/* ── scikit-learn's xorshift (our_rand_r) and Fisher–Yates shuffle ── */
static uint32_t xorshift(uint32_t *s) {
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

static float dot(const float *w, const float *x) {         /* left to right */
    float acc = 0.0f;
    for (int j = 0; j < S2P_FEAT; j++) acc += w[j] * x[j];
    return acc;
}

/* ── per sample ── */
static const float SIGN[S2P_TYPES] = {1, 1, 1, 1, 1, -1, -1, -1};   /* EMAd transients flipped */

void s2p_features(const s2p_device *d, const float raw[S2P_RAW], float f[S2P_FEAT]) {
    for (int g = 0; g < S2P_TYPES; g++) {
        float *x = f + g * S2P_SENSORS, *v = f + S2P_RAW + g * S2P_SENSORS, sum = 0.0f;
        for (int s = 0; s < S2P_SENSORS; s++) {
            x[s] = raw[s * S2P_TYPES + g] * SIGN[g];
            v[s] = s2p_logf(x[s] > d->floor_[g] ? x[s] : d->floor_[g]);
            sum += v[s];
        }
        float mean = sum / (float)S2P_SENSORS;
        for (int s = 0; s < S2P_SENSORS; s++) v[s] = v[s] - mean;
    }
}

void s2p_align(s2p_device *d, const float f[S2P_FEAT], float y[S2P_FEAT]) {
    for (int j = 0; j < S2P_FEAT; j++) {
        d->mu[j] = d->ema_a * d->mu[j] + d->ema_oma * f[j];
        float dev = f[j] - d->mu[j];
        d->var[j] = d->ema_a * d->var[j] + d->ema_oma * (dev * dev);
        y[j] = (f[j] - d->mu[j]) / __builtin_sqrtf(d->var[j] + d->eps);
    }
}

void s2p_decision(const s2p_readout *r, const float y[S2P_FEAT], float dec[S2P_MAX_CLASSES]) {
    for (int u = 0; u < r->n_classes; u++) dec[u] = dot(r->w[u], y) + r->b[u];
}

static int argmax(const float *v, int n) {
    int k = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[k]) k = i;
    return k;
}

/* cgradient_half_binomial, y in {0, 1} */
static float gradient(float y, float p) {
    if (p > -37.0f) {
        float e = s2p_expf(-p);
        return ((1.0f - y) - y * e) / (1.0f + e);
    }
    return s2p_expf(p) - y;
}

/* SGDClassifier.partial_fit on n samples (rows[s], class cls[s], sample weight weight[s]) */
void s2p_partial_fit(s2p_readout *r, const float *const *rows, const int *cls, const float *weight, int n) {
    uint8_t order[S2P_MAX_LABELLED > 1 ? S2P_MAX_LABELLED : 1];
    for (int u = 0; u < r->n_classes; u++) {
        uint32_t seed = r->unit_seed[u];                   /* passed by value, as in scikit-learn */
        fisher_yates(order, n, &seed);
        float *w = r->w[u], b = r->b[u], t = r->t, scale = 1.0f;
        for (int s = 0; s < n; s++) {
            int i = order[s];
            const float *x = rows[i];
            float y = cls[i] == u ? 1.0f : 0.0f;
            float p = dot(w, x) * scale + b;
            float eta = 1.0f / (r->alpha * (r->opt_init + t - 1.0f));
            float dloss = gradient(y, p);
            if (dloss < -1e12f) dloss = -1e12f;
            else if (dloss > 1e12f) dloss = 1e12f;
            float update = -eta * dloss;
            update = update * ((y > 0.0f ? r->pos_weight[u] : 1.0f) * weight[i]);
            float decay = 1.0f - eta * r->alpha;
            scale = scale * (decay > 0.0f ? decay : 0.0f);
            if (scale < 1e-6f) {
                for (int j = 0; j < S2P_FEAT; j++) w[j] = w[j] * scale;
                scale = 1.0f;
            }
            if (update != 0.0f) {
                float c = update / scale;
                for (int j = 0; j < S2P_FEAT; j++) w[j] = w[j] + x[j] * c;
                b = b + update;
            }
            t = t + 1.0f;
        }
        for (int j = 0; j < S2P_FEAT; j++) w[j] = w[j] * scale;
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

/* Pseudo-label confidence: the winner's share of the one-vs-rest probabilities, σ(m) / Σ σ(dᵢ) ≥ τ,
 * computed as 1 / Σ σ(dᵢ)/σ(m) so that it cannot underflow when every score is hugely negative (as
 * happens in drifted batches; the paper's float64 values are then tiny but representable). Where
 * float64's σ(m) is exactly 0 (m < −709.78) the paper's ratio is 0/0 and the test fails; so here. */
#define EXPIT64_ZERO (-709.782712893384f)
static int confident(const float *dec, int n, int k, float tau) {
    float m = dec[k], sum = 0.0f;
    if (m < EXPIT64_ZERO) return 0;
    if (m >= 0.0f) {
        float num = 1.0f + s2p_expf(-m);
        for (int i = 0; i < n; i++) sum += num / (1.0f + s2p_expf(-dec[i]));
    } else {
        float em = s2p_expf(m);
        for (int i = 0; i < n; i++) {
            float a = s2p_expf(dec[i] - m);
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
