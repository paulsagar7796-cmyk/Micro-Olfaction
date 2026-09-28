/*
 * Host test harness: runs one scenario written by check_firmware.py through the s2p library and
 * writes the predictions and final state back for bit-by-bit comparison with device_reference.py.
 *
 *   host_harness scenario.bin output.bin
 *
 * Scenario (little-endian): int32 magic, n_classes, n_batches, pseudo, cap, perms_given, 0, 0;
 * float32 alpha, opt_init, t, tau, frac, k_weight, ema_a, ema_oma, eps; uint32 unit_seed[6], rng;
 * float32 pos_weight[6], w[6][256], b[6], mu[256], var[256], floor[8]; then per batch: int32 n_cal,
 * n_test; float32 raw[n_cal][128]; int32 cls[n_cal]; uint8 perms[50][n_cal] if perms_given;
 * float32 raw[n_test][128].
 * Output: per batch int32 n_test, updates, pred[n_test]; then float32 w[6][256], b[6], t, mu[256], var[256].
 */
#include <stdio.h>
#include <stdlib.h>

#include "s2p.h"

#define MAGIC 0x31503253

static FILE *in, *out;
static void rd(void *p, size_t size, size_t n) {
    if (n && fread(p, size, n, in) != n) { fprintf(stderr, "scenario truncated\n"); exit(2); }
}
static void wr(const void *p, size_t size, size_t n) { if (n) fwrite(p, size, n, out); }

static s2p_device dev;                          /* static: as on the chip */

int main(int argc, char **argv) {
    if (argc != 3 || !(in = fopen(argv[1], "rb")) || !(out = fopen(argv[2], "wb"))) {
        fprintf(stderr, "usage: host_harness scenario.bin output.bin\n");
        return 2;
    }
    int32_t h[8];
    rd(h, 4, 8);
    if (h[0] != MAGIC || h[1] > S2P_MAX_CLASSES) { fprintf(stderr, "bad scenario\n"); return 2; }
    s2p_device *d = &dev;
    d->r.n_classes = h[1];
    int n_batches = h[2], perms_given = h[5];
    d->pseudo = h[3];
    d->cap = h[4];
    float c[9];
    rd(c, 4, 9);
    d->r.alpha = c[0]; d->r.opt_init = c[1]; d->r.t = c[2]; d->tau = c[3]; d->frac = c[4];
    d->k_weight = c[5]; d->ema_a = c[6]; d->ema_oma = c[7]; d->eps = c[8];
    rd(d->r.unit_seed, 4, 6);
    rd(&d->rng, 4, 1);
    rd(d->r.pos_weight, 4, 6);
    rd(d->r.w, 4, 6 * S2P_FEAT);
    rd(d->r.b, 4, 6);
    rd(d->mu, 4, S2P_FEAT);
    rd(d->var, 4, S2P_FEAT);
    rd(d->floor_, 4, S2P_TYPES);

    float raw[S2P_RAW], f[S2P_FEAT], y[S2P_FEAT];
    for (int bt = 0; bt < n_batches; bt++) {
        int32_t n[2];
        rd(n, 4, 2);
        if (n[0] > S2P_MAX_LABELLED) { fprintf(stderr, "too many labelled samples: %d\n", n[0]); return 2; }
        float *cal = malloc(sizeof(float) * S2P_RAW * (n[0] ? n[0] : 1));
        int32_t *cls = malloc(sizeof(int32_t) * (n[0] ? n[0] : 1));
        uint8_t *perms = perms_given ? malloc((size_t)S2P_PASSES * (n[0] ? n[0] : 1)) : NULL;
        rd(cal, 4, (size_t)S2P_RAW * n[0]);
        rd(cls, 4, n[0]);
        if (perms_given) rd(perms, 1, (size_t)S2P_PASSES * n[0]);
        s2p_visit_begin(d);
        for (int i = 0; i < n[0]; i++) {
            s2p_features(d, cal + (size_t)i * S2P_RAW, f);
            s2p_align(d, f, y);
            s2p_add_labelled(d, y, cls[i]);
        }
        s2p_calibrate(d, perms);
        int32_t *pred = malloc(sizeof(int32_t) * (n[1] ? n[1] : 1));
        for (int i = 0; i < n[1]; i++) {
            rd(raw, 4, S2P_RAW);
            s2p_features(d, raw, f);
            s2p_align(d, f, y);
            pred[i] = s2p_classify(d, y);
        }
        int32_t updates = (int32_t)d->updates;
        wr(&n[1], 4, 1);
        wr(&updates, 4, 1);
        wr(pred, 4, n[1]);
        free(cal); free(cls); free(perms); free(pred);
    }
    wr(d->r.w, 4, 6 * S2P_FEAT);
    wr(d->r.b, 4, 6);
    wr(&d->r.t, 4, 1);
    wr(d->mu, 4, S2P_FEAT);
    wr(d->var, 4, S2P_FEAT);
    fclose(out);
    return 0;
}
