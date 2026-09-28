/*
 * System 2+ for a microcontroller: feature map, running alignment, and a one-vs-rest logistic
 * readout trained by plain SGD exactly as scikit-learn's SGDClassifier(loss="log_loss",
 * penalty="l2", learning_rate="optimal"), with pseudo-labelling.
 *
 * Single precision throughout, static memory only, no libm. The specification is
 * device_reference.py (Arith style "chip", float32); check_firmware.py checks that this code
 * gives bit-identical results. Compile without floating-point contraction (-ffp-contract=off):
 * fused multiply-adds would change the rounding.
 *
 * Use per maintenance visit:
 *   s2p_visit_begin(d);
 *   for each labelled sample:  s2p_features(d, raw, f); s2p_align(d, f, y); s2p_add_labelled(d, y, label);
 *   s2p_calibrate(d, NULL);
 *   for each later sample:     s2p_features(d, raw, f); s2p_align(d, f, y); gas = s2p_classify(d, y);
 */
#ifndef S2P_H
#define S2P_H

#include <stdint.h>

#define S2P_SENSORS 16
#define S2P_TYPES 8
#define S2P_RAW (S2P_SENSORS * S2P_TYPES)      /* 128 features in, sensor by sensor */
#define S2P_FEAT (2 * S2P_RAW)                 /* 256 after adding the log-pattern features */
#define S2P_MAX_CLASSES 6
#ifndef S2P_MAX_LABELLED
#define S2P_MAX_LABELLED 50                    /* labelled samples per visit */
#endif
#define S2P_PASSES 50                          /* calibration passes per visit */

typedef struct {
    float w[S2P_MAX_CLASSES][S2P_FEAT];        /* one-vs-rest weights */
    float b[S2P_MAX_CLASSES];                  /* intercepts */
    float pos_weight[S2P_MAX_CLASSES];         /* class weight of each unit's positive class */
    uint32_t unit_seed[S2P_MAX_CLASSES];       /* per-unit shuffle seed, the same in every call */
    float t;                                   /* SGD step counter */
    float alpha, opt_init;                     /* L2 strength; t0 of the "optimal" step size */
    int n_classes;
} s2p_readout;

typedef struct {
    s2p_readout r;
    float mu[S2P_FEAT], var[S2P_FEAT];         /* running alignment statistics */
    float floor_[S2P_TYPES];                   /* per feature type, floor before the log */
    float ema_a, ema_oma, eps;                 /* retention a, 1 − a, variance floor */
    float k_weight;                            /* weight of labelled samples (K) */
    float tau, frac;                           /* pseudo-labels: confidence threshold, weight */
    int cap;                                   /* pseudo-labels: at most per class and visit */
    int pseudo;                                /* pseudo-labelling on/off */
    uint32_t rng;                              /* xorshift state for the calibration passes */
    int used[S2P_MAX_CLASSES];                 /* pseudo-labels used this visit */
    int n_lab;
    float lab_y[S2P_MAX_LABELLED][S2P_FEAT];   /* the visit's labelled samples, aligned */
    int lab_class[S2P_MAX_LABELLED];
    uint32_t updates;                          /* pseudo-label updates this visit */
} s2p_device;

float s2p_expf(float x);
float s2p_logf(float x);

void s2p_features(const s2p_device *d, const float raw[S2P_RAW], float f[S2P_FEAT]);
void s2p_align(s2p_device *d, const float f[S2P_FEAT], float y[S2P_FEAT]);
void s2p_decision(const s2p_readout *r, const float y[S2P_FEAT], float dec[S2P_MAX_CLASSES]);
void s2p_partial_fit(s2p_readout *r, const float *const *rows, const int *cls, const float *weight, int n);

void s2p_visit_begin(s2p_device *d);
int s2p_add_labelled(s2p_device *d, const float y[S2P_FEAT], int cls);
/* perms: S2P_PASSES × n_lab orders of the labelled samples, or NULL for the device's own shuffle */
void s2p_calibrate(s2p_device *d, const uint8_t *perms);
int s2p_classify(s2p_device *d, const float y[S2P_FEAT]);

#endif
