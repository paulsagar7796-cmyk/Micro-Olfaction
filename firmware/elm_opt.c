/*
 * DA-ELM inference, optimised with the same techniques as s2p_opt.c, and bit-identical to elm.c: each
 * hidden neuron's squared distance summed left to right eight terms at a time, the six output sums
 * held in registers across the 1000 neurons (each still summed in neuron order), exp inlined.
 * elm_opt_scores exposes the six output sums for the equivalence check (cm4_profile.py).
 */
#include "elm.h"

#include "fastmath.h"

extern const float elm_scale[ELM_IN], elm_shift[ELM_IN];
extern const float elm_centre[ELM_HIDDEN][ELM_IN], elm_width[ELM_HIDDEN];
extern const float elm_beta[ELM_HIDDEN][ELM_OUT];

void elm_opt_scores(const float raw[ELM_IN], float out[ELM_OUT]) {
    float x[ELM_IN];
    S2P_UNROLL(8)
    for (int i = 0; i < ELM_IN; i++) x[i] = raw[i] * elm_scale[i] + elm_shift[i];
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f, o4 = 0.0f, o5 = 0.0f;
    for (int j = 0; j < ELM_HIDDEN; j++) {
        const float *restrict a = elm_centre[j];
        float d2 = 0.0f;
        for (int i = 0; i < ELM_IN; i += 8) {
            float t;
            t = x[i] - a[i];         d2 = d2 + t * t;
            t = x[i + 1] - a[i + 1]; d2 = d2 + t * t;
            t = x[i + 2] - a[i + 2]; d2 = d2 + t * t;
            t = x[i + 3] - a[i + 3]; d2 = d2 + t * t;
            t = x[i + 4] - a[i + 4]; d2 = d2 + t * t;
            t = x[i + 5] - a[i + 5]; d2 = d2 + t * t;
            t = x[i + 6] - a[i + 6]; d2 = d2 + t * t;
            t = x[i + 7] - a[i + 7]; d2 = d2 + t * t;
        }
        const float h = fm_expf(-elm_width[j] * d2);
        const float *restrict bj = elm_beta[j];
        o0 = o0 + h * bj[0]; o1 = o1 + h * bj[1]; o2 = o2 + h * bj[2];
        o3 = o3 + h * bj[3]; o4 = o4 + h * bj[4]; o5 = o5 + h * bj[5];
    }
    out[0] = o0; out[1] = o1; out[2] = o2; out[3] = o3; out[4] = o4; out[5] = o5;
}

int elm_classify(const float raw[ELM_IN]) {
    float out[ELM_OUT];
    elm_opt_scores(raw, out);
    int best = 0;
    for (int k = 1; k < ELM_OUT; k++) if (out[k] > out[best]) best = k;
    return best;
}
