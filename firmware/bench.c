/*
 * Cycle benchmarks on the Cortex-M4: one kind of operation per run, on real samples, so that every
 * library instruction the run executes belongs to that operation (cm4_profile.py counts them).
 *
 * Input: int32 mode, int32 count, then a scenario as for harness.c with one visit. Modes:
 *   0 front end        features and alignment of `count` later samples
 *   1 inference        front end + one-vs-rest scores + argmax (pseudo-labelling off)
 *   2 + test           as 1, plus the pseudo-label confidence test (no update: cap 0)
 *   3 + pseudo-labels  as 1 with pseudo-labelling on, updates included
 *   4 calibration      front end for the visit's labelled samples, then the 50 calibration passes
 *   5 DA-ELM           DA-ELM inference on `count` later samples
 * Output: int32 prediction per processed sample, then int32 pseudo-label updates.
 */
#include <stdint.h>

#include "elm.h"
#include "io.h"
#include "scenario.h"

enum { FRONT, INFER, INFER_TEST, INFER_PSEUDO, CALIBRATE, DAELM };

static s2p_device dev;
static int32_t cls[S2P_MAX_LABELLED];
static uint8_t perms[S2P_PASSES * S2P_MAX_LABELLED];

int main(int argc, char **argv) {
    io_open(argc > 2 ? argv[1] : "scenario.bin", argc > 2 ? argv[2] : "output.bin");
    int32_t cmd[2];
    io_read(cmd, sizeof cmd);
    s2p_device *d = &dev;
    s2p_scenario sc;
    scenario_read_state(d, &sc);
    int32_t n[2];
    io_read(n, sizeof n);
    if (n[0] < 0 || n[0] > S2P_MAX_LABELLED || n[1] < 0) io_fail("bad visit");
    io_read(cls, sizeof(int32_t) * (size_t)n[0]);
    if (sc.perms_given) io_read(perms, (size_t)S2P_PASSES * (size_t)n[0]);

    int mode = cmd[0];
    float raw[S2P_RAW], f[S2P_FEAT], y[S2P_FEAT];
    s2p_visit_begin(d);
    if (mode == CALIBRATE) {
        for (int i = 0; i < n[0]; i++) {
            io_read(raw, sizeof raw);
            s2p_features(d, raw, f);
            s2p_align(d, f, y);
            s2p_add_labelled(d, y, cls[i]);
        }
        s2p_calibrate(d, 0);
    } else {
        for (int i = 0; i < n[0]; i++) io_read(raw, sizeof raw);       /* labelled samples: not used */
        d->pseudo = mode >= INFER_TEST;
        if (mode == INFER_TEST) d->cap = 0;
        int count = cmd[1] < n[1] ? cmd[1] : n[1];
        for (int i = 0; i < count; i++) {
            io_read(raw, sizeof raw);
            int32_t pred = 0;
            if (mode == DAELM) {
                pred = elm_classify(raw);
            } else {
                s2p_features(d, raw, f);
                s2p_align(d, f, y);
                if (mode != FRONT) pred = s2p_classify(d, y);
            }
            io_write(&pred, sizeof pred);
        }
    }
    int32_t updates = (int32_t)d->updates;
    io_write(&updates, sizeof updates);
    io_close();
    return 0;
}
