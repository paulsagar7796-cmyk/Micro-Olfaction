/*
 * Test harness: runs one scenario written by check_firmware.py through the s2p library and writes the
 * predictions and final state back for bit-by-bit comparison with device_reference.py. The same code
 * runs on the host (io_host.c) and on the emulated Cortex-M4 (cm4/io_semihost.c); samples are read one
 * at a time, so the chip needs no buffer beyond the device state.
 *
 *   host:  harness scenario.bin output.bin        chip: reads scenario.bin, writes output.bin
 *
 * Scenario (little-endian): int32 magic, n_classes, n_batches, pseudo, cap, perms_given, 0, 0;
 * float32 alpha, opt_init, t, tau, frac, k_weight, ema_a, ema_oma, eps; uint32 unit_seed[6], rng;
 * float32 pos_weight[6], w[6][256], b[6], mu[256], var[256], floor[8]; then per visit: int32 n_lab,
 * n_later; int32 cls[n_lab]; uint8 perms[50][n_lab] if perms_given; float32 raw[n_lab][128];
 * float32 raw[n_later][128].
 * Output: per visit int32 pred[n_later], then int32 pseudo-label updates; at the end float32 w[6][256],
 * b[6], t, mu[256], var[256].
 */
#include <stdint.h>

#include "io.h"
#include "scenario.h"

static s2p_device dev;                          /* static, as on the chip */
static int32_t cls[S2P_MAX_LABELLED];
static uint8_t perms[S2P_PASSES * S2P_MAX_LABELLED];

int main(int argc, char **argv) {
    io_open(argc > 2 ? argv[1] : "scenario.bin", argc > 2 ? argv[2] : "output.bin");
    s2p_device *d = &dev;
    s2p_scenario sc;
    scenario_read_state(d, &sc);
    int n_visits = sc.n_visits, perms_given = sc.perms_given;

    float raw[S2P_RAW], f[S2P_FEAT], y[S2P_FEAT];
    for (int v = 0; v < n_visits; v++) {
        int32_t n[2];
        io_read(n, sizeof n);
        if (n[0] < 0 || n[0] > S2P_MAX_LABELLED || n[1] < 0) io_fail("bad visit");
        io_read(cls, sizeof(int32_t) * (size_t)n[0]);
        if (perms_given) io_read(perms, (size_t)S2P_PASSES * (size_t)n[0]);
        s2p_visit_begin(d);
        for (int i = 0; i < n[0]; i++) {
            io_read(raw, sizeof raw);
            s2p_features(d, raw, f);
            s2p_align(d, f, y);
            s2p_add_labelled(d, y, cls[i]);
        }
        s2p_calibrate(d, perms_given ? perms : 0);
        for (int i = 0; i < n[1]; i++) {
            io_read(raw, sizeof raw);
            s2p_features(d, raw, f);
            s2p_align(d, f, y);
            int32_t pred = s2p_classify(d, y);
            io_write(&pred, sizeof pred);
        }
        int32_t updates = (int32_t)d->updates;
        io_write(&updates, sizeof updates);
    }
    io_write(d->r.w, sizeof d->r.w);
    io_write(d->r.b, sizeof d->r.b);
    io_write(&d->r.t, sizeof d->r.t);
    io_write(d->mu, sizeof d->mu);
    io_write(d->var, sizeof d->var);
    io_close();
    return 0;
}
