/* Reading a scenario's header and initial device state (format in harness.c). */
#include <stdint.h>

#include "io.h"
#include "scenario.h"

#define MAGIC 0x31503253

void scenario_read_state(s2p_device *d, s2p_scenario *sc) {
    int32_t h[8];
    io_read(h, sizeof h);
    if (h[0] != MAGIC || h[1] < 1 || h[1] > S2P_MAX_CLASSES) io_fail("bad scenario");
    d->r.n_classes = h[1];
    sc->n_visits = h[2];
    d->pseudo = h[3];
    d->cap = h[4];
    sc->perms_given = h[5];
    float c[9];
    io_read(c, sizeof c);
    d->r.alpha = c[0]; d->r.opt_init = c[1]; d->r.t = c[2]; d->tau = c[3]; d->frac = c[4];
    d->k_weight = c[5]; d->ema_a = c[6]; d->ema_oma = c[7]; d->eps = c[8];
    io_read(d->r.unit_seed, sizeof d->r.unit_seed);
    io_read(&d->rng, sizeof d->rng);
    io_read(d->r.pos_weight, sizeof d->r.pos_weight);
    io_read(d->r.w, sizeof d->r.w);
    io_read(d->r.b, sizeof d->r.b);
    io_read(d->mu, sizeof d->mu);
    io_read(d->var, sizeof d->var);
    io_read(d->floor_, sizeof d->floor_);
}
