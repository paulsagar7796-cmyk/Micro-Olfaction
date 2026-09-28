/* Reading a scenario's header and initial device state (format in harness.c). */
#ifndef S2P_SCENARIO_H
#define S2P_SCENARIO_H

#include "s2p.h"

typedef struct {
    int n_visits;
    int perms_given;
} s2p_scenario;

void scenario_read_state(s2p_device *d, s2p_scenario *sc);

#endif
