/*
 * Host check of the DA-ELM kernels: for a file of raw samples (float32[128] each), write each
 * prediction (int32) and, with elm_opt.c (-DELM_SCORES), the six output sums (float32[6]).
 *
 *   elm_check samples.bin output.bin
 */
#include <stdint.h>
#include <stdio.h>

#include "elm.h"

#ifdef ELM_SCORES
void elm_opt_scores(const float raw[ELM_IN], float out[ELM_OUT]);
#endif

int main(int argc, char **argv) {
    FILE *in = argc == 3 ? fopen(argv[1], "rb") : NULL, *out = argc == 3 ? fopen(argv[2], "wb") : NULL;
    if (!in || !out) {
        fprintf(stderr, "usage: elm_check samples.bin output.bin\n");
        return 2;
    }
    float raw[ELM_IN];
    while (fread(raw, sizeof raw, 1, in) == 1) {
        int32_t pred = elm_classify(raw);
        fwrite(&pred, sizeof pred, 1, out);
#ifdef ELM_SCORES
        float s[ELM_OUT];
        elm_opt_scores(raw, s);
        fwrite(s, sizeof s, 1, out);
#endif
    }
    fclose(in);
    fclose(out);
    return 0;
}
