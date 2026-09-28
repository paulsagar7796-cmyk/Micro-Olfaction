/*
 * DA-ELM inference on the same core, for the cycle comparison: the 128 features scaled to (−1, 1),
 * 1000 radial-basis hidden neurons h_j = exp(−b_j ‖x − a_j‖²), and a 1000 × 6 output layer
 * (the paper's 134,128 multiply-accumulates). Weights are random (timing only) and live in flash:
 * 0.5 MB of centres, too large for the SRAM of a part like the STM32F405.
 */
#ifndef S2P_ELM_H
#define S2P_ELM_H

#define ELM_HIDDEN 1000
#define ELM_IN 128
#define ELM_OUT 6

int elm_classify(const float raw[ELM_IN]);

#endif
