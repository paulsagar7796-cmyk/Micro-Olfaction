[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23074395.svg)](https://doi.org/10.5281/zenodo.23074395)

# Olfactory simulation

Drift-robust gas classification on the UCI Gas Sensor Array Drift dataset ("System 2+", called SOLAR+ in
the preprint below: running feature alignment and an online linear readout, calibrated with a few
labelled samples per gas), plus a simulated MOx sensor front end with a spiking presence detector. The
results for the paper are in [paper/comparison_tables.md](paper/comparison_tables.md).

**Preprint:** [Micro-Olfaction: Few-Shot Streaming Drift Compensation for Gas Sensor Arrays on
Microcontrollers (PDF)](https://doi.org/10.5281/zenodo.23074212), Sagar Paul, 2026.

## Citation

If you use this code or cite the preprint, please use:

```bibtex
@misc{paul2026microolfaction,
  author    = {Paul, Sagar},
  title     = {Micro-Olfaction: Few-Shot Streaming Drift Compensation for
               Gas Sensor Arrays on Microcontrollers},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23074212},
  url       = {https://doi.org/10.5281/zenodo.23074212}
}
```

The code and results archive has its own DOI: `10.5281/zenodo.23074395` (see `CITATION.cff`).

## Reproducing the paper's numbers

```bash
python3 -m pip install -r requirements.txt   # exact versions; Python 3.13.2
./reproduce.sh                               # ≈1.5 h on 10 cores, plus ≈20 min for the C-port steps
```

`reproduce.sh` downloads the dataset if it is missing (`fetch_data.py`) and verifies it against
`data/SHA256SUMS`. It then reruns every script behind the paper and writes to `results/`, with logs in
`results/logs/`. Finally it checks two things:

- `check_reproduction.py` compares every regenerated file with the committed version.
- `check_paper_numbers.py` recomputes each number in `paper/comparison_tables.md` from `results/`.

`./reproduce.sh confound table` reruns only the named steps; the step list is at the top of the script. The C-port steps
(the microcontroller figures) run only when the Arm tools below are installed; otherwise the committed
results are checked as they are.

On the same platform as the committed run (macOS, Apple silicon) and with the pinned versions, every
file should come out identical. **On a different CPU architecture the pinned versions are not
sufficient.** Stochastic-gradient training is sensitive to floating-point details, and a different
instruction set or BLAS implementation (for example OpenBLAS on x86-64 against Apple Accelerate on
arm64) rounds the same operations differently, which is enough to send individual runs onto
different trajectories. An independent reproduction on Windows x86-64 with identical package
versions regenerated 5 of 14 files identically and 9 with small differences; the paper's table
values moved by at most 0.6 points, inside the reported seed spreads
([preprint/audit/windows-x86-2026-10.md](preprint/audit/windows-x86-2026-10.md)). The seed spreads
reported in the paper indicate how large such changes can be.

## Layout

| Files | Purpose |
|---|---|
| `fetch_data.py`, `dataset_adapter.py` | download and load the UCI dataset (gas naming explained in the paper, section 1) |
| `tune_system2.py`, `sweep_system2_budget.py`, `benchmark_literature_comparison.py` | System 2: tuning on Batches 2–5, label budgets, Settings 1 and 2 |
| `improve_system2.py`, `combine_system2.py` | System 2+: candidate improvements and their combination |
| `sequential_system2plus.py` | System 2+ in the sequential maintenance protocol (pseudo-labels carried over) |
| `literature_table.py`, `literature/published_results.csv` | published results and the unified comparison |
| `check_temporal_confound.py` | file-order and concentration-level hold-out checks |
| `deep_model_cost.py` | compute and memory estimates for all compared methods |
| `snn_input.py`, `lif_neurons.py`, `presence_test.py`, `energy_proxy.py` | simulated sensor front end and presence detector |
| `device_reference.py` | NumPy specification of System 2+ as it runs on a microcontroller (single precision) |
| `check_device_reference.py` | the specification against scikit-learn and the paper, and what single precision changes |
| `firmware/` | the C library (`s2p.c`, `s2p.h`), the scenario harness and cycle benchmarks, host and Cortex-M4F builds; `check_firmware.py [--target cm4]` compares it with the specification bit for bit |
| `firmware/s2p_opt.c`, `firmware/elm_opt.c` | optimised kernels (unrolled, register-blocked, inlined exp/log), bit-identical to `s2p.c` / `elm.c`, which stay the portable reference |
| `cm4_profile.py` | cycles on a Cortex-M4F for both variants (exact instruction profile under QEMU, Cortex-M4 timing tables) and energy on the STM32L476 and Apollo3 Blue from their datasheets |
| other scripts, older files in `results/` | exploration before the gas naming was harmonised; not used by the paper |

## Cortex-M4 builds

The Cortex-M4F builds use the xPack Arm GCC and QEMU (Apple-silicon archives from
github.com/xpack-dev-tools, unpacked in `~/opt`; no installer needed). Target: the STM32F405 memory
map (1 MB flash, 128 KB SRAM), emulated as QEMU's `netduinoplus2`; the firmware reads its test cases
from the host through Arm semihosting.

```bash
python3 check_firmware.py --target cm4                 # the Arm binary against the specification, bit for bit
python3 check_firmware.py --target cm4 --variant opt   # the same for the optimised kernels
python3 cm4_profile.py      # instruction counts (exact), cycles (Cortex-M4 timing model) and energy, both variants
```

QEMU is not cycle-accurate: instruction counts are exact, cycles come from applying the Cortex-M4
timing tables to the exact profile (zero-wait-state memory; taken-branch refill P = 1–3 as bounds).
