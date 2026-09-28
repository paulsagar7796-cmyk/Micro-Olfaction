# Olfactory simulation

Drift-robust gas classification on the UCI Gas Sensor Array Drift dataset ("System 2+": running
feature alignment and an online linear readout, calibrated with a few labelled samples per gas), plus
a simulated MOx sensor front end with a spiking presence detector. The results for the paper are in
[paper/comparison_tables.md](paper/comparison_tables.md).

## Reproducing the paper's numbers

```bash
python3 -m pip install -r requirements.txt   # exact versions; Python 3.13.2
./reproduce.sh                               # ≈1.5 h on 10 cores
```

`reproduce.sh` downloads the dataset if it is missing (`fetch_data.py`) and verifies it against
`data/SHA256SUMS`. It then reruns every script behind the paper and writes to `results/`, with logs in
`results/logs/`. Finally it checks two things:

- `check_reproduction.py` compares every regenerated file with the committed version.
- `check_paper_numbers.py` recomputes each number in `paper/comparison_tables.md` from `results/`.

`./reproduce.sh confound table` reruns only the named steps; the step list is at the top of the script.

With the pinned versions, every file should come out identical. Stochastic-gradient training is
sensitive to floating-point details, so other library versions can change individual results. The
seed spreads reported in the paper indicate how large such changes can be.

## Layout

| Files | Purpose |
|---|---|
| `fetch_data.py`, `dataset_adapter.py` | download and load the UCI dataset (gas naming explained in the paper, section 1) |
| `tune_system2.py`, `sweep_system2_budget.py`, `benchmark_literature_comparison.py` | System 2: tuning on Batches 2–5, label budgets, Settings 1 and 2 |
| `improve_system2.py`, `combine_system2.py` | System 2+: candidate improvements and their combination |
| `literature_table.py`, `literature/published_results.csv` | published results and the unified comparison |
| `check_temporal_confound.py` | file-order and concentration-level hold-out checks |
| `deep_model_cost.py` | compute and memory estimates for all compared methods |
| `snn_input.py`, `lif_neurons.py`, `presence_test.py`, `energy_proxy.py` | simulated sensor front end and presence detector |
| other scripts, older files in `results/` | exploration before the gas naming was harmonised; not used by the paper |
