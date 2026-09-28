#!/usr/bin/env bash
# Regenerates every result behind paper/comparison_tables.md, then checks that the results match the
# committed ones (check_reproduction.py) and that every number in the document follows from them
# (check_paper_numbers.py). Takes about 1.5 hours on 10 cores.
#
#   ./reproduce.sh                 # all steps, in order
#   ./reproduce.sh confound table  # only the named steps (later steps read earlier results), then the checks
#   ALLOW_ENV_MISMATCH=1 ./reproduce.sh   # run with other package versions (results may differ)
#
# Each step's output goes to results/logs/<step>.log, with run times blanked so logs compare exactly.
set -euo pipefail
cd "$(dirname "$0")"

STEPS=(tune budget literature improve combine confound table costs presence presence_nobias presence_tracking energy)
step_command() {
  case $1 in
    tune)              echo "tune_system2.py" ;;                               # validation choice of System 2
    budget)            echo "sweep_system2_budget.py --extra-seeds 10" ;;      # sequential protocol, 15 seeds
    literature)        echo "benchmark_literature_comparison.py" ;;            # System 2, Settings 1-2 + sequential
    improve)           echo "improve_system2.py" ;;                            # candidate improvements
    combine)           echo "combine_system2.py" ;;                            # System 2+ (reads improve's CSV)
    confound)          echo "check_temporal_confound.py" ;;                    # section 4
    table)             echo "literature_table.py" ;;                           # section 3 (reads combine's CSV)
    costs)             echo "deep_model_cost.py" ;;                            # section 5
    presence)          echo "presence_test.py" ;;                              # presence detector, LED bias 0.5 µA
    presence_nobias)   echo "presence_test.py --led-bias 0 --out results/presence_test_nobias.csv" ;;
    presence_tracking) echo "presence_test.py --led-bias 0 --tracking --out results/presence_test_tracking.csv" ;;
    energy)            echo "energy_proxy.py --skip-classification" ;;         # presence power (reads the 3 CSVs)
    *) echo "unknown step: $1 (steps: ${STEPS[*]})" >&2; return 1 ;;
  esac
}

if ! python3 check_reproduction.py env; then
  [ "${ALLOW_ENV_MISMATCH:-0}" = 1 ] || { echo "install requirements.txt, or set ALLOW_ENV_MISMATCH=1" >&2; exit 1; }
fi
if [ ! -f data/features.csv ] || [ ! -f data/targets.csv ]; then
  python3 fetch_data.py
fi
shasum -a 256 -c data/SHA256SUMS

selected=("${@:-${STEPS[@]}}")
for s in "${selected[@]}"; do step_command "$s" > /dev/null; done
mkdir -p results/logs
for s in "${selected[@]}"; do
  cmd=$(step_command "$s")
  echo "[$(date +%H:%M:%S)] $s: python3 $cmd"
  # shellcheck disable=SC2086
  if ! python3 -W ignore $cmd > "results/logs/$s.raw" 2>&1; then
    tail -20 "results/logs/$s.raw" >&2
    exit 1
  fi
  sed -E 's/\[[0-9]+ s\]/[… s]/g' "results/logs/$s.raw" > "results/logs/$s.log"
  rm "results/logs/$s.raw"
done

echo "[$(date +%H:%M:%S)] checks"
status=0
python3 check_reproduction.py results || status=1
python3 -W ignore check_paper_numbers.py || status=1
exit $status
