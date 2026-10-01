#!/usr/bin/env bash
# Regenerates the twin-array validation: SOLAR+ on five independent sensor boards (UCI 361). The manuscript
# reports it in its section on independent sensor boards. The script compares every regenerated file with
# its committed version and checks every number of validation/RESULTS_S2COMPOUNDS.md.
#
# Takes about 1 minute on 10 cores, plus the 204 MB download on the first run.
#
#   validation/reproduce_validation.sh
#
# Run it from a git checkout. From a downloaded copy, use preprint/audit_reproduction.sh: it snapshots the
# copy in git first and runs this script after the paper-v3 steps.
set -euo pipefail
cd "$(dirname "$0")/.."

# UCI 361, checksum-verified. The 204 MB download can break off; fetch_twin_arrays.py writes straight to its
# target, so a broken download leaves a partial file that then fails the checksum. Remove it and try again.
archive=data/twin_arrays/raw/uci_361.zip
err=$(mktemp)
for attempt in 1 2 3; do
  if python3 fetch_twin_arrays.py > /dev/null 2> "$err"; then
    break
  fi
  if [ "$attempt" = 3 ]; then
    cat "$err" >&2
    echo "could not get a verified copy of the twin arrays from UCI after 3 attempts; check the connection and rerun" >&2
    exit 1
  fi
  echo "twin-array download or checksum failed (attempt $attempt of 3): $(tail -1 "$err"); retrying" >&2
  rm -f "$archive"
  sleep 10
done
rm -f "$err"
mkdir -p results/logs
for step in "twin_arrays:twin_arrays.py" "system2_compounds:system2_compounds.py confirm"; do
  name=${step%%:*}
  cmd=${step#*:}
  echo "[$(date +%H:%M:%S)] $name: python3 $cmd"
  # shellcheck disable=SC2086
  if ! python3 -W ignore $cmd > "results/logs/$name.raw" 2>&1; then
    tail -20 "results/logs/$name.raw" >&2
    exit 1
  fi
  # as reproduce.sh: run times and the host SDK are blanked so that logs compare exactly
  sed -E -e 's/\[[0-9]+ s\]/[… s]/g' -e 's/ \(SDK [^)]*\)//' "results/logs/$name.raw" > "results/logs/$name.log"
  rm "results/logs/$name.raw"
done

echo "[$(date +%H:%M:%S)] checks"
rc=0
python3 validation/check_validation_outputs.py || rc=1
python3 validation/check_validation_numbers.py || rc=1
exit $rc
