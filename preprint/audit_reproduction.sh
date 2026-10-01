#!/usr/bin/env bash
# Independent reproduction of the preprint's results from a downloaded copy (the review bundle or the
# Zenodo archive). See preprint/AUDIT.md, part A.
#
#   preprint/audit_reproduction.sh                         # everything: paper-v3 (about 1.5 h on 10 cores, plus
#                                                          # about 20 min for the C port), then the twin arrays
#   preprint/audit_reproduction.sh costs table validation  # a quick trial of named steps, reported as partial;
#                                                          # "validation" is the twin-array part
#
# reproduce.sh compares each regenerated file with its version in git HEAD. A downloaded copy has no git
# history. check_reproduction.py then lists every file as "new", yet still ends with "all regenerated
# results match the committed ones", although nothing was compared. This script therefore
#   1. snapshots the folder as unpacked in a local git commit, when it is not a git repository of its own;
#   2. runs ./reproduce.sh (the paper-v3 results), then validation/reproduce_validation.sh (the twin arrays),
#      keeping the full output in preprint/audit/;
#   3. passes only if, in every part that ran, every result file was compared, none differs, and every number
#      check passed.
# It ends with a summary to paste into the audit report.
set -uo pipefail
cd "$(dirname "$0")/.."
root=$(pwd -P)

toplevel=$(git rev-parse --show-toplevel 2>/dev/null || true)
if [ -z "$toplevel" ] || [ "$(cd "$toplevel" && pwd -P)" != "$root" ]; then
  echo "no git history here: snapshotting the folder as unpacked (a local commit, used only for the comparison)"
  git init -q
  git add -A
  git -c user.name="audit snapshot" -c user.email="audit@localhost" commit -q -m "archive as unpacked, before reproduction"
elif [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "tracked files differ from the last commit, so the comparison would not be with the archive as unpacked:" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

# the arguments: reproduce.sh steps, plus "validation" for the twin arrays; none means everything
steps=()
run_paper=1
run_validation=1
if [ $# -gt 0 ]; then
  run_validation=0
  for a in "$@"; do
    if [ "$a" = validation ]; then run_validation=1; else steps+=("$a"); fi
  done
  [ ${#steps[@]} -eq 0 ] && run_paper=0
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p preprint/audit
log="preprint/audit/reproduction-$stamp.log"
part=$(mktemp)
{
  echo "audit reproduction, started $stamp"
  echo "machine: $(uname -s) $(uname -m); $(python3 --version 2>&1)"
  echo "snapshot compared against: $(git rev-parse HEAD)"
  echo "steps: ${*:-all}"
} | tee "$log"

count() { grep -c -E "^$1 " "$2" || true; }
ratio_ok() { [[ $1 =~ ^([0-9]+)/([0-9]+)\ checks ]] && [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[2]}" ]; }
verdict=PASS

paper_line="not run in this selection"
paper_numbers="not run"
cport="not run in this selection of steps"
environment="not checked"
rc_paper="-"
if [ $run_paper = 1 ]; then
  ./reproduce.sh ${steps[@]+"${steps[@]}"} 2>&1 | tee "$part"
  rc_paper=${PIPESTATUS[0]}
  cat "$part" >> "$log"
  identical=$(count identical "$part"); equal=$(count equal "$part")
  differs=$(count DIFFERS "$part"); uncompared=$(count new "$part")
  expected=$(python3 -c "import check_reproduction as c, pathlib; print(len(c.OUTPUTS) + len(list(pathlib.Path('results/logs').glob('*.log'))))")
  compared=$((identical + equal))
  paper_line="$compared of $expected compared ($identical identical, $equal equal within 1e-9); $differs differing; $uncompared not compared"
  paper_numbers=$(grep -E "checks of paper/comparison_tables.md passed" "$part" | tail -1 || true)
  environment=$(grep -E "^environment (matches|differs)" "$part" | tail -1 || true)
  if grep -q -E "^\[[0-9:]+\] firmware_cm4:" "$part"; then
    cport="ran"
  elif grep -q "skipped the C-port steps" "$part"; then
    cport="skipped (no xPack Arm GCC and QEMU in ~/opt; the committed C-port results were compared as they are)"
  fi
  if [ "$rc_paper" -ne 0 ] || [ "$differs" -ne 0 ] || [ "$uncompared" -ne 0 ] || [ "$compared" -ne "$expected" ] || ! ratio_ok "$paper_numbers"; then
    verdict=FAIL
  fi
fi

twin_line="not run in this selection"
twin_numbers="not run"
rc_twin="-"
if [ $run_validation = 1 ]; then
  validation/reproduce_validation.sh 2>&1 | tee "$part"
  rc_twin=${PIPESTATUS[0]}
  cat "$part" >> "$log"
  identical=$(count identical "$part"); equal=$(count equal "$part")
  differs=$(count DIFFERS "$part"); uncompared=$(count new "$part")
  expected=$(python3 -c "import sys; sys.path.insert(0, 'validation'); import check_validation_outputs as c; print(len(c.OUTPUTS))")
  compared=$((identical + equal))
  twin_line="$compared of $expected compared ($identical identical, $equal equal within 1e-9); $differs differing; $uncompared not compared"
  twin_numbers=$(grep -E "checks of validation/RESULTS_S2COMPOUNDS.md passed" "$part" | tail -1 || true)
  if [ "$rc_twin" -ne 0 ] || [ "$differs" -ne 0 ] || [ "$uncompared" -ne 0 ] || [ "$compared" -ne "$expected" ] || ! ratio_ok "$twin_numbers"; then
    verdict=FAIL
  fi
fi
rm -f "$part"

scope="full run"
[ $# -gt 0 ] && scope="PARTIAL run (steps: $*); the results of other steps were compared but not regenerated"
{
  echo
  echo "================ audit summary (paste into your report) ================"
  echo "scope:              $scope"
  echo "machine:            $(uname -s) $(uname -m); $(python3 --version 2>&1)"
  echo "environment:        ${environment:-not checked (reproduce.sh stopped before)}"
  echo "C-port steps:       $cport"
  echo "paper-v3 results:   $paper_line"
  echo "paper-v3 numbers:   ${paper_numbers:-none (reproduce.sh stopped before them)}"
  echo "twin-array results: $twin_line"
  echo "twin-array numbers: ${twin_numbers:-none (reproduce_validation.sh stopped before them)}"
  echo "exit codes:         reproduce.sh $rc_paper; reproduce_validation.sh $rc_twin"
  echo "result:             $verdict"
} | tee -a "$log"
echo "log:                $log (sha256 $(shasum -a 256 "$log" | cut -d' ' -f1))"
[ "$verdict" = PASS ]
