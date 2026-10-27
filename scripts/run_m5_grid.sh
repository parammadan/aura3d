#!/bin/bash
# EXP-M5-001 grid (see EXPERIMENT_REGISTRY.md). bash, not zsh: relies on word splitting.
cd "$(dirname "$0")/.."
cells=("0 0" "0.25 0" "0.5 0" "1 0" "2 0" "0 5" "0 10" "1 10" "3 20")
jobs=()
for c in "${cells[@]}"; do
  set -- $c
  for b in B1 B3cal; do
    jobs+=("$b $1 $2")
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P 4 -L 1 bash -c 'PYTHONPATH=src .venv/bin/python scripts/run_baselines.py --exp m5 --baseline $0 --noise nominal --seeds 2000:2050 --drift-rot $1 --drift-trans $2 > /dev/null 2>&1; echo done $0 $1 $2'
