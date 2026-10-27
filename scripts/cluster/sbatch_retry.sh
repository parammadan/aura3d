#!/bin/bash
# sbatch_retry.sh <unique-job-name> <sbatch args...>   Idempotent: never submits twice under the same name.
name=$1; shift
for i in 1 2 3 4 5 6; do
  id=$(squeue -u "$USER" -h -n "$name" -o %i | head -1)
  [ -n "$id" ] && { echo "$id"; exit 0; }
  out=$(sbatch --parsable -J "$name" "$@" 2>&1) && { echo "$out"; exit 0; }
  sleep $((10 * i))
done
echo "FAILED: $out" >&2; exit 1
