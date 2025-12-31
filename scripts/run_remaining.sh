#!/bin/bash
# Usage: ./run_remaining.sh 31

START=${1:-31}
END=38

for ((i=START; i<=END; i++)); do
  echo "=============================="
  echo "Running experiment $i"
  echo "=============================="
  bash ./scripts/run_ablation_mini.sh <<< "$i"
done

echo "All experiments from $START to $END completed."

