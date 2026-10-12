#!/bin/bash
# Runs each experiment in its own fresh Python process so MPS memory is
# fully reclaimed by the OS between experiments (avoids the fragmentation-
# driven OOM that persists even with in-process gc/empty_cache cleanup).
set -uo pipefail

EXPERIMENTS=(
  exp1_bm25_ar exp1_labse_ar exp1_me5_ar exp1_me5large_ar exp1_bgem3_ar
  exp1_bm25_ms exp1_labse_ms exp1_me5_ms exp1_me5large_ms exp1_bgem3_ms
  exp2_ar_mono exp2_ar_cross exp2_ms_mono exp2_ms_cross
  exp3_base_ar exp3_lora_ar exp3_base_ms exp3_lora_ms
  exp4_k3 exp4_k5 exp4_k10
  exp5_ar_to_ms exp5_ms_to_ar
  exp2_ar_mono_me5large exp2_ar_cross_me5large exp2_ms_mono_me5large exp2_ms_cross_me5large
  exp5_ar_to_ms_me5large exp5_ms_to_ar_me5large
)

LOG=isolated_rerun_log.txt
echo "=== Isolated rerun started $(date) ===" >> "$LOG"

for name in "${EXPERIMENTS[@]}"; do
  echo "" >> "$LOG"
  echo "########## $name : starting $(date) ##########" >> "$LOG"
  python3 run_all_experiments.py --exp "$name" --skip-existing >> "$LOG" 2>&1
  status=$?
  echo "########## $name : exit=$status finished $(date) ##########" >> "$LOG"
  if [ $status -ne 0 ]; then
    echo "!!!!!!!!!! $name FAILED — continuing with next experiment !!!!!!!!!!" >> "$LOG"
  fi
  sleep 5
done

echo "=== Isolated rerun finished $(date) ===" >> "$LOG"
