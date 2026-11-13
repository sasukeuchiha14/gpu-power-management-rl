#!/bin/bash
# lockin_hw_trials.sh -- runs on the RENTED GPU. Hardware test of deploy_live.py --lockin-k
# (park at the lowest-live-EDP level once the clock targets form a short limit cycle).
# Entries are label:weights:extra-args.
cd "$(dirname "$0")/.."
REPS=2
AI_DURATION=90
RESULTS_DIR=lockin_hw_trials
mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
RC=reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth
D1=dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090_seed1.pth
CKPTS=("old:$OLD:" "old_lock:$OLD:--lockin-k 8" "rc030:$RC:" "dwell_s1_lock:$D1:--lockin-k 8")

run_baseline() {
    for i in $(seq 1 8); do
        nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader; sleep 5
    done > "$RESULTS_DIR/baseline_$1.csv"
}
run_ai_trial() {   # weights extra label
    python3 -u common/deploy_live.py --weights "common/weights/$1" $2 --live > "$RESULTS_DIR/ai_$3.log" 2>&1 &
    local pid=$!
    sleep "$AI_DURATION"
    kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 1
}
for batch in 4 32; do
    pkill -f "[l]ive_resnet50_train" 2>/dev/null; sleep 2
    (nohup python3 common/benchmarks/live_resnet50_train.py --minutes 60 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &)
    sleep 8
    run_baseline "b${batch}_pre"
    for rep in $(seq 1 $REPS); do
        for c in "${CKPTS[@]}"; do
            IFS=: read -r label w extra <<< "$c"
            run_ai_trial "$w" "$extra" "b${batch}_${label}_rep${rep}"; sleep 3
        done
    done
    run_baseline "b${batch}_post"
    pkill -f "[l]ive_resnet50_train" 2>/dev/null; sleep 2
done
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
