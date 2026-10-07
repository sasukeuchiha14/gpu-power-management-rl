#!/bin/bash
# reversal_cost_repeat_trials.sh -- runs on the RENTED GPU, not locally. Confirms
# on real hardware whether the reversal_cost=0.30 fix (context.md, NORTHSTAR
# row 11, 7 Oct 2026) actually resolves the limit cycle observed on real
# hardware -- offline validation is solid (5-GPU/10-seed, cycle rate down
# from ~0.3-0.55 to ~0.01-0.08 per GPU) but every other reward-shaping
# change in this investigation has been hardware-confirmed before being
# trusted, and this one hasn't yet.
#
# Compares the OLD pure-log-reward checkpoint (known to cycle, the original
# problem) against the NEW reversal_cost=0.30 checkpoint (the fix), on both
# conditions used throughout Phase 2: batch=4 (launch-bound) and batch=32
# (compute-bound, the condition log-reward was built for -- included so the
# fix isn't validated on one condition while quietly regressing the other).
# Same interleaved-reps protocol as repeat_trials.sh and
# debounce_repeat_trials.sh (both superseded investigations of the same
# problem) -- single trials and even small seed counts have misled
# repeatedly this session.
#
#   scp common/reversal_cost_repeat_trials.sh root@<rented-host>:/root/gpu-power-management-rl/common/
#   ssh root@<rented-host> "cd /root/gpu-power-management-rl && bash common/reversal_cost_repeat_trials.sh"
#   # then pull /root/gpu-power-management-rl/reversal_cost_repeat_trials/ back and
#   # run common/analysis/reversal_cost_repeat_trials_summary.py locally
cd "$(dirname "$0")/.."
REPS=3
AI_DURATION=90
RESULTS_DIR=reversal_cost_repeat_trials
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/DONE.flag

run_baseline() {
    local label=$1
    for i in $(seq 1 8); do
        nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader
        sleep 5
    done > "$RESULTS_DIR/baseline_${label}.csv"
}

run_ai_trial() {
    local ckpt=$1
    local label=$2
    python3 -u common/deploy_live.py --weights "common/weights/${ckpt}" --live > "$RESULTS_DIR/ai_${label}.log" 2>&1 &
    local pid=$!
    sleep "$AI_DURATION"
    kill -2 "$pid" 2>/dev/null
    sleep 3
    kill -9 "$pid" 2>/dev/null
    sleep 1
}

for batch in 4 32; do
    pkill -f live_resnet50_train 2>/dev/null
    sleep 2
    nohup python3 common/benchmarks/live_resnet50_train.py --minutes 30 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &
    sleep 6

    run_baseline "b${batch}_pre"

    for rep in $(seq 1 $REPS); do
        run_ai_trial "log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_old_rep${rep}"
        sleep 3
        run_ai_trial "reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_new_rep${rep}"
        sleep 3
    done

    run_baseline "b${batch}_post"

    pkill -f live_resnet50_train 2>/dev/null
    sleep 2
done

echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
