#!/bin/bash
# reversal_cost_010_multiseed_trials.sh -- runs on the RENTED GPU, not locally.
# Round 3 of hardware confirmation (context.md, NORTHSTAR row 11, 7 Oct 2026).
#
# Round 2 found reversal_cost=0.10's seed=0 checkpoint does NOT suppress
# cycling on batch=32 (compute-bound) on real hardware (flip_rate identical
# to the uncycled baseline), despite a strong offline reduction across the
# full fleet. reversal_cost=0.20 was dropped after its seed=1/2 checkpoints
# were found offline to regress to the UNSAFE settling point (0.737, same
# as the pre-log-reward checkpoint) -- a real seed-instability problem not
# worth spending more rental time confirming.
#
# This tests whether reversal_cost=0.10's batch=32 result was itself a
# seed=0 artifact: seed=1 and seed=2 checkpoints (both settle safely at
# 0.866 offline, unlike 0.20's seeds) against old, both conditions.
#
#   scp common/reversal_cost_010_multiseed_trials.sh root@<rented-host>:/root/gpu-power-management-rl/common/
#   ssh root@<rented-host> "cd /root/gpu-power-management-rl && bash common/reversal_cost_010_multiseed_trials.sh"
cd "$(dirname "$0")/.."
REPS=2
AI_DURATION=90
RESULTS_DIR=reversal_cost_010_multiseed_trials
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/DONE.flag

run_baseline() {
    local label=$1
    for i in $(seq 1 6); do
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
    nohup python3 common/benchmarks/live_resnet50_train.py --minutes 30 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &
    sleep 6

    run_baseline "b${batch}_pre"

    for rep in $(seq 1 $REPS); do
        run_ai_trial "log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_old_rep${rep}"
        sleep 3
        run_ai_trial "reversal_cost_010_fix_holdout_nvidia_geforce_rtx_4090_seed1.pth" "b${batch}_r010s1_rep${rep}"
        sleep 3
        run_ai_trial "reversal_cost_010_fix_holdout_nvidia_geforce_rtx_4090_seed2.pth" "b${batch}_r010s2_rep${rep}"
        sleep 3
    done

    run_baseline "b${batch}_post"

    pkill -9 -f live_resnet50_train
    sleep 2
done

echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
