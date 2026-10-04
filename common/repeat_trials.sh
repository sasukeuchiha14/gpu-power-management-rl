#!/bin/bash
# repeat_trials.sh -- runs on the RENTED GPU, not locally. Produced the 5 Oct 2026
# early-morning repeated-trials finding (context.md, NORTHSTAR row 11): the
# reward-plateau fix is a genuine trade-off, not an unqualified win -- fixes the
# compute-bound case it was built for, destabilises a launch-bound condition that
# was already fine. A single trial per condition (the entry just before this one
# in context.md) missed this; 3 reps each caught it.
#
# For each of batch=32 (compute-bound) and batch=4 (launch-bound): baseline
# sample, then 3 interleaved 90s reps of the OLD (linear-reward) and NEW
# (log-reward) checkpoints against common/benchmarks/live_resnet50_train.py,
# baseline sample again. Deliberately NOT parameterised (hardcoded checkpoint
# names/paths/durations) -- this was a one-off diagnostic run, not a reusable
# tool; common/analysis/repeat_trials_summary.py parses its output.
#
#   scp common/repeat_trials.sh root@<rented-host>:/root/
#   ssh root@<rented-host> "cd /root && bash repeat_trials.sh"
#   # then pull /root/repeat_trials/ back and run repeat_trials_summary.py locally
cd /root/gpu-power-management-rl
REPS=3
AI_DURATION=90
RESULTS_DIR=/root/repeat_trials
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

for batch in 32 4; do
    pkill -f live_resnet50_train 2>/dev/null
    sleep 2
    nohup python3 common/benchmarks/live_resnet50_train.py --minutes 30 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &
    sleep 6

    run_baseline "b${batch}_pre"

    for rep in 1 2 3; do
        run_ai_trial "response_aware_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_old_rep${rep}"
        sleep 3
        run_ai_trial "log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_new_rep${rep}"
        sleep 3
    done

    run_baseline "b${batch}_post"

    pkill -f live_resnet50_train 2>/dev/null
    sleep 2
done

echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
