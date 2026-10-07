#!/bin/bash
# reversal_cost_low_repeat_trials.sh -- runs on the RENTED GPU, not locally.
# Follow-up to reversal_cost_repeat_trials.sh's hardware confirmation
# (context.md, NORTHSTAR row 11, 7 Oct 2026): reversal_cost=0.30 completely
# eliminated the limit cycle on real hardware (flip_rate 0.000 in every rep,
# both conditions) but on the compute-bound condition (batch=32) the cost
# was severe enough to erase the entire power-saving benefit (new: -1.5%,
# i.e. net WORSE than doing nothing; old: +7.9%). This tests whether a
# LESS aggressive reversal_cost (0.10, 0.20 -- offline looked less
# conservative, see reversal_cost_validate.py) lands in a better real
# spot: still meaningfully suppresses cycling while keeping real power
# savings, rather than fully eliminating the cycle at the cost of the
# whole benefit.
#
# Compares OLD (pure-log, cycles) against reversal_cost=0.10 and
# reversal_cost=0.20 on both conditions, same interleaved 3-rep protocol.
#
#   scp common/reversal_cost_low_repeat_trials.sh root@<rented-host>:/root/gpu-power-management-rl/common/
#   ssh root@<rented-host> "cd /root/gpu-power-management-rl && bash common/reversal_cost_low_repeat_trials.sh"
#   # then pull /root/gpu-power-management-rl/reversal_cost_low_repeat_trials/ back and
#   # run common/analysis/reversal_cost_low_repeat_trials_summary.py locally
cd "$(dirname "$0")/.."
REPS=3
AI_DURATION=90
RESULTS_DIR=reversal_cost_low_repeat_trials
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
    nohup python3 common/benchmarks/live_resnet50_train.py --minutes 40 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &
    sleep 6

    run_baseline "b${batch}_pre"

    for rep in $(seq 1 $REPS); do
        run_ai_trial "log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_old_rep${rep}"
        sleep 3
        run_ai_trial "reversal_cost_010_fix_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_r010_rep${rep}"
        sleep 3
        run_ai_trial "reversal_cost_020_fix_holdout_nvidia_geforce_rtx_4090.pth" "b${batch}_r020_rep${rep}"
        sleep 3
    done

    run_baseline "b${batch}_post"

    pkill -9 -f live_resnet50_train
    sleep 2
done

echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
