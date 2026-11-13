#!/bin/bash
# dwell_hw_trials.sh -- runs on the RENTED GPU. Hardware check of the dwell-window +
# hold-bonus candidate (reversal_cost=0.30, reversal_window=2, hold_bonus=0.02) against
# the old pure-log checkpoint and the reversal_cost=0.30 fix, on both Phase 2
# conditions (batch=4 launch-bound, batch=32 compute-bound), 3 training seeds of the
# new candidate, 2 interleaved reps each. Same protocol as reversal_cost_repeat_trials.sh.
cd "$(dirname "$0")/.."
REPS=2
AI_DURATION=90
RESULTS_DIR=dwell_hw_trials
mkdir -p "$RESULTS_DIR"
rm -f "$RESULTS_DIR"/DONE.flag

CKPTS=(
  "old:log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth"
  "rc030:reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth"
  "dwell_s0:dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090.pth"
  "dwell_s1:dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090_seed1.pth"
  "dwell_s2:dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090_seed2.pth"
)

run_baseline() {
    for i in $(seq 1 8); do
        nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader
        sleep 5
    done > "$RESULTS_DIR/baseline_$1.csv"
}

run_ai_trial() {
    python3 -u common/deploy_live.py --weights "common/weights/$1" --live > "$RESULTS_DIR/ai_$2.log" 2>&1 &
    local pid=$!
    sleep "$AI_DURATION"
    kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 1
}

for batch in 4 32; do
    pkill -f live_resnet50_train 2>/dev/null; sleep 2
    nohup python3 common/benchmarks/live_resnet50_train.py --minutes 60 --batch-size "$batch" > "$RESULTS_DIR/workload_b${batch}.log" 2>&1 &
    sleep 8
    run_baseline "b${batch}_pre"
    for rep in $(seq 1 $REPS); do
        for c in "${CKPTS[@]}"; do
            run_ai_trial "${c#*:}" "b${batch}_${c%%:*}_rep${rep}"
            sleep 3
        done
    done
    run_baseline "b${batch}_post"
    pkill -f live_resnet50_train 2>/dev/null; sleep 2
done
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
