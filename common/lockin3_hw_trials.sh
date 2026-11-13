#!/bin/bash
# lockin3_hw_trials.sh -- runs on the RENTED GPU. Generalisation test for deploy_live.py's cycle
# lock-in: 3 live workloads (compute-bound GEMM, memory-bound elementwise, ResNet-50 train b=32),
# 5 policies incl. the slowdown-capped lock-in, 2 interleaved reps. Entries: label:weights:extra.
cd "$(dirname "$0")/.."
REPS=2; AI_DURATION=90; RESULTS_DIR=lockin3_hw_trials
mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
RC=reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth
D1=dwell_win2_hold02_holdout_nvidia_geforce_rtx_4090_seed1.pth
CKPTS=("old:$OLD:" "old_lock:$OLD:--lockin-k 8" "old_cap8:$OLD:--lockin-k 8 --lockin-max-slowdown 0.08"
       "rc030_lock:$RC:--lockin-k 8" "dwell_s1_lock:$D1:--lockin-k 8")
WORKLOADS=("matmul:live_synthetic.py --mode matmul" "membound:live_synthetic.py --mode membound"
           "resnet32:live_resnet50_train.py --batch-size 32")

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
for w in "${WORKLOADS[@]}"; do
    wname="${w%%:*}"; wcmd="${w#*:}"
    pkill -f "[l]ive_synthetic|[l]ive_resnet50_train" 2>/dev/null; sleep 2
    (nohup python3 common/benchmarks/$wcmd --minutes 60 > "$RESULTS_DIR/workload_${wname}.log" 2>&1 &)
    sleep 10
    run_baseline "${wname}_pre"
    for rep in $(seq 1 $REPS); do
        for c in "${CKPTS[@]}"; do
            IFS=: read -r label wt extra <<< "$c"
            run_ai_trial "$wt" "$extra" "${wname}_${label}_rep${rep}"; sleep 3
        done
    done
    run_baseline "${wname}_post"
    pkill -f "[l]ive_synthetic|[l]ive_resnet50_train" 2>/dev/null; sleep 2
done
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
