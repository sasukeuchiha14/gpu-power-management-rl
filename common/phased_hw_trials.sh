#!/bin/bash
# phased_hw_trials.sh -- runs on the RENTED GPU. Does the cycle lock-in re-adapt when the workload
# changes phase? live_synthetic --mode phased alternates compute-bound (GEMM) and memory-bound
# phases every 30 s. 'base' is deploy_live in observe-only mode (default driver clocks).
cd "$(dirname "$0")/.."
REPS=2; AI_DURATION=150; RESULTS_DIR=phased_hw_trials
mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
POLS=("base:--lockin-k 0" "old:--live" "old_cap8:--live --lockin-k 8 --lockin-max-slowdown 0.08"
      "old_cap8_noshift:--live --lockin-k 8 --lockin-max-slowdown 0.08 --lockin-ms-shift 0")
pkill -f "[l]ive_synthetic" 2>/dev/null; sleep 2
(nohup python3 common/benchmarks/live_synthetic.py --mode phased --minutes 90 > "$RESULTS_DIR/workload.log" 2>&1 &)
sleep 10
for rep in $(seq 1 $REPS); do
    for p in "${POLS[@]}"; do
        label="${p%%:*}"; args="${p#*:}"
        python3 -u common/deploy_live.py --weights "common/weights/$OLD" $args > "$RESULTS_DIR/ai_${label}_rep${rep}.log" 2>&1 &
        pid=$!
        sleep "$AI_DURATION"
        kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 4
    done
done
pkill -f "[l]ive_synthetic" 2>/dev/null; sleep 1
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
