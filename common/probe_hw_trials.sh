#!/bin/bash
# probe_hw_trials.sh -- runs on the RENTED GPU. Does probe-before-park (deploy_live.py default)
# fix the lock-in re-parking at max clock on a memory-bound phase, and does it keep the
# steady-workload wins? Entries label:args; 'base' = observe-only (default driver clocks).
cd "$(dirname "$0")/.."
REPS=2; RESULTS_DIR=probe_hw_trials
mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
POLS=("base:--lockin-k 0" "old_cap8_noprobe:--live --lockin-k 8 --lockin-max-slowdown 0.08 --no-lockin-probe"
      "old_cap8_probe:--live --lockin-k 8 --lockin-max-slowdown 0.08")
WORKLOADS=("phased:live_synthetic.py --mode phased:150" "membound:live_synthetic.py --mode membound:100"
           "resnet32:live_resnet50_train.py --batch-size 32:100")
for w in "${WORKLOADS[@]}"; do
    IFS=: read -r wname wcmd dur <<< "$w"
    pkill -f "[l]ive_synthetic|[l]ive_resnet50_train" 2>/dev/null; sleep 2
    (nohup python3 common/benchmarks/$wcmd --minutes 90 > "$RESULTS_DIR/workload_${wname}.log" 2>&1 &)
    sleep 10
    for rep in $(seq 1 $REPS); do
        for p in "${POLS[@]}"; do
            label="${p%%:*}"; args="${p#*:}"
            [ "$label" = "base" ] && [ "$rep" = "2" ] && [ "$wname" != "phased" ] && continue
            python3 -u common/deploy_live.py --weights "common/weights/$OLD" $args > "$RESULTS_DIR/ai_${wname}_${label}_rep${rep}.log" 2>&1 &
            pid=$!
            sleep "$dur"
            kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 4
        done
    done
done
pkill -f "[l]ive_synthetic|[l]ive_resnet50_train" 2>/dev/null; sleep 1
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
