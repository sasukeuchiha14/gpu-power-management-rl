#!/bin/bash
# gpu_xfer_trials.sh -- runs on a RENTED GPU of a different model than the 4090 the checkpoint was
# held out on. Does deploy_live.py's lock-in transfer? 3 live workloads x {base (observe-only),
# old, old+lock-in(cap 8%, probe)}, 2 reps, plus one longer lock-in run on the steady ResNet workload.
# Usage: bash common/gpu_xfer_trials.sh <resnet-batch-size> <results-dir>
cd "$(dirname "$0")/.."
BS=${1:-32}; RESULTS_DIR=${2:-xfer_hw_trials}; REPS=2
mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
POLS=("base:--lockin-k 0" "old:--live" "old_cap8_probe:--live --lockin-k 8 --lockin-max-slowdown 0.08")
WORKLOADS=("phased:live_synthetic.py --mode phased:150" "membound:live_synthetic.py --mode membound:100"
           "resnet:live_resnet50_train.py --batch-size $BS:100")
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
            pid=$!; sleep "$dur"; kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 4
        done
    done
done
# longer steady run (stability over time)
python3 -u common/deploy_live.py --weights "common/weights/$OLD" --live --lockin-k 8 --lockin-max-slowdown 0.08 > "$RESULTS_DIR/ai_resnet_long_old_cap8_probe_rep1.log" 2>&1 &
pid=$!; sleep 420; kill -2 "$pid" 2>/dev/null; sleep 3; kill -9 "$pid" 2>/dev/null; sleep 3
pkill -f "[l]ive_synthetic|[l]ive_resnet50_train" 2>/dev/null; sleep 1
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
