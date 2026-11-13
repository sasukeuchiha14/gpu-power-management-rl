#!/bin/bash
# xfer_probe_check.sh -- RENTED GPU: does the time-aware probe cut the probing share on a slow workload?
# ResNet only, base + lock-in (time-aware probe), 2 reps x 100 s.
cd "$(dirname "$0")/.."
RESULTS_DIR=${1:-probe_time_trials}; mkdir -p "$RESULTS_DIR"; rm -f "$RESULTS_DIR"/DONE.flag
OLD=log_reward_response_aware_holdout_nvidia_geforce_rtx_4090.pth
pkill -f "[l]ive_resnet50_train" 2>/dev/null; sleep 2
(nohup python3 common/benchmarks/live_resnet50_train.py --batch-size 32 --minutes 30 > "$RESULTS_DIR/workload_resnet.log" 2>&1 &)
sleep 10
python3 -u common/deploy_live.py --weights common/weights/$OLD --lockin-k 0 > "$RESULTS_DIR/ai_resnet_base_rep1.log" 2>&1 &
pid=$!; sleep 100; kill -2 $pid; sleep 3; kill -9 $pid 2>/dev/null; sleep 4
for rep in 1 2; do
  python3 -u common/deploy_live.py --weights common/weights/$OLD --live --lockin-k 8 --lockin-max-slowdown 0.08 > "$RESULTS_DIR/ai_resnet_old_cap8_probe_rep${rep}.log" 2>&1 &
  pid=$!; sleep 100; kill -2 $pid; sleep 3; kill -9 $pid 2>/dev/null; sleep 4
done
pkill -f "[l]ive_resnet50_train" 2>/dev/null; sleep 1
nvidia-smi -rgc >/dev/null 2>&1
echo "ALL_DONE $(date)" > "$RESULTS_DIR/DONE.flag"
