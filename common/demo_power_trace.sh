#!/bin/bash
# demo_power_trace.sh -- runs on the RENTED GPU. Collects a clean, directly
# comparable power-over-time trace for a presentation graph: same workload,
# once pinned at max clock ("without" the controller) and once with the
# trained response-aware DDQN controlling clocks live ("with" the
# controller). Power is sampled independently at a fixed interval (not from
# deploy_live.py's own irregular tick timing) so both traces share the same
# clean time axis.
#
#   scp common/demo_power_trace.sh root@<host>:/root/
#   ssh root@<host> "cd /root/gpu-power-management-rl && bash /root/demo_power_trace.sh"
#   # then pull /root/demo_trace/ back and run common/analysis/plot_demo_trace.py locally

cd /root/gpu-power-management-rl
RESULTS_DIR=/root/demo_trace
mkdir -p "$RESULTS_DIR"
DURATION=90          # seconds per phase
SAMPLE_INTERVAL=2     # seconds between power samples
BATCH_SIZE=4           # launch-bound condition -- the clean, no-caveats result
CHECKPOINT="common/weights/response_aware_holdout_nvidia_geforce_rtx_4090.pth"

sample_power() {
    local outfile=$1
    local n=$(( DURATION / SAMPLE_INTERVAL ))
    local t0=$(date +%s.%N)
    echo "elapsed_s,power_w,clock_mhz" > "$outfile"
    for i in $(seq 1 "$n"); do
        now=$(date +%s.%N)
        elapsed=$(python3 -c "print(f'{$now-$t0:.2f}')")
        line=$(nvidia-smi --query-gpu=power.draw,clocks.sm --format=csv,noheader,nounits)
        power=$(echo "$line" | cut -d',' -f1 | tr -d ' ')
        clock=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
        echo "$elapsed,$power,$clock" >> "$outfile"
        sleep "$SAMPLE_INTERVAL"
    done
}

echo "=== starting workload (batch=$BATCH_SIZE) ==="
pkill -f live_resnet50_train 2>/dev/null
sleep 2
nohup python3 common/benchmarks/live_resnet50_train.py --minutes 10 --batch-size "$BATCH_SIZE" > "$RESULTS_DIR/workload.log" 2>&1 &
sleep 6

echo "=== phase 1: WITHOUT the controller (clocks pinned at max) ==="
nvidia-smi -lgc $(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits),$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits) > /dev/null 2>&1
sample_power "$RESULTS_DIR/without.csv"
nvidia-smi -rgc > /dev/null 2>&1
sleep 3

echo "=== phase 2: WITH the controller (trained DDQN, live) ==="
python3 -u common/deploy_live.py --weights "$CHECKPOINT" --live > "$RESULTS_DIR/deploy_live.log" 2>&1 &
AIPID=$!
sample_power "$RESULTS_DIR/with.csv"
kill -2 "$AIPID" 2>/dev/null
sleep 3
kill -9 "$AIPID" 2>/dev/null
nvidia-smi -rgc > /dev/null 2>&1

pkill -f live_resnet50_train 2>/dev/null
echo "DONE $(date)" > "$RESULTS_DIR/DONE.flag"
echo "=== done ==="
