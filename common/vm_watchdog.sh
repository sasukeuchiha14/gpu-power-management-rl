#!/bin/bash
# vm_watchdog.sh -- periodic health check for any running vast.ai rental for
# this project. Meant to run from the local user's crontab every 10 minutes
# (needs local vastai CLI auth + ~/.ssh/vast_key, which a cloud-based
# scheduler cannot access -- see context.md if this comment seems odd,
# there's a reason this isn't a cloud routine).
#
# Quiet by design: writes nothing when nothing's running and nothing's wrong.
# Appends one line to LOGFILE only when something is actually worth flagging.
# Install:
#   crontab -e
#   */10 * * * * /home/govind/Documents/Capstone/gpu-power-management-rl/common/vm_watchdog.sh

VASTAI=~/.local/bin/vastai
SSH_KEY=~/.ssh/vast_key
LOGFILE=~/Documents/Capstone/gpu-power-management-rl/common/vm_watchdog.log
STUCK_PROC_MINS=30   # flag a single process alive this long with no corresponding workload

log() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $1" >> "$LOGFILE"; }

instances=$("$VASTAI" show instances --raw 2>/dev/null)
[ -z "$instances" ] && exit 0

running=$(echo "$instances" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for o in d:
    if o.get('actual_status') == 'running':
        print(o['id'], o.get('dph_total', 0), o.get('duration', 0))
" 2>/dev/null)

[ -z "$running" ] && exit 0

while read -r id dph duration; do
    [ -z "$id" ] && continue

    # flag: running for a long time (cost accumulates even if idle)
    hours=$(python3 -c "print(f'{$duration/3600:.1f}')" 2>/dev/null)
    cost=$(python3 -c "print(f'{$duration/3600*$dph:.2f}')" 2>/dev/null)
    if python3 -c "exit(0 if $duration > 3*3600 else 1)" 2>/dev/null; then
        log "WARNING instance $id has been running ${hours}h (~\$${cost} so far) -- confirm this is intended, stop if not."
    fi

    sshurl=$("$VASTAI" ssh-url "$id" 2>/dev/null)
    [ -z "$sshurl" ] && continue
    host=$(echo "$sshurl" | sed -E 's#ssh://root@([^:]+):.*#\1#')
    port=$(echo "$sshurl" | sed -E 's#.*:##')

    remote_check=$(timeout 15 ssh -i "$SSH_KEY" -p "$port" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 root@"$host" "
        echo '---clocks---'
        nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,utilization.gpu --format=csv,noheader 2>/dev/null
        echo '---longrunning---'
        ps -eo pid,etimes,comm --sort=-etimes 2>/dev/null | awk -v t=\$(($STUCK_PROC_MINS*60)) 'NR>1 && \$2>t {print}' | head -5
    " 2>/dev/null)

    if [ -z "$remote_check" ]; then
        log "WARNING instance $id: SSH unreachable (host may be stuck, or key/port stale)"
        continue
    fi

    clock_line=$(echo "$remote_check" | sed -n '/---clocks---/{n;p}')
    cur_clock=$(echo "$clock_line" | cut -d',' -f1 | tr -d ' MHz')
    max_clock=$(echo "$clock_line" | cut -d',' -f2 | tr -d ' MHz')
    util=$(echo "$clock_line" | cut -d',' -f4 | tr -d ' %')

    if [ -n "$cur_clock" ] && [ -n "$max_clock" ] && [ "$cur_clock" -lt "$max_clock" ] 2>/dev/null; then
        if [ -n "$util" ] && [ "$util" -lt 5 ] 2>/dev/null; then
            log "WARNING instance $id: GPU clock locked below max (${cur_clock}/${max_clock}MHz) while idle (util=${util}%) -- a deploy_live.py may have exited without resetting clocks."
        fi
    fi

    longrunning=$(echo "$remote_check" | sed -n '/---longrunning---/,$p' | tail -n +2)
    if [ -n "$longrunning" ]; then
        log "WARNING instance $id: long-running process(es) found (>${STUCK_PROC_MINS}min): $(echo "$longrunning" | tr '\n' ';')"
    fi

done <<< "$running"
