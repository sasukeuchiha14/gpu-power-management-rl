#!/usr/bin/env python3
"""
gpu_probe.py -- does this GPU support the control we need? Run FIRST on every
machine (rented or physical) before spending time or money profiling it.

    python3 gpu_probe.py                 # probe + print verdict
    python3 gpu_probe.py --survey out.csv  # also append a row to a survey CSV

Answers in ~60 seconds:
  1. What GPU is this, and what are its NVML normalisation limits?
  2. Can we lock the CORE clock?      <- required; no lock = machine unusable
  3. Can we lock the MEMORY clock?    <- expected to fail on consumer cards
  4. Can we set the POWER limit?      <- expected to fail on consumer cards
  5. How many usable clock levels are there to tune over?

SAFETY: every write test is wrapped so clocks are ALWAYS reset, including on
exception or Ctrl+C. Nothing is left locked.
"""
import argparse, csv, json, os, subprocess, sys, time

try:
    import pynvml
except ImportError:
    sys.exit("pynvml missing.  pip install nvidia-ml-py")


# nvidia-smi exits 0 even when an operation is unsupported or unpermitted --
# it prints a warning and carries on. Exit code alone is NOT a success signal.
_FAIL_PHRASES = (
    "not supported", "insufficient permission", "does not have permission",
    "treating as warning", "not permitted", "unable to", "failed to",
)


def _smi(*args):
    """Run nvidia-smi; return (really_ok, combined_output)."""
    try:
        r = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, timeout=30)
        out = (r.stdout + r.stderr).strip()
        if r.returncode != 0:
            return False, out
        low = out.lower()
        if any(p in low for p in _FAIL_PHRASES):
            return False, out
        return True, out
    except Exception as e:
        return False, str(e)


def hardware_limits(h):
    """The normalisation constants. These make features comparable across GPUs."""
    lim = {}
    lim["max_core_mhz"] = float(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
    lim["max_mem_mhz"] = float(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
    try:
        lim["power_limit_w"] = pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    except pynvml.NVMLError:
        # Some consumer cards reject the NVML call but still report via CLI.
        ok, out = _smi("--query-gpu=power.max_limit", "--format=csv,noheader,nounits")
        try:
            lim["power_limit_w"] = float(out.splitlines()[0].strip()) if ok else float("nan")
        except (ValueError, IndexError):
            lim["power_limit_w"] = float("nan")
    try:
        lim["slowdown_temp_c"] = float(pynvml.nvmlDeviceGetTemperatureThreshold(
            h, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
    except pynvml.NVMLError:
        lim["slowdown_temp_c"] = 90.0
    lim["total_vram_mb"] = pynvml.nvmlDeviceGetMemoryInfo(h).total / (1024.0 * 1024.0)
    return lim


def supported_clocks(h):
    """Usable core levels at the TOP memory state (the only 3D-performance one)."""
    try:
        mems = sorted(pynvml.nvmlDeviceGetSupportedMemoryClocks(h), reverse=True)
    except pynvml.NVMLError:
        return [], []
    if not mems:
        return [], []
    try:
        cores = sorted(pynvml.nvmlDeviceGetSupportedGraphicsClocks(h, mems[0]))
    except pynvml.NVMLError:
        cores = []
    return mems, cores


def _first_reason(msg):
    """Pick the line that actually explains the failure."""
    for line in (msg or "").splitlines():
        low = line.lower()
        if any(p in low for p in _FAIL_PHRASES):
            return line.strip()
    return (msg or "failed").splitlines()[0].strip() if msg else "failed"


def _read_gpu_field(field):
    """Read back a live value via nvidia-smi -- used to verify a lock/limit
    actually applied, not just that the command claimed success."""
    ok, out = _smi(f"--query-gpu={field}", "--format=csv,noheader,nounits")
    if not ok or not out:
        return None
    try:
        return float(out.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def _wait_for_clock(field, target, tolerance, max_wait=4.0, poll_interval=0.5):
    """Poll `field` until it settles within `tolerance` of `target`, or give
    up after max_wait seconds. Replaces a single fixed-delay readback:
    empirically on one rented RTX 3060 (25 Aug 2026), the SAME lock command
    took less than 0.5s to settle on one run and more than 1s on the very
    next run of the identical script -- a fixed delay produced inconsistent
    false negatives no single number could reliably fix. Polling tolerates
    that variability directly. Returns (last_reading, matched)."""
    achieved = None
    waited = 0.0
    while waited < max_wait:
        time.sleep(poll_interval)
        waited += poll_interval
        achieved = _read_gpu_field(field)
        if achieved is not None and abs(achieved - target) <= tolerance:
            return achieved, True
    return achieved, False


def _get_persistence_mode():
    ok, out = _smi("--query-gpu=persistence_mode", "--format=csv,noheader")
    return out.strip().splitlines()[0].strip() if ok and out else "Unknown"


def _set_persistence_mode(enabled):
    _smi("-pm", "1" if enabled else "0")


def test_core_lock(h, limits):
    """Lock core to a mid-range value, verify the ACHIEVED clock actually
    moved (not just that the command reported success), then always reset.

    Polls via _wait_for_clock rather than checking once after a fixed delay:
    empirically on one rented RTX 3060 (25 Aug 2026), the SAME lock command
    took under 0.5s to settle on one run and over 1s on the very next run of
    the identical script -- no single fixed delay was reliable. A false NO
    here means rejecting a host that actually works fine."""
    target = int(limits["max_core_mhz"] * 0.55)
    try:
        ok, msg = _smi("-i", "0", "-lgc", f"{target},{target}")
        if not ok:
            return False, _first_reason(msg)
        tolerance = max(50.0, target * 0.1)
        achieved, matched = _wait_for_clock("clocks.current.sm", target, tolerance)
        if achieved is None:
            return False, "lock reported success but clock unreadable"
        if not matched:
            return False, f"reported success but clock reads {achieved:.0f} MHz, not ~{target} MHz after retrying -- likely a silent no-op"
        return True, f"locked to {target} MHz (confirmed: {achieved:.0f} MHz)"
    finally:
        _smi("-i", "0", "-rgc")          # never leave the GPU locked


def test_mem_lock(h, limits):
    """Same verify-don't-trust, poll-don't-guess approach as test_core_lock.
    Also matters here for a second reason: nvidia-smi -lmc reports success
    and prints the new target even when persistence mode is off and the
    change silently never takes effect -- caught empirically on a rented
    RTX 4070 (23 Aug 2026). main() enables persistence mode before calling
    this for exactly that reason; this readback check is a second,
    independent line of defense."""
    target = int(limits["max_mem_mhz"])
    try:
        ok, msg = _smi("-i", "0", "-lmc", f"{target},{target}")
        if not ok:
            return False, _first_reason(msg)
        tolerance = max(50.0, target * 0.05)
        achieved, matched = _wait_for_clock("clocks.current.memory", target, tolerance)
        if achieved is None:
            return False, "lock reported success but clock unreadable"
        if not matched:
            return False, f"reported success but clock reads {achieved:.0f} MHz, not ~{target} MHz after retrying -- likely a silent no-op (needs persistence mode)"
        return True, f"locked to {target} MHz (confirmed: {achieved:.0f} MHz)"
    finally:
        _smi("-i", "0", "-rmc")


def test_power_limit(h, limits):
    """Same verify-don't-trust approach -- nvidia-smi -pl has the identical
    silent-no-op-without-persistence-mode behaviour as -lmc."""
    cur = limits["power_limit_w"]
    if cur != cur:                        # NaN
        return False, "power limit unreadable"
    target = int(cur * 0.9)
    try:
        ok, msg = _smi("-i", "0", "-pl", str(target))
        if not ok:
            return False, _first_reason(msg)
        tolerance = max(2.0, target * 0.05)
        achieved, matched = _wait_for_clock("power.limit", target, tolerance)
        if achieved is None:
            return False, "set reported success but limit unreadable"
        if not matched:
            return False, f"reported success but limit reads {achieved:.0f} W, not ~{target} W after retrying -- likely a silent no-op (needs persistence mode)"
        return True, f"set to {target} W (confirmed: {achieved:.0f} W)"
    finally:
        _smi("-i", "0", "-pl", str(int(cur)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", help="append a row to this CSV")
    args = ap.parse_args()

    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
    except pynvml.NVMLError as e:
        print("=" * 62)
        print(f"  No NVIDIA GPU visible to this process: {e}")
        print("  On a rented container this usually means GPU passthrough")
        print("  wasn't enabled for this instance -- check the host's docs,")
        print("  or try a different offer. Nothing was tested; nothing to")
        print("  reset.")
        print("=" * 62)
        return 1
    name = pynvml.nvmlDeviceGetName(h)
    if isinstance(name, bytes):
        name = name.decode()
    driver = pynvml.nvmlSystemGetDriverVersion()
    if isinstance(driver, bytes):
        driver = driver.decode()

    limits = hardware_limits(h)
    mems, cores = supported_clocks(h)

    print("=" * 62)
    print(f"  {name}")
    print(f"  driver {driver}   |   {os.name}   |   {sys.platform}")
    print("=" * 62)
    print(f"  max core        : {limits['max_core_mhz']:.0f} MHz")
    print(f"  max memory      : {limits['max_mem_mhz']:.0f} MHz")
    print(f"  power limit     : {limits['power_limit_w']:.0f} W")
    print(f"  slowdown temp   : {limits['slowdown_temp_c']:.0f} C")
    print(f"  VRAM            : {limits['total_vram_mb']:.0f} MB")
    print(f"  core levels     : {len(cores)}"
          + (f"  ({min(cores)}-{max(cores)} MHz)" if cores else ""))
    print(f"  memory states   : {len(mems)}  {mems[:6]}")
    print("-" * 62)

    # Persistence mode matters: nvidia-smi -lmc/-pl both report success and
    # print the requested target even when persistence mode is off and the
    # change silently never applies (caught on a rented RTX 4070, 23 Aug
    # 2026 -- see the docstrings on test_mem_lock/test_power_limit). -lgc is
    # unaffected. Enable it for the duration of the tests, restore whatever
    # it was set to afterward -- this is someone else's rented machine.
    orig_persistence = _get_persistence_mode()
    _set_persistence_mode(True)
    try:
        core_ok, core_msg = test_core_lock(h, limits)
        mem_ok, mem_msg = test_mem_lock(h, limits)
        pl_ok, pl_msg = test_power_limit(h, limits)
    finally:
        _set_persistence_mode(orig_persistence.strip().lower() == "enabled")

    mark = lambda b: "YES" if b else "NO "
    print(f"  core clock lock : {mark(core_ok)}  {core_msg}")
    print(f"  mem clock lock  : {mark(mem_ok)}  {mem_msg}")
    print(f"  power limit set : {mark(pl_ok)}  {pl_msg}")
    print("-" * 62)

    # `cores` comes from a separate NVML enumeration call that can fail even
    # when locking itself (test_core_lock, tested directly via nvidia-smi)
    # works fine -- e.g. under some virtualised/passthrough setups. Don't let
    # a failed enumeration alone downgrade a GPU that actually took the lock.
    usable = core_ok and (len(cores) >= 8 or not cores)
    if usable and not cores:
        print("  VERDICT: USABLE -- core lock works. (Could not enumerate the")
        print("           full level list, but the control path is live.)")
    elif usable:
        print("  VERDICT: USABLE -- core control works. Proceed to profiling.")
    elif core_ok:
        print(f"  VERDICT: MARGINAL -- core lock works but only {len(cores)} levels.")
    else:
        print("  VERDICT: UNUSABLE -- cannot lock core clock. Do not rent/profile.")
        print("           (On a rented box this usually means the container")
        print("            lacks privileges. Try a different host.)")
    print("=" * 62)

    out = {
        "gpu_name": name, "driver": driver, "platform": sys.platform,
        "hardware_limits": limits,
        "core_levels": len(cores),
        "core_min_mhz": min(cores) if cores else None,
        "core_max_mhz": max(cores) if cores else None,
        "memory_states": len(mems),
        "core_lock": core_ok, "mem_lock": mem_ok, "power_limit": pl_ok,
        "usable": usable,
        # True from this version onward: lock/limit results below are
        # confirmed by reading the achieved value back, not just trusting
        # nvidia-smi's own success message (which lies for -lmc/-pl without
        # persistence mode -- see test_mem_lock's docstring). Rows without
        # this column predate the fix.
        "verified_readback": True,
    }
    with open("gpu_capability.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote gpu_capability.json")

    if args.survey:
        new = not os.path.exists(args.survey)
        with open(args.survey, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out.keys()))
            if new:
                w.writeheader()
            flat = dict(out)
            flat["hardware_limits"] = json.dumps(limits)
            w.writerow(flat)
        print(f"appended survey row -> {args.survey}")

    pynvml.nvmlShutdown()
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
