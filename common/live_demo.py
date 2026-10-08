"""
live_demo.py -- a LIVE, watchable terminal dashboard for showing the DVFS
controller's energy savings to an audience in real time. Different from
demo_power_trace.sh (which records two fixed 90s phases and plots a static
chart afterward, docs/demo_power_trace.html): this redraws a live chart in
the terminal every tick, continuously, so it can be screen-shared during a
live demo and watched update as it happens -- no post-processing step.

Runs on whatever machine has the GPU and clock-lock permission (almost
always the RENTED GPU, not a laptop -- see the permission note below).
Alternates automatically between two phases, forever, until Ctrl+C:

  WITHOUT  -- clocks pinned at max (no management, the "before" case)
  WITH     -- the trained response-aware DDQN controlling clocks live
              (reuses deploy_live.py's LiveController directly, not a
              reimplementation, so the demo is showing the real controller)

Rendering: a braille sub-pixel canvas (the same technique btop uses for its
graphs -- each terminal cell holds a 2x4 dot grid via Unicode Braille
Patterns, giving ~8x the resolution of block-character bars) drawn with
`rich` for panels, borders, and flicker-free live redraw (`rich.live.Live`
diffs the screen instead of clearing it every tick). One extra dependency
(`pip install rich`) versus the previous pure-ANSI version, chosen because
`rich` is mature and API-stable (unlike the terminal-charting library tried
first, which had an incompatible API on the test box) and because the
visual quality difference for something shown to an audience is real.

Each tick samples real NVML power, keeps a running "energy saved so far"
statistic computed from the actual measured phase means.

Usage (on the GPU machine, same checklist as every other live run in this
project -- gpu_probe.py USABLE check first):

    pip install rich

    # terminal 1
    python3 common/benchmarks/live_resnet50_train.py --minutes 60 --batch-size 4

    # terminal 2 (the live demo -- this is what you screen-share)
    python3 common/live_demo.py --weights common/weights/reversal_cost_030_fix_holdout_nvidia_geforce_rtx_4090.pth --live

Permission note: needs the same clock-lock permission every other script in
this project needs (gpu_probe.py USABLE) -- a laptop's own integrated/
consumer GPU usually doesn't have this on a managed OS, and a rented
container-type cloud instance usually doesn't either (see METHODOLOGY.md
§5.3). A rented VM-type instance, or a physical Linux desktop/workstation
the presenter administers directly, both work. --live is required to
actually lock clocks; without it this runs in a safe dry-run preview mode
(still live-charts, just never touches hardware).

Tested live on a rented RTX 4090: short phases are too short for the policy
to settle and give noisy, sometimes backwards-looking numbers -- a 15s-phase
test showed -7% ("savings" in the wrong direction), pure settling noise.
The default 45s phases settle to a stable, correctly-signed +10% on the
same checkpoint and workload condition this project's own hardware-
validated launch-bound result used (that result, over longer 90s phases,
measured +18.6% -- same direction, same order of magnitude). Don't shorten
--without-seconds/--with-seconds much below 45 for a demo anyone will
actually trust the number from.
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time

import pynvml
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_common = os.path.dirname(os.path.abspath(__file__))
if _common not in sys.path:
    sys.path.insert(0, _common)
from gpu_probe import hardware_limits  # noqa: E402

WITHOUT_SECONDS = 45
WITH_SECONDS = 45
SAMPLE_HZ = 2
HISTORY_SECONDS = 240

WITHOUT_COLOR = "#ff5f5f"   # warm red
WITH_COLOR = "#5fd787"      # fresh green
ACCENT = "#5fafff"
DIM = "#5c6370"


class BrailleCanvas:
    """Sub-pixel terminal graph, the technique btop/similar TUIs use: each
    character cell is a 2x4 dot grid via Unicode Braille Patterns (U+2800
    block), giving 2x horizontal and 4x vertical resolution versus one
    block-character per data point."""
    # dot bit values by (row 0-3, col 0-1) within one braille cell
    DOTS = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.px_w, self.px_h = cols * 2, rows * 4
        self.cells = [[0] * cols for _ in range(rows)]
        self.cell_color = [[None] * cols for _ in range(rows)]

    def set_pixel(self, x, y, color=None):
        if not (0 <= x < self.px_w and 0 <= y < self.px_h):
            return
        cx, cy = x // 2, y // 4
        bit = self.DOTS[y % 4][x % 2]
        self.cells[cy][cx] |= bit
        if color:
            self.cell_color[cy][cx] = color

    def draw_series(self, values, lo, hi, color):
        """values: one y-value per pixel-column (already resampled to
        self.px_w length). Draws a CONTINUOUS solid line: consecutive
        columns are connected with a vertical segment spanning both their
        y-positions, not just one pixel per column -- a single pixel per
        column looks like scattered dots whenever the value jumps between
        adjacent columns (which it does constantly with noisy power
        samples); connecting them is what makes it read as one solid line."""
        span = max(hi - lo, 1e-6)
        y_vals = [int((1.0 - (v - lo) / span) * (self.px_h - 1)) if v is not None else None
                  for v in values]
        prev_y = None
        for x, y in enumerate(y_vals):
            if y is None:
                prev_y = None
                continue
            if prev_y is not None:
                for yy in range(min(prev_y, y), max(prev_y, y) + 1):
                    self.set_pixel(x, yy, color)
            else:
                self.set_pixel(x, y, color)
            prev_y = y

    def render(self):
        lines = []
        for r in range(self.rows):
            text = Text()
            for c in range(self.cols):
                bits = self.cells[r][c]
                ch = chr(0x2800 + bits) if bits else " "
                color = self.cell_color[r][c] or DIM
                text.append(ch, style=color)
            lines.append(text)
        return lines


def resample(xs, ys, width):
    """Linear-interpolate (xs, ys) onto `width` evenly spaced columns
    spanning [xs[0], xs[-1]]. Returns a list of length `width` (None where
    there's no data yet, e.g. right after startup)."""
    if len(xs) < 2:
        return [ys[-1] if ys else None] * width
    x0, x1 = xs[0], xs[-1]
    span = max(x1 - x0, 1e-6)
    out = []
    j = 0
    for i in range(width):
        xt = x0 + span * i / (width - 1)
        while j < len(xs) - 2 and xs[j + 1] < xt:
            j += 1
        if xs[j + 1] == xs[j]:
            out.append(ys[j])
        else:
            t = (xt - xs[j]) / (xs[j + 1] - xs[j])
            out.append(ys[j] + t * (ys[j + 1] - ys[j]))
    return out


class PhaseStats:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def add(self, power_w):
        self.sum += power_w
        self.n += 1

    @property
    def mean(self):
        return self.sum / self.n if self.n else 0.0


def build_frame(console, name, hw, live_flag, current_phase, phase_remaining,
                 times, powers, phases, stats, current_power):
    width = max(40, console.width - 4)
    graph_cols = width - 2
    graph_rows = 14
    canvas = BrailleCanvas(graph_cols, graph_rows)

    if powers:
        lo, hi = min(powers), max(powers)
        if hi - lo < 1:
            hi = lo + 1
        # resample WITHOUT and WITH separately so each keeps its own colour,
        # with None where the other phase owns that time slice
        px_w = canvas.px_w
        rs_power = resample(times, powers, px_w)
        rs_phase_idx = resample(times, [0.0 if p == "WITHOUT" else 1.0 for p in phases], px_w)
        without_series = [v if idx is not None and idx < 0.5 else None
                           for v, idx in zip(rs_power, rs_phase_idx)]
        with_series = [v if idx is not None and idx >= 0.5 else None
                        for v, idx in zip(rs_power, rs_phase_idx)]
        canvas.draw_series(without_series, lo, hi, WITHOUT_COLOR)
        canvas.draw_series(with_series, lo, hi, WITH_COLOR)
        y_label_hi, y_label_lo = f"{hi:.0f}W", f"{lo:.0f}W"
    else:
        y_label_hi, y_label_lo = "", ""

    graph_lines = canvas.render()
    graph_group = []
    for i, line in enumerate(graph_lines):
        label = Text(f"{y_label_hi:>6} ", style=DIM) if i == 0 else \
            (Text(f"{y_label_lo:>6} ", style=DIM) if i == len(graph_lines) - 1 else Text("       "))
        graph_group.append(label + line)

    phase_color = WITHOUT_COLOR if current_phase == "WITHOUT" else WITH_COLOR
    legend = Text()
    legend.append("  ", style=WITHOUT_COLOR).append("●", style=WITHOUT_COLOR)
    legend.append(" WITHOUT (pinned max)    ", style=DIM)
    legend.append("●", style=WITH_COLOR)
    legend.append(" WITH (DDQN controller)", style=DIM)
    graph_group.append(Text(""))
    graph_group.append(legend)

    graph_panel = Panel(
        Group(*graph_group),
        title=f"[bold]power draw[/bold]  --  [{phase_color}]{current_phase}[/{phase_color}] "
              f"({phase_remaining:.0f}s left)",
        border_style=phase_color,
        padding=(0, 1),
    )

    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(justify="left", style=DIM)
    stats_table.add_column(justify="right")
    stats_table.add_row("current power", f"[bold]{current_power:6.1f} W[/bold]")
    stats_table.add_row(f"[{WITHOUT_COLOR}]WITHOUT mean[/{WITHOUT_COLOR}]",
                         f"{stats['WITHOUT'].mean:6.1f} W   (n={stats['WITHOUT'].n})")
    stats_table.add_row(f"[{WITH_COLOR}]WITH mean[/{WITH_COLOR}]   ",
                         f"{stats['WITH'].mean:6.1f} W   (n={stats['WITH'].n})")

    w_mean, with_mean = stats["WITHOUT"].mean, stats["WITH"].mean
    if w_mean > 0 and stats["WITH"].n > 0:
        saved = 100.0 * (w_mean - with_mean) / w_mean
        color = WITH_COLOR if saved > 0 else WITHOUT_COLOR
        bar_width = 30
        filled = max(0, min(bar_width, int(abs(saved) / 30 * bar_width)))
        bar = ("█" * filled).ljust(bar_width, "░")
        label_line = Text(f"  ENERGY SAVED SO FAR: {saved:+.1f}%", style=f"bold {color}")
        bar_line = Text("  ") + Text(bar, style=color)
    else:
        label_line = Text("  collecting baseline...", style=DIM)
        bar_line = Text("")

    stats_panel = Panel(
        Group(stats_table, Text(""), label_line, bar_line),
        title="[bold]live stats[/bold]",
        border_style=ACCENT,
        padding=(0, 1),
    )

    header = Align.center(
        Text.assemble(("  GPU DVFS CONTROLLER -- LIVE DEMO  ", f"bold {ACCENT}"),
                       (f"\n{name}", DIM),
                       (f"   {'[LIVE]' if live_flag else '[DRY RUN]'}",
                        f"bold {'red' if live_flag else 'yellow'}"))
    )

    footer = Align.center(Text("Ctrl+C to stop  --  clocks reset automatically", style=DIM))

    layout = Layout()
    layout.split_column(
        Layout(Panel(header, border_style=DIM), size=4),
        Layout(graph_panel, size=graph_rows + 4),
        Layout(stats_panel, size=9),
        Layout(footer, size=1),
    )
    return layout


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--live", action="store_true", help="actually lock clocks (default: dry-run preview)")
    ap.add_argument("--without-seconds", type=int, default=WITHOUT_SECONDS)
    ap.add_argument("--with-seconds", type=int, default=WITH_SECONDS)
    a = ap.parse_args()

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(a.gpu_index)
    hw = hardware_limits(handle)
    name = pynvml.nvmlDeviceGetName(handle)

    console = Console()
    times, powers, phases = [], [], []
    t0 = time.time()
    stats = {"WITHOUT": PhaseStats(), "WITH": PhaseStats()}
    deploy_proc = None
    current_phase = None
    phase_deadline = 0.0
    cleaned_up = False

    def cleanup(*_):
        nonlocal deploy_proc, cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        console.print("\n[dim]\\[live_demo] resetting clocks and exiting...[/dim]")
        if deploy_proc is not None:
            deploy_proc.send_signal(signal.SIGINT)
            try:
                deploy_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                deploy_proc.kill()
        try:
            pynvml.nvmlDeviceResetGpuLockedClocks(handle)
        except (AttributeError, pynvml.NVMLError):
            subprocess.run(["nvidia-smi", "-rgc"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    def start_without():
        nonlocal deploy_proc
        if deploy_proc is not None:
            deploy_proc.send_signal(signal.SIGINT)
            deploy_proc.wait(timeout=5)
            deploy_proc = None
        if a.live:
            try:
                pynvml.nvmlDeviceSetGpuLockedClocks(handle, int(hw["max_core_mhz"]), int(hw["max_core_mhz"]))
            except (AttributeError, pynvml.NVMLError):
                subprocess.run(["nvidia-smi", "-lgc",
                                f"{int(hw['max_core_mhz'])},{int(hw['max_core_mhz'])}"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start_with():
        nonlocal deploy_proc
        cmd = [sys.executable, "-u", os.path.join(_common, "deploy_live.py"),
               "--weights", a.weights, "--gpu-index", str(a.gpu_index)]
        if a.live:
            cmd.append("--live")
        deploy_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        with Live(console=console, refresh_per_second=SAMPLE_HZ, screen=True) as live:
            while True:
                now = time.time() - t0
                if current_phase is None or now >= phase_deadline:
                    current_phase = "WITHOUT" if current_phase != "WITHOUT" else "WITH"
                    phase_deadline = now + (a.without_seconds if current_phase == "WITHOUT" else a.with_seconds)
                    if current_phase == "WITHOUT":
                        start_without()
                    else:
                        start_with()

                power = float(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                times.append(now)
                powers.append(power)
                phases.append(current_phase)
                stats[current_phase].add(power)
                cutoff = now - HISTORY_SECONDS
                while times and times[0] < cutoff:
                    times.pop(0); powers.pop(0); phases.pop(0)

                frame = build_frame(console, name, hw, a.live, current_phase,
                                     phase_deadline - now, times, powers, phases, stats, power)
                live.update(frame)
                time.sleep(1.0 / SAMPLE_HZ)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
