import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")
logs_plots_dir = os.path.join(gpu_root, "logs_and_plots")
os.makedirs(logs_plots_dir, exist_ok=True)

def load(path):
    t, p = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            p.append(float(row["power_W"]))
    return t, p

def stats(t, p):
    mean_p = sum(p) / len(p)
    energy = 0.0
    for (t1, p1), (t2, p2) in zip(zip(t, p), zip(t[1:], p[1:])):
        energy += 0.5 * (p1 + p2) * (t2 - t1)
    return mean_p, energy

def main():
    gov_path = os.path.join(data_dir, "power_governor.csv")
    ai_path = os.path.join(data_dir, "power_ai.csv")

    try:
        gt, gp = load(gov_path)
        at, ap = load(ai_path)
    except FileNotFoundError:
        print(f"Missing power files. Run measure_edp.py for both 'governor' and 'ai' first.")
        return

    gmean, genergy = stats(gt, gp)
    amean, aenergy = stats(at, ap)
    p_save = (gmean - amean) / gmean * 100 if gmean else 0
    e_save = (genergy - aenergy) / genergy * 100 if genergy else 0

    print("\n============  A/B COMPARISON (RTX 3050 Ti)  ============")
    print(f" Governor  : mean {gmean:5.1f} W | energy {genergy:8.1f} J")
    print(f" AI daemon : mean {amean:5.1f} W | energy {aenergy:8.1f} J")
    print(f" --> power saved  : {p_save:+.1f}%")
    print(f" --> energy saved : {e_save:+.1f}%")
    print("=========================================================")

    plt.figure(figsize=(11, 5))
    plt.plot(gt, gp, color="red", lw=1.5,
             label=f"Governor (max clocks)   mean {gmean:.1f} W")
    plt.plot(at, ap, color="green", lw=1.5,
             label=f"DDQN AI daemon           mean {amean:.1f} W")
    plt.axhline(gmean, color="red", ls="--", lw=1, alpha=0.6)
    plt.axhline(amean, color="green", ls="--", lw=1, alpha=0.6)
    plt.title(f"RTX 3050 Ti GPU Power: AI DVFS vs Max-Clock Governor | Energy Saved: {e_save:.1f}%")
    plt.xlabel("Time (s)")
    plt.ylabel("GPU Power (W) -- Lower is Better")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_img = os.path.join(logs_plots_dir, "edp_comparison.png")
    plt.savefig(out_img, dpi=200)
    print(f"Saved {out_img}")

if __name__ == "__main__":
    main()
