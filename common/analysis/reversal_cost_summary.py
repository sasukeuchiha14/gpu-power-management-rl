"""
reversal_cost_summary.py -- aggregates the six reversal_cost_validate.py
runs (0.05/0.10/0.20/0.30/0.40/0.60, each full 5-GPU/10-seed) into the
summary table and paired comparison reported in context.md / NORTHSTAR.md
row 11's 7 Oct 2026 entry. Does not train anything -- just parses the
already-saved CSVs.

    python common/analysis/reversal_cost_summary.py
"""
import os
import pandas as pd

from _common import REPO  # noqa: F401

VALUES = ["005", "010", "020", "030", "040", "060"]


def main():
    dfs = {}
    for tag in VALUES:
        d = pd.read_csv(f"{REPO}/docs/results/reversal_cost_10seed_{tag}.csv")
        d["cycle_rate"] = d.n_cycling / d.n_total
        dfs[tag] = d.set_index(["gpu", "seed"])

    print(f"{'reversal_cost':<14} {'cycle_rate':>10} {'ss_mean':>10} {'ss_std':>10}")
    for tag, d in dfs.items():
        v = f"{int(tag) / 100:.2f}"
        print(f"{v:<14} {d.cycle_rate.mean():>10.4f} {d.ss.mean():>10.4f} {d.ss.std():>10.4f}")

    print("\n=== paired win-rate vs reversal_cost=0.30 (lower ss wins) ===")
    base = dfs["030"]
    for tag, d in dfs.items():
        if tag == "030":
            continue
        common = d.index.intersection(base.index)
        wins = (d.loc[common, "ss"].values < base.loc[common, "ss"].values).sum()
        print(f"{int(tag)/100:.2f} beats 0.30 on ss: {wins}/{len(common)}")

    print("\n=== paired win-rate vs reversal_cost=0.30 (lower cycle_rate wins) ===")
    for tag, d in dfs.items():
        if tag == "030":
            continue
        common = d.index.intersection(base.index)
        wins = (d.loc[common, "cycle_rate"].values < base.loc[common, "cycle_rate"].values).sum()
        ties = (d.loc[common, "cycle_rate"].values == base.loc[common, "cycle_rate"].values).sum()
        print(f"{int(tag)/100:.2f} beats 0.30 on cycle_rate: {wins}/{len(common)} (ties: {ties})")


if __name__ == "__main__":
    main()
