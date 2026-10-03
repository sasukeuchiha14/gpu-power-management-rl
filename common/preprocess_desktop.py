#!/usr/bin/env python3
"""
preprocess_desktop.py -- converts every gpus/<slug>/data/<slug>_dataset.csv
produced by collect_dataset.py into a preprocessed_dataset.csv carrying the
6-feature schema (common/features.py), so common/unified_env.py's
discover_datasets() can find and train on it.

Only touches the collect_dataset.py-produced desktop trees (nvidia_geforce_*
folders). Does not touch gpus/rtx3050ti_laptop or gpus/rtx4500_ada -- those
have their own preprocessing/preprocess.py scripts with card-specific logic
and are left alone.

    python3 preprocess_desktop.py            # process every known desktop GPU
    python3 preprocess_desktop.py --gpu nvidia_geforce_rtx_3090   # just one

WHY hardware limits are pinned here instead of auto-matched from
fleet_survey.csv: that file can carry MULTIPLE rows for the same GPU model
across different rental instances / driver versions, with genuinely
different max_core_mhz (e.g. RTX 3060 shows 2160 / 2100 / 2145 MHz across
three drivers). Naively matching "first row with this gpu_name" would
silently use the wrong normalisation ceiling for some cards -- a real bug
class, not hypothetical: RTX 3060's own committed dataset only cross-checks
correctly against the driver-580.95.05 row (max Core_F actually recorded in
its data is 2145.0, matching that row exactly and no other). So each
dataset here is pinned to the exact (slug, driver) pair that actually
produced it, verified against session records, rather than guessed.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
from features import FEATURES, build_features  # noqa: E402


def _slugify(name):
    # Kept identical to collect_dataset.py's _slugify -- not imported from
    # there because that module imports pynvml at load time (real GPU
    # access), which this CSV-only script has no reason to require.
    s = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return s or "gpu"

_repo_root = os.path.abspath(os.path.join(_script_dir, ".."))
_fleet_survey_csv = os.path.join(_repo_root, "common", "fleet_survey", "fleet_survey.csv")

# slug -> driver version that actually produced that slug's committed dataset.
# See module docstring for why this can't be auto-matched from fleet_survey.csv.
DRIVER_PIN = {
    # 3060 re-collected 3 Oct 2026 on a different physical rental unit that happens to
    # report the SAME driver string (580.95.05) as the original one, but a genuinely
    # different factory boost ceiling (2190 MHz vs the original unit's 2145 MHz) -- i.e.
    # (gpu_name, driver) is not actually a unique key across physical units, only across
    # driver versions. Disambiguated with a synthetic suffix on the survey row itself
    # rather than letting two same-keyed rows collide silently in _load_hardware_limits().
    "nvidia_geforce_rtx_3060":     "580.95.05-unitB",  # cross-checked: max Core_F in data == 2190.0
    "nvidia_geforce_rtx_3070":     "580.95.05",
    # 3080/4090 re-collected 3 Oct 2026 on a fresh rental running 575.51.03, not the
    # original 580.95.05 instance -- the old dataset rows were replaced, not merged, so
    # the pin must follow. Verified the ceiling actually differs for 4090 (3105 -> 3135
    # MHz across drivers; 3080's is coincidentally identical at 2100) -- this is a real
    # case of the exact bug class this module's docstring warns about, not a formality.
    "nvidia_geforce_rtx_3080":     "575.51.03",
    # 3090 re-collected 3 Oct 2026: same driver string (580.95.05) and same core ceiling
    # (2130 MHz) as the original unit, but a different power_limit_w (350 vs 390 W) --
    # power_frac = power_w / hw["power_limit_w"] in features.py, so this would have
    # silently mis-normalised every power reading by ~11% if left pinned to the old row.
    # The existing sanity check in preprocess_one() only validates core clock, not power
    # limit, so this one would NOT have been caught automatically -- found by manually
    # diffing the fresh probe survey against the committed row before trusting it.
    "nvidia_geforce_rtx_3090":     "580.95.05-unitB",
    "nvidia_geforce_rtx_4070":     "580.95.05",
    "nvidia_geforce_rtx_4070_ti":  "575.51.03",
    "nvidia_geforce_rtx_4090":     "575.51.03",
}

EXCLUDE_SLUGS = {"rtx3050ti_laptop", "rtx4500_ada"}  # have their own preprocess.py, not ours to touch


def _load_hardware_limits():
    """{(slug, driver): hardware_limits dict} for every fleet_survey.csv row."""
    survey = pd.read_csv(_fleet_survey_csv)
    out = {}
    for _, row in survey.iterrows():
        slug = _slugify(row["gpu_name"])
        driver = str(row["driver"])
        out[(slug, driver)] = json.loads(row["hardware_limits"])
    return out


def _discover_raw_datasets():
    """{slug: csv_path} for every gpus/<slug>/data/<slug>_dataset.csv, excluding
    the two legacy per-card trees that have their own preprocessing already."""
    pattern = os.path.join(_repo_root, "gpus", "*", "data", "*_dataset.csv")
    found = {}
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        if not fname.endswith("_dataset.csv") or "quicktest" in fname:
            continue
        slug = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if slug in EXCLUDE_SLUGS:
            continue
        # collect_dataset.py always names the file "<slug>_dataset.csv" --
        # guard against picking up an unrelated same-directory csv.
        if fname != f"{slug}_dataset.csv":
            continue
        found[slug] = path
    return found


def _scale_edp_per_workload(edp_series):
    span = edp_series.max() - edp_series.min()
    if span <= 0:
        return pd.Series(0.0, index=edp_series.index)
    return (edp_series - edp_series.min()) / span


def preprocess_one(slug, csv_path, hw_limits_by_key, strict=True):
    if slug not in DRIVER_PIN:
        raise ValueError(
            f"'{slug}' has no entry in DRIVER_PIN -- add one (see module docstring) "
            f"before preprocessing it, so hardware limits aren't silently guessed."
        )
    key = (slug, DRIVER_PIN[slug])
    if key not in hw_limits_by_key:
        raise ValueError(
            f"No fleet_survey.csv row found for {key}. Available drivers for this "
            f"slug: {sorted(d for (s, d) in hw_limits_by_key if s == slug)}"
        )
    hw = hw_limits_by_key[key]

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} is empty.")

    required_raw = ["Core_F", "Avg_GPU_Util", "Avg_Mem_Util", "Avg_Pwr_W",
                     "Peak_T_C", "VRAM_Used_MB", "Workload_Name", "Avg_Time_S", "EDP"]
    missing = [c for c in required_raw if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    # Sanity check: for a dataset with zero skipped points, the highest
    # requested core level should equal the true max_core_mhz exactly (the
    # collector always includes the hardware max as one of its swept
    # targets -- see _build_core_sweep). A big gap here would mean the
    # pinned hardware limits don't actually match this dataset.
    observed_max_core = df["Core_F"].max()
    if observed_max_core > hw["max_core_mhz"] * 1.02:
        msg = (f"[{slug}] observed max Core_F ({observed_max_core}) exceeds pinned "
               f"max_core_mhz ({hw['max_core_mhz']}) by >2% -- DRIVER_PIN is probably "
               f"wrong for this slug.")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)

    # Sanity check: observed power can't physically exceed the true power cap
    # by more than measurement noise. Catches a pinned power_limit_w that's
    # too LOW for this dataset (found 3 Oct 2026: the RTX 3090 re-collection
    # used a rental with a genuinely lower power_limit_w than the original
    # pinned row -- same driver string, same core ceiling, different power
    # cap). NOTE: this only catches the "pin too low" direction -- a pin
    # that's too HIGH produces no physically-impossible value, just silently
    # compressed power_frac features, and isn't caught by any automated check
    # here. That direction must still be caught by diffing a fresh gpu_probe.py
    # survey against the committed fleet_survey.csv row before trusting it.
    observed_max_pwr = df["Avg_Pwr_W"].max()
    if observed_max_pwr > hw["power_limit_w"] * 1.05:
        msg = (f"[{slug}] observed max Avg_Pwr_W ({observed_max_pwr:.1f}) exceeds pinned "
               f"power_limit_w ({hw['power_limit_w']}) by >5% -- DRIVER_PIN is probably "
               f"wrong for this slug.")
        if strict:
            raise ValueError(msg)
        print("WARNING:", msg)

    feat_matrix = [
        build_features(
            core_mhz=row.Core_F, gpu_util=row.Avg_GPU_Util, mem_util=row.Avg_Mem_Util,
            power_w=row.Avg_Pwr_W, temp_c=row.Peak_T_C, vram_used_mb=row.VRAM_Used_MB,
            hw=hw,
        )
        for row in df.itertuples()
    ]
    feat_df = pd.DataFrame(feat_matrix, columns=FEATURES, index=df.index)
    df = pd.concat([df, feat_df], axis=1)

    df["Reward"] = 1.0 - df.groupby("Workload_Name")["EDP"].transform(_scale_edp_per_workload)

    out_cols = (["Workload_Name"] + FEATURES +
                ["Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "EDP", "Reward"])
    df_out = df[out_cols]

    if df_out.isna().any().any():
        bad_cols = df_out.columns[df_out.isna().any()].tolist()
        raise ValueError(f"{csv_path}: NaN values produced in columns {bad_cols} -- aborting, not writing a corrupt file.")

    out_path = os.path.join(os.path.dirname(csv_path), "preprocessed_dataset.csv")
    df_out.to_csv(out_path, index=False)
    return out_path, len(df_out), df["Workload_Name"].nunique(), (df["Reward"].min(), df["Reward"].max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default=None, help="only process this one slug (default: all discovered)")
    ap.add_argument("--no-strict", action="store_true",
                     help="warn instead of aborting on a hardware-limit sanity-check mismatch")
    args = ap.parse_args()

    hw_limits_by_key = _load_hardware_limits()
    raw = _discover_raw_datasets()
    if args.gpu:
        if args.gpu not in raw:
            sys.exit(f"'{args.gpu}' not found among discovered raw datasets: {sorted(raw.keys())}")
        raw = {args.gpu: raw[args.gpu]}

    if not raw:
        sys.exit("No raw desktop datasets found under gpus/*/data/*_dataset.csv.")

    print(f"Processing {len(raw)} dataset(s): {sorted(raw.keys())}")
    results = []
    for slug, csv_path in sorted(raw.items()):
        out_path, n_rows, n_workloads, (rmin, rmax) = preprocess_one(
            slug, csv_path, hw_limits_by_key, strict=not args.no_strict)
        print(f"  [OK] {slug}: {n_rows} rows, {n_workloads} workloads, "
              f"reward range [{rmin:.4f}, {rmax:.4f}] -> {out_path}")
        results.append((slug, n_rows))

    print(f"\nDone. {len(results)} dataset(s) preprocessed.")


if __name__ == "__main__":
    main()
