import pandas as pd
import numpy as np
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")

INPUT_CSV = os.path.join(data_dir, "rtx3050ti_dataset.csv")
SCALER_JSON = os.path.join(data_dir, "scaler_params.json")
OUTPUT_CSV = os.path.join(data_dir, "preprocessed_dataset.csv")

def preprocess_data():
    if not os.path.exists(INPUT_CSV):
        raise FileNotFoundError(f"Input dataset not found at {INPUT_CSV}. Please run generate_dataset.py first.")

    # 1. Load Hardware Normalisation Limits
    if os.path.exists(SCALER_JSON):
        with open(SCALER_JSON) as f:
            sc = json.load(f)
            hw = sc.get("hardware_limits", {})
    else:
        hw = {}

    max_core = hw.get("max_core_mhz", 1900.0)
    power_limit = hw.get("power_limit_w", 70.0)
    slowdown_temp = hw.get("slowdown_temp_c", 87.0)
    total_vram = hw.get("total_vram_mb", 4096.0)

    print(f"Loading raw dataset from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # 2. Derive 6 Universal Device-Normalised Features (Page 6 of Technical Assessment)
    print("Engineering 6 Universal Device-Normalised Features...")
    df['core_frac'] = np.clip(df['Core_F'] / max_core, 0.0, 1.0)
    
    # Check if new column names exist or fallback to legacy column
    if 'Avg_GPU_Util' in df.columns:
        df['sm_util'] = np.clip(df['Avg_GPU_Util'] / 100.0, 0.0, 1.0)
    elif 'Avg_Util_PCT' in df.columns:
        df['sm_util'] = np.clip(df['Avg_Util_PCT'] / 100.0, 0.0, 1.0)
    else:
        df['sm_util'] = 0.5

    if 'Avg_Mem_Util' in df.columns:
        df['mem_util'] = np.clip(df['Avg_Mem_Util'] / 100.0, 0.0, 1.0)
    else:
        df['mem_util'] = 0.0 # Default if collecting legacy dataset

    df['power_frac'] = np.clip(df['Avg_Pwr_W'] / power_limit, 0.0, 1.0)
    df['temp_frac'] = np.clip(df['Peak_T_C'] / slowdown_temp, 0.0, 1.0)

    if 'VRAM_Used_MB' in df.columns:
        df['vram_frac'] = np.clip(df['VRAM_Used_MB'] / total_vram, 0.0, 1.0)
    else:
        df['vram_frac'] = 0.25 # Nominal proxy if legacy

    # 3. Calculate Per-Workload Scaled EDP Reward (Full 0..1 Gradient per Workload)
    print("Calculating Per-Workload Scaled Reward (1.0 = Optimal Efficiency)...")
    def _scale_edp(e):
        span = e.max() - e.min()
        if span <= 0:
            return pd.Series(0.0, index=e.index)
        return (e - e.min()) / span

    df['Reward'] = 1.0 - df.groupby('Workload_Name')['EDP'].transform(_scale_edp)

    # 4. Save preprocessed dataset
    out_cols = [
        'Workload_Name',
        'core_frac',
        'sm_util',
        'mem_util',
        'power_frac',
        'temp_frac',
        'vram_frac',
        'Core_F',
        'Mem_F',
        'Avg_Time_S',
        'Avg_Pwr_W',
        'Peak_T_C',
        'EDP',
        'Reward'
    ]
    df_out = df[[c for c in out_cols if c in df.columns]]
    df_out.to_csv(OUTPUT_CSV, index=False)
    
    print(f"Exported preprocessed dataset -> {OUTPUT_CSV}")
    print(f"  Rows: {len(df_out)} | Workloads: {df['Workload_Name'].nunique()}")
    print("  State vector: 6 Universal Features (core_frac, sm_util, mem_util, power_frac, temp_frac, vram_frac)")
    print("[OK] Pre-processing complete!")

if __name__ == "__main__":
    preprocess_data()
