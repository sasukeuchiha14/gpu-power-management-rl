import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def _resolve_data_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(gpu_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    candidates = [
        "rtx4500_8000_dataset.csv",
        "filtered_dataset.csv",
        "rtx4500_300_dataset.csv"
    ]
    input_file = None
    for cand in candidates:
        full_cand = os.path.join(data_dir, cand)
        if os.path.exists(full_cand):
            input_file = full_cand
            break
        elif os.path.exists(cand):
            input_file = cand
            break
            
    output_file = os.path.join(data_dir, "preprocessed_dataset.csv")
    return input_file, output_file

def preprocess_data():
    input_file, output_file = _resolve_data_paths()

    print(f"Loading raw data from {input_file}...")
    if not input_file or not os.path.exists(input_file):
        print(f"Error: Raw dataset not found in data directory. Ensure a raw dataset is present.")
        return
    df = pd.read_csv(input_file)

    # ==========================================
    # STEP 0: DROP DEAD WORKLOADS + DE-DUPLICATE REPEAT TRIALS
    # ==========================================
    # hotspot/heartwall/cupy_fft never actually ran on the GPU (0-2% util,
    # try/except:pass in the collector silently recorded the failures as data).
    # See dvfs-triage A2. Keeping them teaches the policy noise, not signal.
    DEAD_WORKLOADS = ["hotspot", "heartwall", "cupy_fft"]
    before = len(df)
    df = df[~df['Workload_Name'].isin(DEAD_WORKLOADS)].copy()
    print(f"0. Dropped dead workloads {DEAD_WORKLOADS}: {before} -> {len(df)} rows.")

    # Every (workload, core, mem) cell was profiled twice because the collector
    # appends on every run (triage A6). Treat the pair as repeat trials and
    # average them rather than silently double-counting or de-duplicating blind.
    group_keys = ['Workload_Name', 'Workload_Type', 'Core_F', 'Mem_F']
    avg_cols = ['Avg_Time_S', 'Avg_Pwr_W', 'Peak_T_C', 'Avg_Util_PCT', 'EDP']
    trial_counts = df.groupby(group_keys).size()
    print(f"   Repeat trials per (workload, core, mem) cell: min={trial_counts.min()}, "
          f"max={trial_counts.max()} (n={int(trial_counts.median())} typical).")
    n_rows_pre_avg = int(trial_counts.sum())
    df = df.groupby(group_keys, as_index=False)[avg_cols].mean()
    print(f"   Averaged repeat trials: {n_rows_pre_avg} rows collapsed to {len(df)} unique configs.")

    # ==========================================
    # STEP 1: ONE-HOT ENCODING
    # ==========================================
    print("1. Performing One-Hot Encoding on categorical variables...")
    # Convert Workload_Type to a string first so the math doesn't treat '0', '1', '2' as magnitudes
    df['Workload_Type'] = df['Workload_Type'].astype(str)
    
    # get_dummies turns categorical columns into multiple 0 and 1 columns
    # dtype=float ensures we get 1.0 and 0.0 instead of True/False (which PyTorch prefers)
    df = pd.get_dummies(df, columns=['Workload_Name', 'Workload_Type'], dtype=float)

    # ==========================================
    # STEP 2: MIN-MAX SCALING (NORMALIZATION)
    # ==========================================
    print("2. Normalizing numerical features to a [0, 1] range...")
    numeric_cols = ['Core_F', 'Mem_F', 'Avg_Time_S', 'Avg_Pwr_W', 'Peak_T_C', 'Avg_Util_PCT']
    
    feature_scaler = MinMaxScaler()
    df[numeric_cols] = feature_scaler.fit_transform(df[numeric_cols])

    # ==========================================
    # STEP 3: REWARD ENGINEERING (PER-WORKLOAD EDP)
    # ==========================================
    print("3. Engineering the RL Reward from EDP (scaled WITHIN each workload)...")
    # Why per-workload: a single global Min-Max scale is dominated by the heaviest
    # workload (e.g. pt_bert), which squashes light workloads (lud, myocyte, ...)
    # into a near-flat reward band, leaving the agent no gradient to learn from.
    # Scaling EDP within each workload gives EVERY workload a full 0..1 reward
    # range, so the best clock setting for that workload scores 1.0 and the worst
    # scores 0.0. Raw EDP is kept for logging and fair evaluation.
    wl_cols = [c for c in df.columns if c.startswith('Workload_Name_')]
    workload_label = df[wl_cols].idxmax(axis=1)

    def _scale_edp(e):
        span = e.max() - e.min()
        if span <= 0:                       # constant EDP -> scaled 0 -> reward 1.0
            return pd.Series(0.0, index=e.index)
        return (e - e.min()) / span

    df['Reward'] = 1.0 - df.groupby(workload_label)['EDP'].transform(_scale_edp)

    # ==========================================
    # STEP 4: EXPORT
    # ==========================================
    print(f"4. Saving pre-processed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n[OK] Pre-processing complete! The data is mathematically ready for the Deep Q-Network.")

if __name__ == "__main__":
    preprocess_data()