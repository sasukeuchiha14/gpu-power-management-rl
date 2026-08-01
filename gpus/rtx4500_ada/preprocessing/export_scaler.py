"""
export_scaler.py
Dumps the exact Min-Max normalization ranges + feature order the DDQN was
trained on, so the live daemon can normalize NVML readings identically.

Run this ONCE (on any machine) after preprocessing:
    python3 export_scaler.py

It writes scaler_params.json. For perfect fidelity the min/max should come from
the SAME file preprocess.py fit on (filtered_dataset.csv). If that file is gone,
we fall back to the raw profiling dataset; the frequency grid (Core_F / Mem_F)
is fixed by hardware so those ranges are exact regardless.
"""
import json
import os
import pandas as pd

NUMERIC_COLS = ["Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT"]

def _resolve_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(gpu_root, "data")
    
    candidates = ["filtered_dataset.csv", "rtx4500_8000_dataset.csv", "rtx4500_300_dataset.csv"]
    raw = None
    for c in candidates:
        full_path = os.path.join(data_dir, c)
        if os.path.isfile(full_path):
            raw = full_path
            break
        elif os.path.isfile(c):
            raw = c
            break
            
    prep = os.path.join(data_dir, "preprocessed_dataset.csv")
    if not os.path.isfile(prep) and os.path.isfile("preprocessed_dataset.csv"):
        prep = "preprocessed_dataset.csv"
        
    out_json = os.path.join(data_dir, "scaler_params.json")
    return raw, prep, out_json

def main():
    raw, prep, out_json = _resolve_paths()
    if raw is None:
        raise FileNotFoundError("No raw dataset found for scaler extraction.")
    print(f"Reading normalization ranges from: {raw}")
    df = pd.read_csv(raw)

    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in NUMERIC_COLS}

    # Feature order MUST match the network's input. Read it from the processed file.
    order = None
    if os.path.isfile(prep):
        cols = pd.read_csv(prep, nrows=1).columns.tolist()
        order = [c for c in cols if c not in ("EDP", "Reward")]

    out = {"numeric_cols": NUMERIC_COLS, "ranges": ranges, "state_col_order": order}
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Wrote {out_json}")
    for c in NUMERIC_COLS:
        print(f"  {c:<14} min={ranges[c]['min']:.3f}  max={ranges[c]['max']:.3f}")
    if order:
        print(f"  state vector length = {len(order)} features")

if __name__ == "__main__":
    main()
