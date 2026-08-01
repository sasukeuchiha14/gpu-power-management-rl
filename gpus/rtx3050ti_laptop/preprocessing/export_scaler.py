import json
import os
import pandas as pd

NUMERIC_COLS = ["Core_F", "Mem_F", "Avg_Time_S", "Avg_Pwr_W", "Peak_T_C", "Avg_Util_PCT"]

def _resolve_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(gpu_root, "data")
    
    raw = os.path.join(data_dir, "rtx3050ti_dataset.csv")
    if not os.path.isfile(raw) and os.path.isfile("rtx3050ti_dataset.csv"):
        raw = "rtx3050ti_dataset.csv"
        
    prep = os.path.join(data_dir, "preprocessed_dataset.csv")
    if not os.path.isfile(prep) and os.path.isfile("preprocessed_dataset.csv"):
        prep = "preprocessed_dataset.csv"
        
    out_json = os.path.join(data_dir, "scaler_params.json")
    return raw, prep, out_json

def main():
    raw, prep, out_json = _resolve_paths()
    if not os.path.isfile(raw):
        raise FileNotFoundError("No raw dataset found (rtx3050ti_dataset.csv). Please run generate_dataset.py first.")
        
    print(f"Reading normalization ranges from: {raw}")
    df = pd.read_csv(raw)

    ranges = {c: {"min": float(df[c].min()), "max": float(df[c].max())} for c in NUMERIC_COLS}

    # If Mem_F is constant, force ranges so min != max to avoid division by zero during live norm
    if ranges["Mem_F"]["min"] == ranges["Mem_F"]["max"]:
        ranges["Mem_F"]["min"] = 0.0 # Force a base to prevent div by zero in normalizer

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
