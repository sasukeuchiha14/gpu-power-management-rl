import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def _resolve_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
    data_dir = os.path.join(gpu_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    input_file = os.path.join(data_dir, "rtx3050ti_dataset.csv")
    if not os.path.exists(input_file) and os.path.exists("rtx3050ti_dataset.csv"):
        input_file = "rtx3050ti_dataset.csv"
        
    output_file = os.path.join(data_dir, "preprocessed_dataset.csv")
    return input_file, output_file

def preprocess_data():
    input_file, output_file = _resolve_paths()

    print(f"Loading raw data from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Ensure you have run generate_dataset.py first.")
        return

    df = pd.read_csv(input_file)

    # ==========================================
    # STEP 1: ONE-HOT ENCODING
    # ==========================================
    print("1. Performing One-Hot Encoding on categorical variables...")
    # Convert Workload_Type to string so the math doesn't treat '0', '1', '2' as magnitudes
    df['Workload_Type'] = df['Workload_Type'].astype(str)
    
    # get_dummies turns categorical columns into multiple 0 and 1 columns
    # dtype=float ensures we get 1.0 and 0.0 instead of True/False
    df = pd.get_dummies(df, columns=['Workload_Name', 'Workload_Type'], dtype=float)

    # Ensure all one-hot columns are present (even if some workloads were skipped)
    # This maintains the exact feature shape for the network
    expected_workloads = [
        'Workload_Name_pt_resnet50', 'Workload_Name_pt_bert',
        'Workload_Name_pt_matmul', 'Workload_Name_pt_fft',
        'Workload_Name_pt_vector_add', 'Workload_Name_pt_resnet50_train',
        'Workload_Name_pt_crypto_proxy'
    ]
    for col in expected_workloads:
        if col not in df.columns:
            df[col] = 0.0

    expected_types = ['Workload_Type_0', 'Workload_Type_1', 'Workload_Type_2']
    for col in expected_types:
        if col not in df.columns:
            df[col] = 0.0

    # ==========================================
    # STEP 2: MIN-MAX SCALING (NORMALIZATION)
    # ==========================================
    print("2. Normalizing numerical features to a [0, 1] range...")
    numeric_cols = ['Core_F', 'Mem_F', 'Avg_Time_S', 'Avg_Pwr_W', 'Peak_T_C', 'Avg_Util_PCT']
    
    # MinMaxScaler handles constant columns (like Mem_F which is always 5871) by mapping them to 0.0
    feature_scaler = MinMaxScaler()
    df[numeric_cols] = feature_scaler.fit_transform(df[numeric_cols])

    # ==========================================
    # STEP 3: REWARD ENGINEERING (PER-WORKLOAD EDP)
    # ==========================================
    print("3. Engineering the RL Reward from EDP (scaled WITHIN each workload)...")
    wl_cols = [c for c in df.columns if c.startswith('Workload_Name_')]
    workload_label = df[wl_cols].idxmax(axis=1)

    def _scale_edp(e):
        span = e.max() - e.min()
        if span <= 0: # constant EDP -> scaled 0 -> reward 1.0
            return pd.Series(0.0, index=e.index)
        return (e - e.min()) / span

    df['Reward'] = 1.0 - df.groupby(workload_label)['EDP'].transform(_scale_edp)

    # ==========================================
    # STEP 4: EXPORT
    # ==========================================
    print(f"4. Saving pre-processed dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n[OK] Pre-processing complete! Ready for training.")

if __name__ == "__main__":
    preprocess_data()
