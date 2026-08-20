import json
import os
import pynvml

script_dir = os.path.dirname(os.path.abspath(__file__))
gpu_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(gpu_root, "data")
os.makedirs(data_dir, exist_ok=True)

OUTPUT_JSON = os.path.join(data_dir, "scaler_params.json")

def get_device_constants(index=0):
    """Query real hardware limits from NVML for universal device-normalisation."""
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        
        # Max core clock
        max_core = float(pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM))
        # Power limit (in Watts)
        try:
            power_limit = float(pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
        except pynvml.NVMLError:
            power_limit = 70.0 # Default 70W TGP for RTX 3050 Ti Laptop
            
        # Thermal slowdown threshold
        try:
            temp_threshold = float(pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
        except pynvml.NVMLError:
            temp_threshold = 87.0 # Default NVIDIA slowdown temp
            
        # Total VRAM (in MB)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_vram = float(mem_info.total / (1024.0 * 1024.0))
        
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"NVML hardware query notice ({e}), using default RTX 3050 Ti hardware profile.")
        max_core = 1900.0
        power_limit = 70.0
        temp_threshold = 87.0
        total_vram = 4096.0

    return {
        "max_core_mhz": max_core,
        "power_limit_w": power_limit,
        "slowdown_temp_c": temp_threshold,
        "total_vram_mb": total_vram
    }

def main():
    hw = get_device_constants()
    
    # 6 Universal, Device-Normalised Features (Fixed across all GPUs)
    features = [
        "core_frac",   # Core_F / max_core_mhz
        "sm_util",     # Avg_GPU_Util / 100.0
        "mem_util",    # Avg_Mem_Util / 100.0
        "power_frac",  # Avg_Pwr_W / power_limit_w
        "temp_frac",   # Peak_T_C / slowdown_temp_c
        "vram_frac"    # VRAM_Used_MB / total_vram_mb
    ]
    
    scaler_payload = {
        "features": features,
        "feature_dim": len(features),
        "hardware_limits": hw
    }
    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(scaler_payload, f, indent=2)
        
    print("=========================================================")
    print(f" Exported Universal Scaler Parameters -> {OUTPUT_JSON}")
    print("=========================================================")
    print(f"  max_core_mhz    : {hw['max_core_mhz']:.1f} MHz")
    print(f"  power_limit_w   : {hw['power_limit_w']:.1f} W")
    print(f"  slowdown_temp_c : {hw['slowdown_temp_c']:.1f} C")
    print(f"  total_vram_mb   : {hw['total_vram_mb']:.1f} MB")
    print(f"  feature_count   : {len(features)} universal features")
    print("=========================================================")

if __name__ == "__main__":
    main()
