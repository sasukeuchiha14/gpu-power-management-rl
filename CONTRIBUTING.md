# Contributing to GPU DVFS RL Controller

Thank you for your interest in contributing to the **Deep Reinforcement Learning Dynamic Voltage and Frequency Scaling (DVFS) GPU Power Management Framework**!

## Guidelines & Best Practices

1. **Hardware Safety First**
   - When introducing new clock locking mechanisms or telemetry sampling code, ensure a hard thermal safety override is always active (`TEMP_LIMIT_C = 80°C` or `85°C`).
   - Always run the live controller in `DRY_RUN = True` mode first before enabling hardware clock actuation (`DRY_RUN = False`).

2. **Cross-Platform Compatibility**
   - Use `os.path.join()` or relative path helpers for all file path resolutions to maintain compatibility across Windows and Linux environments.
   - Do not introduce Linux-only C++ build dependencies without providing pure-Python or platform-independent fallbacks.

3. **Submitting Changes**
   - Fork the repository and create a feature branch (`git checkout -b feature/awesome-feature`).
   - Run baseline evaluation scripts (`python evaluation/evaluate_baselines.py`) to verify model convergence and stability.
   - Ensure clean code formatting and document any changes to state dimensions or action spaces.

4. **Reporting Issues**
   - Include details on your target GPU model, NVIDIA driver version, OS (Windows 11 / Ubuntu), PyTorch CUDA version, and error tracebacks.
