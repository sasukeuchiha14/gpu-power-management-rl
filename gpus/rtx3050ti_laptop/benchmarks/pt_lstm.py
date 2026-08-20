import torch
import time
import gc

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    # Recurrent workload -- the only non-feedforward benchmark in the set,
    # unlike every other trained workload here (all feedforward/parallel).
    # Scaled down for 4GB VRAM.
    model = torch.nn.LSTM(input_size=128, hidden_size=256, num_layers=2, batch_first=True).to(device)
    data = torch.randn(32, 100, 128, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(40):
            model(data)
    torch.cuda.synchronize()
    duration = time.time() - start
    print(f"Duration: {duration:.4f}")

    # Explicit cleanup
    del model, data
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    run()
