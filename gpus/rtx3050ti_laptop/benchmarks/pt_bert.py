import torch
import time
import gc
from transformers import BertModel, BertConfig

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("CUDA is not available!")

    # Clear memory cache
    gc.collect()
    torch.cuda.empty_cache()

    config = BertConfig()
    model = BertModel(config).to(device)
    model.eval()

    # Scaled down for 4GB VRAM
    data = torch.randint(0, 2000, (4, 512), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            model(data)

    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(30):
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
