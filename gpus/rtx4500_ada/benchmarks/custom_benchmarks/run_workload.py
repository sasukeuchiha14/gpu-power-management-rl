import torch
import torch.nn as nn
import time
import sys
import gc

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_workload.py <workload_name>")
        sys.exit(1)
        
    workload = sys.argv[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("Error: CUDA is not available! Workload must run on GPU.")
        sys.exit(1)

    print(f"Running workload '{workload}' on {torch.cuda.get_device_name(0)}")

    # Clear cache before starting
    gc.collect()
    torch.cuda.empty_cache()

    if workload == 'pt_resnet50':
        # Category 0: Compute-Bound
        # Inference using ResNet-50
        from torchvision.models import resnet50
        model = resnet50().to(device)
        model.eval()
        data = torch.randn(16, 3, 224, 224, device=device) # Scaled batch size to 16 (fits in 4GB VRAM)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                model(data)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            for _ in range(30):
                model(data)
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_bert':
        # Category 0: Compute-Bound
        # Inference using BERT
        from transformers import BertModel, BertConfig
        config = BertConfig()
        model = BertModel(config).to(device)
        model.eval()
        # Scaled batch size to 4 (fits in 4GB VRAM)
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
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_matmul':
        # Category 1: Memory-Bound (Large matrix multiplications, bandwidth-bound if arithmetic intensity is kept moderate)
        # We perform large single-precision matrix multiplications
        size = 4096
        A = torch.randn(size, size, device=device)
        B = torch.randn(size, size, device=device)
        
        # Warmup
        for _ in range(3):
            torch.matmul(A, B)
            
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(25):
            torch.matmul(A, B)
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_fft':
        # Category 1: Memory-Bound
        # 3D Fast Fourier Transform
        # Size: 256x256x256 complex64 tensor (16 million elements, ~128MB VRAM)
        size = 256
        A = torch.randn(size, size, size, dtype=torch.complex64, device=device)
        
        # Warmup
        for _ in range(2):
            torch.fft.fftn(A)
            
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(15):
            torch.fft.fftn(A)
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_vector_add':
        # Category 1: Extreme Memory-Bound (Zero computation, purely VRAM memory bandwidth copy/add)
        # Size: 64 million float elements (~256MB VRAM per tensor)
        num_elements = 64 * 1024 * 1024
        A = torch.randn(num_elements, device=device)
        B = torch.randn(num_elements, device=device)
        
        # Warmup
        for _ in range(3):
            C = A + B
            
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(80):
            C = A + B
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_resnet50_train':
        # Category 2: Mixed (Stresses core compute during forward pass and VRAM bandwidth during backward weight updates)
        from torchvision.models import resnet50
        model = resnet50().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        data = torch.randn(16, 3, 224, 224, device=device)
        targets = torch.randint(0, 1000, (16,), device=device)
        
        # Warmup
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(15):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")

    elif workload == 'pt_crypto_proxy':
        # Category 2: Mixed / Integer stress
        # Generate large integer vectors on GPU and perform bitwise XOR, shift, and modulo arithmetic
        num_elements = 30 * 1024 * 1024 # 30M integers
        A = torch.randint(0, 2**30, (num_elements,), dtype=torch.int64, device=device)
        
        # Warmup
        for _ in range(3):
            A = (A ^ (A << 3)) % 1000003
            
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(40):
            A = (A ^ (A << 3)) % 1000003
        torch.cuda.synchronize()
        print(f"Duration: {time.time() - start:.4f} seconds")
        
    else:
        print(f"Unknown workload: {workload}")
        sys.exit(1)

    # Clean up before exit
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
