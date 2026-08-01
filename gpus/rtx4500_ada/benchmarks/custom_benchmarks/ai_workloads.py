import torch
import time
import sys
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, BertConfig

device = torch.device('cuda')
workload = sys.argv[1]

if workload == 'resnet':
    model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    data = torch.randn(64, 3, 224, 224, device=device)
    for _ in range(50): model(data)
    
elif workload == 'bert':
    config = BertConfig()
    model = BertModel(config).to(device)
    data = torch.randint(0, 2000, (32, 512), device=device)
    for _ in range(100): model(data)
