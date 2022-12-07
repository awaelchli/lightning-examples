import torch

def gpu_utilization(device: torch.device) -> float:
    return torch.cuda.utilization() if device.type == "cuda" else 0
