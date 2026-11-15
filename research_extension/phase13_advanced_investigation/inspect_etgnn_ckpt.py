import torch
from pathlib import Path

ckpt = torch.load("results/checkpoints/etgnn.pt", map_location="cpu", weights_only=False)
print("Type:", type(ckpt))
if isinstance(ckpt, dict):
    print("Keys:", ckpt.keys())
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    for k, v in state.items():
        print(f"  {k}: {v.shape}")

