import math
import torch


def encode_time(time_step: torch.Tensor, dimension: int = 16) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError("dimension must be even")
    time = time_step.float().unsqueeze(1)
    frequency = torch.exp(
        torch.arange(0, dimension, 2, device=time.device).float()
        * (-math.log(10000.0) / dimension)
    )
    angles = time * frequency
    return torch.stack(
        (torch.sin(angles), torch.cos(angles)), dim=-1
    ).flatten(1)
