import torch
from typing import Tuple

def apply_noise(value: torch.Tensor, noise_range: Tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """
    Apply uniform noise within a specified range to the given tensor.

    Args:
        value (torch.Tensor): Input tensor to apply noise.
        noise_range (Tuple[float, float]): Range for uniform noise (default is (0.9, 1.1)).

    Returns:
        torch.Tensor: Tensor with noise applied.
    """
    noise = torch.empty_like(value).uniform_(*noise_range)
    return value * noise