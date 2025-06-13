"""
Contains various utility functions for PyTorch model training and saving.
"""

from pathlib import Path

import torch


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """Saves a PyTorch model to a target directory.

    Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
    save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

# Updated scale calculation
def calculate_scale(weights, scale_factor=1):
    abs_max = max(weights.max().item(), abs(weights.min().item()))
    scale = abs_max / scale_factor  # Can change '1' for Adjusting to better match [-1, 0, 1] range, 
    return scale

def ternary_quantize(weights, scale, zero_point=0):
    """
    Applies ternary quantization to the given weights using scale and zero_point.
    Maps values to {-1, 0, 1} based on calculated quantized values.

    Args:
        weights: The model's weights (torch.Tensor).
        scale: Scale value for quantization.
        zero_point: Zero point for quantization.

    Returns:
        Quantized weights (torch.Tensor).
    """
    # Quantize the weights to integer values and directly map to -1, 0, 1
    ternary_weights = torch.round((weights - zero_point) / scale).clamp(-1, 1)  # Clamp ensures values stay within -1, 0, 1
    return ternary_weights

def custom_quantize(weights, levels, scale, zero_point=0):
    """
    Quantize weights to nearest value among a custom set of levels after scaling.

    Args:
        weights (torch.Tensor): Input tensor to quantize.
        levels (list or torch.Tensor): List of quantization levels (e.g., [0, 0.5, 0.7, 0.8, 1]).
        scale (float): Scale factor used for normalization.
        zero_point (float): Zero point used for normalization (default 0.0).

    Returns:
        torch.Tensor: Quantized tensor mapped back to original range.
    """
    levels = torch.tensor(levels, dtype=weights.dtype, device=weights.device)

    # Normalize weights
    normalized_weights = (weights - zero_point) / scale

    # Expand for broadcasting
    weights_expanded = normalized_weights.unsqueeze(-1)  # shape: (..., 1)
    levels_expanded = levels.view(*([1] * weights.dim()), -1)  # shape: (..., L)

    # Find nearest level
    diffs = torch.abs(weights_expanded - levels_expanded)
    indices = torch.argmin(diffs, dim=-1)

    # Map back to original range
    quantized_normalized = levels[indices]
    # quantized_weights = quantized_normalized * scale + zero_point
    return quantized_normalized
