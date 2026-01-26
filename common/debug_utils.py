"""
Debug utilities for Dreamer training.
Helps diagnose issues like:
- Infinite gradient norms (expected with mixed precision)
- NaN values in losses or gradients
- Memory issues

Based on DreamerV2 official debugging guide.
"""

import torch
import numpy as np


def check_tensor_stats(name, tensor, check_grad=False):
    """Check if tensor has NaN or Inf values and print statistics."""
    if tensor is None:
        print(f"{name}: None")
        return
    
    if isinstance(tensor, (list, tuple)):
        for i, t in enumerate(tensor):
            check_tensor_stats(f"{name}[{i}]", t, check_grad)
        return
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    stats = {
        'shape': tuple(tensor.shape),
        'dtype': tensor.dtype,
        'device': tensor.device,
        'min': tensor.min().item() if tensor.numel() > 0 else None,
        'max': tensor.max().item() if tensor.numel() > 0 else None,
        'mean': tensor.mean().item() if tensor.numel() > 0 else None,
        'has_nan': has_nan,
        'has_inf': has_inf,
    }
    
    if check_grad and tensor.requires_grad and tensor.grad is not None:
        grad = tensor.grad
        stats['grad_min'] = grad.min().item()
        stats['grad_max'] = grad.max().item()
        stats['grad_mean'] = grad.mean().item()
        stats['grad_has_nan'] = torch.isnan(grad).any().item()
        stats['grad_has_inf'] = torch.isinf(grad).any().item()
    
    status = "⚠️ " if has_nan or has_inf else "✓ "
    print(f"{status}{name}: {stats}")
    
    return has_nan or has_inf


def check_model_parameters(model, name="model"):
    """Check all parameters in a model for NaN/Inf."""
    print(f"\n=== Checking {name} parameters ===")
    has_issues = False
    for param_name, param in model.named_parameters():
        if check_tensor_stats(f"{name}.{param_name}", param, check_grad=True):
            has_issues = True
    return has_issues


def gradient_norm(parameters):
    """Compute total gradient norm (like in the official implementation)."""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def print_gradient_norms(models_dict):
    """Print gradient norms for all models.
    
    Args:
        models_dict: Dict of {name: model} pairs
    """
    print("\n=== Gradient Norms ===")
    for name, model in models_dict.items():
        norm = gradient_norm(model.parameters())
        if np.isinf(norm):
            print(f"⚠️  {name}: {norm:.2e} (INFINITE - expected with mixed precision)")
        elif np.isnan(norm):
            print(f"❌ {name}: {norm:.2e} (NaN - THIS IS BAD)")
        else:
            print(f"✓  {name}: {norm:.2e}")


def memory_summary():
    """Print GPU memory usage."""
    if torch.cuda.is_available():
        print("\n=== GPU Memory ===")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            print(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


def enable_anomaly_detection():
    """Enable PyTorch anomaly detection for debugging NaN/Inf.
    
    WARNING: This is SLOW. Only use for debugging.
    """
    torch.autograd.set_detect_anomaly(True)
    print("⚠️  Anomaly detection enabled (SLOW - for debugging only)")


def disable_anomaly_detection():
    """Disable anomaly detection."""
    torch.autograd.set_detect_anomaly(False)
    print("✓ Anomaly detection disabled")
