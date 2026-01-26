"""
Utility functions for DreamerV3.

Includes:
- Symlog transformations for value normalization
- TwoHot encoding for distributional regression
- Adaptive Gradient Clipping (AGC)
"""

import torch
import torch.nn.functional as F


def symlog(x):
    """Symlog transformation: sign(x) * log(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    """Inverse of symlog: sign(x) * (exp(|x|) - 1)"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def create_symlog_bins(num_bins=255, device='cuda'):
    """Create symlog-spaced bins for twohot encoding.
    
    Args:
        num_bins: Number of discrete bins (default: 255)
        device: Device to create bins on
        
    Returns:
        Tensor of shape (num_bins,) with symlog-spaced bin centers
    """
    bins = torch.linspace(-20, 20, num_bins, device=device)
    return symexp(bins)


def twohot_encode(x, bins):
    """Encode scalar values as two-hot vectors for distributional regression.
    
    Places probability mass on the two nearest bins, interpolating between them.
    
    Args:
        x: Tensor of scalar values to encode, shape (...)
        bins: Tensor of bin centers, shape (num_bins,)
        
    Returns:
        Two-hot encoded tensor, shape (..., num_bins)
    """
    x = x.unsqueeze(-1)  # Add bin dimension: (..., 1)
    
    # Find indices of bins below and above x
    below = (bins <= x).sum(dim=-1, dtype=torch.int64) - 1
    below = torch.clamp(below, 0, len(bins) - 2)
    above = below + 1
    
    # Get bin values
    below_val = bins[below]
    above_val = bins[above]
    
    # Interpolate weights
    weight_above = torch.clamp((x.squeeze(-1) - below_val) / (above_val - below_val + 1e-8), 0, 1)
    weight_below = 1 - weight_above
    
    # Create two-hot target
    target = torch.zeros((*x.shape[:-1], len(bins)), device=x.device)
    target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    
    return target


def twohot_loss(pred_logits, target_values, bins):
    """Compute cross-entropy loss between predicted logits and two-hot encoded targets.
    
    Args:
        pred_logits: Predicted logits, shape (..., num_bins)
        target_values: Target scalar values, shape (...)
        bins: Bin centers for two-hot encoding, shape (num_bins,)
        
    Returns:
        Scalar loss (mean cross-entropy)
    """
    target_twohot = twohot_encode(target_values, bins)
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return -(target_twohot * log_probs).sum(dim=-1).mean()


def adaptive_gradient_clip(parameters, clip=0.3, pmin=1e-3):
    """Adaptive Gradient Clipping (AGC) as used in DreamerV3.
    
    Clips gradients based on parameter norms rather than global gradient norm.
    For each parameter:
        clip_value = clip * max(pmin, ||param||)
        if ||grad|| > clip_value:
            grad = grad * (clip_value / ||grad||)
    
    Args:
        parameters: Model parameters with gradients
        clip: Clipping factor (default: 0.3 as in paper)
        pmin: Minimum parameter norm (default: 1e-3)
    
    Returns:
        dict with statistics: avg_norm, max_norm, num_clipped
    """
    if clip <= 0:
        return {'avg_norm': 0.0, 'max_norm': 0.0, 'num_clipped': 0}
    
    grad_norms = []
    num_clipped = 0
    
    for param in parameters:
        if param.grad is None:
            continue
        
        # Flatten for norm computation
        grad_flat = param.grad.detach().flatten()
        param_flat = param.detach().flatten()
        
        # Compute norms
        grad_norm = torch.linalg.norm(grad_flat, ord=2)
        param_norm = torch.linalg.norm(param_flat, ord=2)
        
        grad_norms.append(grad_norm.item())
        
        # Compute adaptive clip threshold
        max_norm = clip * torch.maximum(param_norm, torch.tensor(pmin, device=param.device))
        
        # Clip if necessary
        if grad_norm > max_norm:
            param.grad.mul_(max_norm / grad_norm)
            num_clipped += 1
    
    return {
        'avg_norm': sum(grad_norms) / max(len(grad_norms), 1),
        'max_norm': max(grad_norms) if grad_norms else 0.0,
        'num_clipped': num_clipped
    }
