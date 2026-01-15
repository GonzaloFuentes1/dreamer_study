import torch.nn as nn

class MLP(nn.Module):
    """Simple shared MLP block."""
    def __init__(self, input_dim, hidden_dim, output_dim, layers=2):
        super().__init__()
        # Common MLP implementation if needed across versions
        pass
