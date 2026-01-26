"""
Dreamer V3: Mastering Diverse Domains through World Models

Pure PyTorch implementation following Hafner et al., 2024.
Paper: https://arxiv.org/abs/2301.04104

Key Features:
- Symexp twohot distributions (255 bins) for reward and value
- Return normalization with EMA for adaptive scaling
- Continue predictor (binary) instead of discount
- RMSNorm, BlockLinear, 5-layer MLPs with 640 units
- LaProp optimizer with adaptive gradient clipping

Structure:
- models.py: WorldModel, ActorModel, CriticModel, RSSM_V3, RewardModel, ContinueModel
- networks.py: Encoder, Decoder, MLP, RMSNorm, BlockLinear, GRUCell
- agent.py: DreamerV3Agent (training algorithm)
- utils.py: symlog/symexp, twohot encoding, AGC

Loss magnitudes: ~7 (vs ~11,000 in V1/V2 due to different formulations)
"""

from .agent import DreamerV3Agent
from .models import WorldModel, ActorModel, CriticModel

__version__ = "3.0.0"
__all__ = ["DreamerV3Agent", "WorldModel", "ActorModel", "CriticModel"]
