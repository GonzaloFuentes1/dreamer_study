# Dreamer Study: V1, V2, V3 Implementation

Comprehensive PyTorch implementations of the Dreamer world model family for model-based reinforcement learning.

## Project Structure

```
dreamer_study/
├── dreamer_v1/          # Dreamer V1 (Hafner et al., 2020)
│   ├── agent.py         # Training algorithm
│   ├── models.py        # RSSM, RewardModel, ValueModel, ActionModel
│   ├── networks.py      # Encoder, Decoder, MLP
│   └── configs/         # Environment configs
│
├── dreamer_v2/          # Dreamer V2 (Hafner et al., 2021)
│   ├── agent.py         # Training algorithm
│   ├── models.py        # RSSM_V2, RewardModel, DiscountModel, ActorModel, ValueModel
│   ├── networks.py      # Encoder, Decoder
│   └── configs/         # Environment configs
│
├── dreamer_v3/          # Dreamer V3 (Hafner et al., 2024) ⭐ Latest
│   ├── agent.py         # Training algorithm
│   ├── models.py        # WorldModel, ActorModel, CriticModel, RSSM_V3
│   ├── networks.py      # Encoder, Decoder, MLP, RMSNorm, BlockLinear
│   ├── utils.py         # symlog/symexp, twohot, AGC
│   ├── configs/         # Environment configs
│   └── README.md        # V3-specific documentation
│
├── scripts/             # Training and evaluation scripts
├── quick_test_all.py    # Quick validation test
└── README.md            # This file
```

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchvision numpy pyyaml dm_control pytorch-optimizer
```

### Run Tests

```bash
source venv/bin/activate
python quick_test_all.py
```

Expected output:
```
V1: ✓ PASSED (wm_loss: ~11,800)
V2: ✓ PASSED (wm_loss: ~11,800)
V3: ✓ PASSED (wm_loss: ~7)
```

### Training

```bash
python scripts/train.py --version v3 --exp walker_walk
```

## Key Differences Between Versions

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| **Latent States** | Gaussian | 32×32 Categorical | 32×32 Categorical |
| **Reward Pred** | MSE | MSE | Symexp twohot (255 bins) |
| **Value Pred** | Scalar MSE | Scalar MSE | Symexp twohot (255 bins) |
| **Continue** | Fixed γ | Discount predictor | Binary continue |
| **Decoder** | Gaussian | Gaussian | Sigmoid [0,1] |
| **Architecture** | 2-3 layers | 4 layers | 5 layers × 640 units |
| **Normalization** | None | LayerNorm | RMSNorm |
| **Optimizer** | Adam | Adam | LaProp + AGC |
| **Return Norm** | None | None | EMA with percentiles |
| **Loss Magnitude** | ~11,800 | ~11,800 | ~7 |

## V3 Highlights (Latest)

✅ **Symexp twohot**: 255 exponentially spaced bins for reward/value  
✅ **Return normalization**: Adaptive EMA scaling across domains  
✅ **Continue predictor**: Binary c_t ∈ {0,1} instead of discount  
✅ **Architecture**: RMSNorm, BlockLinear, deeper networks  
✅ **Robustness**: Fixed hyperparameters work across diverse domains  

See [dreamer_v3/README.md](dreamer_v3/README.md) for detailed documentation.

## Loss Magnitude Explanation

**Why is V3 loss (~7) different from V1/V2 (~11,800)?**

Different loss formulations:
- **V1/V2**: Gaussian decoder `-log p(x|z)` includes constants → ~11,800
- **V3**: Sigmoid MSE + categorical cross-entropy → ~7

Both are correct! Same gradients, different scales.

## Citations

```bibtex
@article{hafner2024dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2024}
}
```
