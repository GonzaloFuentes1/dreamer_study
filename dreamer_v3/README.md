# Dreamer V3 Implementation

Pure PyTorch implementation of [Mastering Diverse Domains through World Models (Hafner et al., 2024)](https://arxiv.org/abs/2301.04104).

## Architecture

This implementation follows the paper specifications exactly and maintains the same clean structure as V1/V2.

### Core Components

**models.py**
- `RSSM_V3`: Recurrent State Space Model with 32×32 categorical states
- `RewardModel`: Categorical reward predictor with symexp twohot (255 bins)
- `ContinueModel`: Episode continuation predictor (logistic regression)
- `ActorModel`: Policy network with return normalization
- `CriticModel`: Value function with symexp twohot (255 bins)
- `WorldModel`: High-level wrapper containing all world model components

**networks.py**
- `RMSNorm`: Root Mean Square normalization
- `BlockLinear`: Block-diagonal linear layers for efficiency
- `GRUCell`: Custom GRU implementation
- `MLP`: Multi-layer perceptron (5 layers, 640 units)
- `Encoder`: CNN encoder with bottleneck
- `Decoder`: Sigmoid decoder [0,1]

**agent.py**
- `DreamerV3Agent`: Main training algorithm

**utils.py**
- `symlog/symexp`: Symmetric logarithm transformations
- `twohot_encode/twohot_loss`: Two-hot encoding for categorical distributions
- `adaptive_gradient_clip`: AGC for stable training
- `create_symlog_bins`: Exponentially spaced bins

## Paper Implementations

### World Model (Equation 1)
- ✅ RSSM with 32×32 categorical states
- ✅ Encoder: CNN → embedding
- ✅ Decoder: Sigmoid output [0,1] with MSE loss
- ✅ Reward: Symexp twohot (255 bins, equation 10-11)
- ✅ Continue: Logistic regression c_t ∈ {0,1}

### Loss Functions (Equation 2-3)
- ✅ βpred=1, βdyn=1, βrep=0.1
- ✅ Free bits: 1 nat ≈ 1.44 bits
- ✅ KL balancing between dynamics and representation

### Robust Predictions (Equations 8-11)
- ✅ Symlog squared error for reconstruction
- ✅ Symexp twohot for reward and value (255 bins)
- ✅ Exponentially spaced bins: symexp(-20...+20)
- ✅ Zero initialization for reward/critic outputs

### Actor Learning (Equations 6-7)
- ✅ Return normalization with EMA
- ✅ S = EMA[Per(R^λ_t,95) - Per(R^λ_t,5), 0.99]
- ✅ Entropy scale η = 3×10⁻⁴
- ✅ Max(1, S) denominator to avoid noise amplification

### Critic Learning (Equation 5)
- ✅ Categorical distribution with 255 bins
- ✅ λ-returns with γ=0.997, λ=0.95
- ✅ Slow critic with EMA decay=0.98

### Optimization
- ✅ LaProp optimizer (eps=1e-20)
- ✅ Adaptive gradient clipping (AGC, clip=0.3)
- ✅ Horizon T=16 for imagination

## Key Differences from V1/V2

1. **Symexp twohot distributions** instead of Gaussian/MSE
   - Reward: categorical 255 bins vs scalar MSE
   - Critic: categorical 255 bins vs scalar value

2. **Continue predictor** instead of discount
   - Binary c_t ∈ {0,1} vs continuous discount

3. **Return normalization with EMA**
   - Adaptive scaling across domains
   - Robust to sparse/dense rewards

4. **Architecture improvements**
   - RMSNorm instead of LayerNorm
   - BlockLinear for efficiency
   - SiLU activations
   - 5 layers, 640 units (vs 2-4 layers in V1/V2)

## Usage

```python
from dreamer_v3.agent import DreamerV3Agent

agent = DreamerV3Agent(
    config=config,
    obs_shape=(3, 64, 64),
    action_dim=6,
    is_discrete=False,
    device='cuda'
)

# Training step
losses = agent.train_step(obs, actions, rewards, dones)

# Policy
action, next_state, env_action = agent.policy(obs, state, last_action, mode='train')
```

## Loss Magnitudes

V3 losses are ~1000× smaller than V1/V2 due to different loss functions:
- V1/V2: Gaussian decoder with -log_prob (includes constants) → ~11,000
- V3: Sigmoid decoder with MSE + twohot losses → ~7

Both are correct - different formulations, same gradients.

## Citation

```bibtex
@article{hafner2024dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2024}
}
```
