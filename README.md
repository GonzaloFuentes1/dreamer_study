# Dreamer Study: Complete Implementation of the Dreamer Family

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org/)

Comprehensive PyTorch implementations of the Dreamer world model family for model-based reinforcement learning. This repository includes implementations of Dreamer V1, V2, V3, and V4 variants.

**Implementation Status:**
- ✅ **Dreamer V1**: Complete and working
- ✅ **Dreamer V2**: Complete and working  
- 🚧 **Dreamer V3**: Implemented but needs fixes
- ❌ **Dreamer V4**: Not yet implemented

## Table of Contents

- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Version Comparison](#version-comparison)
- [Architectures](#architectures)
- [Results and Benchmarks](#results-and-benchmarks)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Key Features

- **4 Dreamer versions**: Implementations of V1, V2, V3, and V4
- **Pure PyTorch**: Clean and modular code without unnecessary dependencies
- **Reproducible**: Configurations and seeds for reproducing results
- **Optimized**: Mixed precision training (FP16), torch.compile, and CUDA optimizations
- **Flexible**: Support for continuous and discrete environments (DMControl, Gymnasium)
- **Complete evaluation**: Scripts for metrics analysis and visualization
- **Well documented**: Detailed READMEs and code comments

## Project Structure

```
dreamer_study/
├── dreamer_v1/              # ✅ Dreamer V1 (Hafner et al., 2020) - Complete
│   ├── agent.py             # Training algorithm
│   ├── models.py            # RSSM, RewardModel, ValueModel, ActionModel
│   ├── networks.py          # Encoder, Decoder, MLP
│   ├── configs/             # Environment configurations
│   │   ├── walker_walk.yaml
│   │   ├── cheetah_run.yaml
│   │   └── small.yaml
│   └── docs/                # V1-specific documentation
│
├── dreamer_v2/              # ✅ Dreamer V2 (Hafner et al., 2021) - Complete
│   ├── agent.py             # Training algorithm
│   ├── models.py            # RSSM_V2, RewardModel, DiscountModel, ActorModel
│   ├── networks.py          # Encoder, Decoder with LayerNorm
│   ├── configs/             # Environment configurations
│   └── docs/                # V2-specific documentation
│
├── dreamer_v3/              # 🚧 Dreamer V3 (Hafner et al., 2024) - Needs fixes
│   ├── agent.py             # Training algorithm
│   ├── models.py            # WorldModel, ActorModel, CriticModel, RSSM_V3
│   ├── networks.py          # Encoder, Decoder, MLP, RMSNorm, BlockLinear
│   ├── utils.py             # symlog/symexp, twohot, AGC
│   ├── configs/             # Environment configurations
│   ├── docs/                # V3-specific documentation
│   ├── README.md            # Detailed V3 documentation
│   └── 2301.04104v2.pdf     # Original paper
│
├── dreamer_v4/              # ❌ Dreamer V4 - Not implemented yet
│   └── __init__.py
│
├── common/                  # Shared modules
│   ├── agent.py             # Base Agent class
│   ├── buffer.py            # Standard ReplayBuffer
│   ├── buffer_parallel.py   # Parallel buffer for multiple envs
│   ├── prefetch_buffer.py   # Buffer with prefetching
│   ├── utils.py             # General utilities
│   └── debug_utils.py       # Debugging tools
│
├── envs/                    # Environment wrappers
│   └── wrappers.py          # Wrappers for DMControl and Gymnasium
│
├── scripts/                 # Training and analysis scripts
│   ├── train.py             # Main training script
│   ├── analyze_training.py  # Training metrics analysis
│   ├── plot_metrics.py      # Results visualization
│   ├── test_render_simple.py # Rendering test
│   ├── check_tags.py        # TensorBoard tags verification
│   └── run_benchmark.sh     # Automatic benchmark
│
├── docs/                    # General documentation
│   └── README.md            # Documentation index
├── runs/                    # TensorBoard logs (generated)
├── requirements.txt         # Project dependencies
├── .editorconfig           # Editor configuration
├── .gitignore              # Files ignored by git
├── CONTRIBUTING.md         # Contribution guidelines
├── LICENSE                 # Apache 2.0 License
└── README.md               # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (optional, for GPU)
- 8GB RAM minimum (16GB+ recommended)
- GPU with 6GB+ VRAM (optional but recommended)

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd dreamer_study

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

```
Core:
- torch>=2.10.0
- torchvision>=0.25.0
- numpy>=2.2.6

RL Environments:
- gymnasium>=1.2.3
- dm-control>=1.0.36
- mujoco>=3.4.0
- shimmy>=2.0.0

Optimization:
- pytorch_optimizer>=3.9.0 (LaProp for V3)

Visualization:
- tensorboard>=2.20.0
- matplotlib>=3.10.8
- opencv-python>=4.13.0
- moviepy>=2.2.1
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check DMControl
python -c "from dm_control import suite; print('DMControl OK')"
```

## Quick Start

### Basic Training

```bash
# Train Dreamer V1 on Walker Walk (recommended - working)
python scripts/train.py --version v1 --exp walker_walk --gpu 0

# Train Dreamer V2 on Cheetah Run (recommended - working)
python scripts/train.py --version v2 --exp cheetah_run --gpu 0

# Train Dreamer V3 on Walker Walk (needs fixes)
python scripts/train.py --version v3 --exp walker_walk --gpu 0

# Train on CPU (slower)
python scripts/train.py --version v1 --exp small --gpu -1
```

### Training Script Arguments

```bash
python scripts/train.py \
  --version v1              # Version: v1, v2, v3
  --exp walker_walk         # Experiment name (.yaml file)
  --env walker-walk         # Override environment (optional)
  --gpu 0                   # GPU to use (0-7, or -1 for CPU)
  --resume path/to/ckpt.pt  # Resume from checkpoint (optional)
```

### Monitoring with TensorBoard

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Open in browser: http://localhost:6006
```

### Evaluation

```bash
# Analyze training metrics
python scripts/analyze_training.py --run_dir runs/v1_walker_walk_<timestamp>

# Plot specific metrics
python scripts/plot_metrics.py --run_dir runs/v1_walker_walk_<timestamp> \
                               --metrics reward episode_length
```

## Training

### Experiment Configuration

Configuration files are in YAML format in `dreamer_*/configs/`:

```yaml
# Example: dreamer_v1/configs/walker_walk.yaml
env: walker-walk
action_repeat: 2
seed: 42

model:
  rssm:
    deter_dim: 200
    stoch_dim: 30
  num_units: 400
  lr: 6e-4

actor:
  lr: 8e-5
  
critic:
  lr: 8e-5

training:
  batch_size: 50
  seq_len: 50
  prefill_steps: 5000
  total_steps: 1000000
  imagination_horizon: 15
  mixed_precision: true
```

### Supported Environments

#### DMControl Suite
```
walker-walk, walker-run, walker-stand
cheetah-run, cheetah-walk
hopper-hop, hopper-stand
humanoid-walk, humanoid-run, humanoid-stand
quadruped-walk, quadruped-run
reacher-easy, reacher-hard
cartpole-balance, cartpole-swingup
```

#### Gymnasium
```
Classic Control: CartPole-v1, MountainCar-v0, Pendulum-v1
Box2D: LunarLander-v2, BipedalWalker-v3
Atari: (requires additional configuration)
```

### Best Practices

1. **Start with V1 or V2**: These versions are complete and working
2. **Use GPU**: Training is 10-50x faster
3. **Mixed Precision**: Enabled by default, reduces memory usage
4. **Batch Size**: 16-50 for GPU with 8-12GB VRAM
5. **Checkpoints**: Automatically saved every 50K steps
6. **Logs**: TensorBoard logs in `runs/`

## Version Comparison

| Feature | V1 (2020) ✅ | V2 (2021) ✅ | V3 (2024) 🚧 | V4 ❌ |
|----------------|-----------|-----------|--------------|-------------------|
| **Status** | Complete | Complete | Needs fixes | Not implemented |
| **Latent States** | Gaussian | 32×32 Categorical | 32×32 Categorical | - |
| **Reward Pred** | MSE | MSE | Symexp twohot (255 bins) | - |
| **Value Pred** | Scalar MSE | Scalar MSE | Symexp twohot (255 bins) | - |
| **Continue/Discount** | Fixed γ | Discount predictor | Binary continue c_t | - |
| **Decoder** | Gaussian | Gaussian | Sigmoid [0,1] | - |
| **MLP Layers** | 2-3 layers | 4 layers | 5 layers | - |
| **Units per Layer** | 200-400 | 400 | 640 | - |
| **Normalization** | None | LayerNorm | RMSNorm | - |
| **Optimizer** | Adam | Adam | LaProp + AGC | - |
| **Return Norm** | No | No | EMA percentiles | - |
| **Loss Magnitude** | ~11,800 | ~11,800 | ~7 | - |

### Version Recommendations

**When to use each version:**
- **V1 or V2**: For production use and new projects (both complete and working)
- **V3**: For experimental work (needs fixes before production use)
- **V4**: Not available yet

## Architectures

### RSSM (Recurrent State Space Model)

All versions use variants of the RSSM:

```
Observation → Encoder → Representation
                            ↓
         Deterministic ← Dynamics → Stochastic
              h_t               z_t
                ↓                 ↓
            Decoder         Reward/Value
```

**V1 RSSM**: Gaussian stochastic states
```python
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  # GRU
z_t ~ N(μ(h_t, o_t), σ(h_t, o_t))   # Gaussian
```

**V2/V3 RSSM**: 32×32 categorical states
```python
h_t = f(h_{t-1}, z_{t-1}, a_{t-1})      # GRU
z_t ~ Categorical(32×32)(h_t, o_t)      # 32 classes × 32 variables
```

### Main Componentde 2 → ReLU
Conv 4×4, 64 filters, stride 2 → ReLU  
Conv 4×4, 128 filters, stride 2 → ReLU
Conv 4×4, 256 filters, stride 2 → ReLU
Flatten → Linear → (1024,)
```

**Decoder (Transposed CNN)**
```
Linear: latent → (256, 4, 4)
TransConv 5×5, 128 filters, stride 2 → ReLU
TransConv 5×5, 64 filters, stride 2 → ReLU
TransConv 6×6, 32 filters, stride 2 → ReLU
TransConv 6×6, 3 filters, stride 2 → Sigmoid (V3) / Gaussian (V1/V2)
```

**MLP (V3)**
```
5 layers × 640 units
RMSNorm + SiLU activation
Block-diagonal linear for efficiency
```

## Documentation

### Documentation Structure

- [General Documentation](docs/README.md) - Documentation index and guides
- [Dreamer V1 Documentation](dreamer_v1/docs/) - V1-specific docs
- [Dreamer V2 Documentation](dreamer_v2/docs/) - V2-specific docs
- [Dreamer V3 Documentation](dreamer_v3/docs/) - V3-specific docs
- [Dreamer V3 README](dreamer_v3/README.md) - Detailed V3 implementation notes
- [Original V3 Paper](dreamer_v3/2301.04104v2.pdf) - Mastering Diverse Domains through World Models
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

### Loss Magnitude Explanation

**Why is V3 loss (~7) different from V1/V2 (~11,800)?**

Different loss formulations:
- **V1/V2**: Gaussian decoder `-log p(x|z)` includes constants → ~11,800
- **V3**: Sigmoid decoder MSE + categorical cross-entropy → ~7

**Both are correct!** Same gradients, different scales. Do not compare magnitudes directly between versions.

### Debugging

```bash
# Detailed outputs
python scripts/train.py --version v1 --exp walker_walk --debug

# Quick component test
python -c "from dreamer_v1.agent import DreamerV1Agent; print('V1 OK')"

# Check buffers
python common/debug_utils.py
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution

- [ ] Fix Dreamer V3 implementation
- [ ] Populate V3 docs/ folder with detailed documentation
- [ ] Complete Dreamer V4 implementation
- [ ] More environments (Atari, MuJoCo, etc.)
- [ ] Distributed training
- [ ] World model visualization
- [ ] Documentation improvements

## References

### Original Papers

**Dreamer V1** (2020)
```bibtex
@article{hafner2020dream,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={ICLR},
  year={2020}
}
```

**Dreamer V2** (2021)
```bibtex
@article{hafner2021mastering,
  title={Mastering Atari with Discrete World Models},
  author={Hafner, Danijar and Lillicrap, Timothy and Norouzi, Mohammad and Ba, Jimmy},
  journal={ICLR},
  year={2021}
}
```

**Dreamer V3** (2024)
```bibtex
@article{hafner2024dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2024}
}Additional Resources

- [Dreamer Official Website](https://danijar.com/project/dreamerv3/)
- [DMControl Documentation](https://github.com/deepmind/dm_control)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

This project is licensed under Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Authors and Acknowledgments

- **Original Implementation**: Based on papers by Danijar Hafner et al.
- **Contributors**: See full list on GitHub
- **Special thanks**: To the RL community and original authors

## Contact

For questions, issues, or collaborations, please open an issue on GitHub.

---

**Last updated**: January 2026  
**Project status**: Active development  
**Recommended version**: Dreamer V1 or V2 (complete and working)
**Última actualización**: Enero 2026  
**Estado del proyecto**: Activo en desarrollo  
**Versión recomendada**: Dreamer V3

## Citations

```bibtex
@article{hafner2024dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2024}
}
```
