# Dreamer Study

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red.svg)](https://pytorch.org/)

PyTorch implementations of the Dreamer family of world models for model-based reinforcement learning, written for study and comparison. The repository contains Dreamer V1, V2 and V3 sharing a common training script, replay buffers and environment wrappers.

## Status

| Version | Status |
|---------|--------|
| Dreamer V1 | Implemented and run on DMControl (`walker_walk`, `cheetah_run` configs). |
| Dreamer V2 | Implemented and run on DMControl (`walker_walk`, `cheetah_run` configs). |
| Dreamer V3 | Work in progress. The model and agent code is in `dreamer_v3/`, but it needs fixes and is not wired into `scripts/train.py` (selecting `--version v3` raises `NotImplementedError`). |

Dreamer V4 is not part of this repository. A separate implementation is available at https://github.com/GonzaloFuentes1/dreamer4.

This README reports no benchmark results. No training curves or scores are claimed here.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Version Comparison](#version-comparison)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [References](#references)
- [License](#license)

## Project Structure

```
dreamer_study/
├── dreamer_v1/              # Dreamer V1: Gaussian latent states
│   ├── agent.py             # Training algorithm
│   ├── models.py            # RSSM_V1, WorldModel, RewardModel, ValueModel, ActionModel
│   ├── networks.py          # ConvEncoder, ConvDecoder, MLP
│   ├── configs/             # walker_walk.yaml, cheetah_run.yaml, small.yaml
│   └── docs/                # ALGORITHM.md
├── dreamer_v2/              # Dreamer V2: 32x32 categorical latent states
│   ├── agent.py             # Training algorithm
│   ├── models.py            # RSSM_V2, WorldModel, RewardModel, DiscountModel, ActorModel, ValueModel
│   ├── networks.py          # ConvEncoder, ConvDecoder
│   ├── configs/             # walker_walk.yaml, cheetah_run.yaml, small.yaml
│   └── docs/                # ALGORITHM.md
├── dreamer_v3/              # Dreamer V3 (work in progress)
│   ├── agent.py             # DreamerV3Agent
│   ├── models.py            # RSSM_V3, WorldModel, RewardModel, ContinueModel, CriticModel, ActorModel
│   ├── networks.py          # Encoder, Decoder, MLP, RMSNorm, BlockLinear, GRUCell
│   ├── utils.py             # symlog/symexp, twohot, adaptive gradient clipping
│   ├── configs/             # walker_walk.yaml, cheetah_run.yaml, breakout.yaml
│   ├── docs/                # V3 status notes
│   └── README.md            # V3 implementation notes
├── common/                  # Shared modules (base agent, replay buffers, utilities)
├── envs/                    # Environment wrappers (DMControl, Gymnasium)
├── scripts/                 # train.py and analysis/plotting helpers
├── docs/                    # Documentation index
├── requirements.txt         # Pinned dependencies
├── CONTRIBUTING.md
└── LICENSE                  # Apache 2.0
```

## Installation

Python 3.10 or newer is required by the pinned dependencies. A CUDA GPU is recommended; `--gpu -1` runs on CPU.

```bash
git clone https://github.com/GonzaloFuentes1/dreamer_study.git
cd dreamer_study

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Dependency versions are pinned in `requirements.txt`. Note that `scripts/train.py` sets `MUJOCO_GL=egl` at startup, which targets Linux headless rendering.

Quick checks:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from dm_control import suite; print('DMControl OK')"
```

## Usage

### Training

Run from the repository root. `--version` selects `v1` or `v2`; `--exp` selects a YAML file in `dreamer_<version>/configs/`.

```bash
python scripts/train.py --version v1 --exp walker_walk --gpu 0
python scripts/train.py --version v2 --exp cheetah_run --gpu 0
python scripts/train.py --version v1 --exp small --gpu -1   # CPU
```

Arguments of `scripts/train.py`:

| Argument | Description |
|----------|-------------|
| `--version` | `v1` or `v2` (`v3` is not available yet). Default `v1`. |
| `--exp` | Config name without `.yaml`. |
| `--env` | Override the `env` field of the config, for example `dm_control/walker-walk-v0`. |
| `--gpu` | GPU id, or `-1` for CPU. Default `0`. |
| `--resume` | Path to a checkpoint (`.pt`) to resume from. |

Each run writes to `runs/<version>/<version>_<exp>_<timestamp>/` with `logs/` (TensorBoard events), `checkpoints/`, `videos/` and `metrics.jsonl`. Checkpoint frequency is set by `logging.save_interval` in the config.

### Configuration

Configs are YAML files. The fields below are taken from `dreamer_v1/configs/walker_walk.yaml`:

```yaml
env: "dm_control/walker-walk-v0"
action_repeat: 2
num_envs: 4

training:
  total_steps: 1000000
  batch_size: 50
  seq_length: 50

model:
  lr: 0.0006
  kl_free_nats: 3.0
  rssm:
    stoch_dim: 30
    deter_dim: 600

actor:
  lr: 0.00008
  horizon: 15

critic:
  lr: 0.00008
  gamma: 0.99
  lambda_: 0.95
```

Environment ids with the `dm_control/` prefix use the DMControl wrapper in `envs/wrappers.py`. Any other id is passed to `gymnasium.make`; only DMControl has been exercised with V1 and V2.

### Monitoring and analysis

```bash
tensorboard --logdir runs/

# Plot metrics from a run
python scripts/plot_metrics.py runs/v1/<run_id>/metrics.jsonl --out plots.png
```

`scripts/analyze_training.py` currently contains hardcoded run paths and has to be edited before use.

## Version Comparison

The table summarizes the implementations in this repository, as read from the code and configs.

| Aspect | V1 | V2 | V3 (work in progress) |
|--------|----|----|-----------------------|
| Latent state | Diagonal Gaussian (`stoch_dim` 30) | 32x32 categorical | 32x32 categorical |
| Reward head | Scalar, MSE | Scalar, MSE | Symexp twohot, 255 bins |
| Value head | Scalar, MSE | Scalar, MSE | Symexp twohot, 255 bins |
| Termination | Fixed discount | Learned discount predictor | Learned continue predictor |
| Decoder | Gaussian | Gaussian | Sigmoid output in [0, 1] |
| Normalization | None in the RSSM | LayerNorm in the RSSM | RMSNorm |
| MLP activation | ELU | ELU | SiLU |
| Optimizer | Adam | Adam | LaProp (falls back to Adam) with adaptive gradient clipping |
| Return normalization | No | No | EMA of return percentiles |

Absolute loss values are not comparable across versions, since the objectives are formulated differently (for example a Gaussian decoder log-likelihood in V1/V2 versus a sigmoid decoder with squared error in V3).

## Documentation

- [Documentation index](docs/README.md)
- [Dreamer V1 algorithm notes](dreamer_v1/docs/ALGORITHM.md)
- [Dreamer V2 algorithm notes](dreamer_v2/docs/ALGORITHM.md)
- [Dreamer V3 implementation notes](dreamer_v3/README.md)
- [Dreamer V3 status](dreamer_v3/docs/README.md)
- [Contributing guide](CONTRIBUTING.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Open areas:

- Fix the Dreamer V3 implementation and wire it into `scripts/train.py`
- Documentation and tests
- Additional environments
- Distributed training

## References

1. Hafner, D., Lillicrap, T., Ba, J., Norouzi, M. *Dream to Control: Learning Behaviors by Latent Imagination* (Dreamer V1), ICLR 2020. arXiv:[1912.01603](https://arxiv.org/abs/1912.01603)
2. Hafner, D., Lillicrap, T., Norouzi, M., Ba, J. *Mastering Atari with Discrete World Models* (Dreamer V2), ICLR 2021. arXiv:[2010.02193](https://arxiv.org/abs/2010.02193)
3. Hafner, D., Pasukonis, J., Ba, J., Lillicrap, T. *Mastering Diverse Domains through World Models* (Dreamer V3). arXiv:[2301.04104](https://arxiv.org/abs/2301.04104)
4. Hafner, D., Yan, W., Lillicrap, T. *Training Agents Inside of Scalable World Models* (Dreamer 4). arXiv:[2509.24527](https://arxiv.org/abs/2509.24527). Implementation in a separate repository: https://github.com/GonzaloFuentes1/dreamer4

Other resources: [DMControl](https://github.com/google-deepmind/dm_control), [Gymnasium](https://gymnasium.farama.org/).

## License

Apache License 2.0. See [LICENSE](LICENSE).
