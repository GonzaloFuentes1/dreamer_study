# Dreamer Study (v1-v4)

This repository contains educational implementations of the Dreamer algorithm family. The goal is to break down each version to understand the evolution of World Models in Reinforcement Learning.

## Educational Roadmap

1. **DreamerV1**: Continuous latent states and planning.
2. **DreamerV2**: Introduction of discrete latent states (Categorical) for better stability in environments like Atari.
3. **DreamerV3**: Scaling and normalization to work across diverse domains with the same hyperparameters.
4. **DreamerV4**: (Future/Experimental)

## Project Structure

- `dreamer_vX/`: Specific logic for each version.
- `common/`: Shared neural networks (RSSM, Encoder, Decoder), experience buffers.
- `envs/`: Wrappers for Gymnasium and other environments.
- `scripts/`: Entry points for training and visualization.

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run a specific experiment (e.g., DreamerV1 on CartPole):
   ```bash
   python -m dreamer_study.scripts.train --version v1 --exp cartpole
   ```

## Experiments and Configuration

Each version has its own `configs/` folder with YAML files. You can create new experiments there and call them with the `--exp` flag.

Examples included for v1:
- `cartpole`: Classic control (fast for testing).
- `mountain_car`: Classic control with sparse rewards.
- `acrobot`: Classic control with 6-dim state.