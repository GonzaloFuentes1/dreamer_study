# Dreamer Study (v1-v4)

This repository contains educational implementations of the Dreamer algorithm family. The goal is to break down each version to understand the evolution of World Models in Reinforcement Learning.

## Project Status

| Version | Status | Description |
|:---:|:---:|---|
| **DreamerV1** | ✅ Ready | Continuous latent states and planning. |
| **DreamerV2** | ✅ Ready | Discrete latent states (Categorical). |
| **DreamerV3** | 🚧 In Progress | Scaling and normalization across domains. |
| **DreamerV4** | 🚧 In Progress | Future/Experimental. |

## Project Structure

- `dreamer_vX/`: Specific logic for each version.
- `common/`: Shared neural networks (RSSM, Encoder, Decoder), experience buffers.
- `envs/`: Wrappers for Gymnasium and other environments.
- `scripts/`: Entry points for training and visualization.

## Getting Started

Each version has its own `configs/` folder with YAML files. You can create new experiments there and call them with the `--exp` flag.

Examples included for v1:
- `cartpole`: Classic control (fast for testing).
- `mountain_car`: Classic control with sparse rewards.
- `acrobot`: Classic control with 6-dim state.
