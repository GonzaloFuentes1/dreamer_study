# Documentation

This directory holds the documentation index for the Dreamer Study repository. Each version keeps its own notes next to its code.

## Contents

- [Main README](../README.md): status, installation, usage and version comparison
- [Contributing guide](../CONTRIBUTING.md)

## Per-version documentation

- [Dreamer V1 algorithm notes](../dreamer_v1/docs/ALGORITHM.md)
- [Dreamer V2 algorithm notes](../dreamer_v2/docs/ALGORITHM.md)
- [Dreamer V3 implementation notes](../dreamer_v3/README.md)
- [Dreamer V3 status](../dreamer_v3/docs/README.md)

## Code entry points

- [Common modules](../common/): base agent, replay buffers, utilities
- [Environment wrappers](../envs/): DMControl and Gymnasium wrappers
- [Training scripts](../scripts/): `train.py` and analysis helpers

## Papers

The original papers are not stored in this repository. They are available on arXiv:

- Dreamer V1: https://arxiv.org/abs/1912.01603
- Dreamer V2: https://arxiv.org/abs/2010.02193
- Dreamer V3: https://arxiv.org/abs/2301.04104
- Dreamer 4: https://arxiv.org/abs/2509.24527 (implemented in a separate repository, https://github.com/GonzaloFuentes1/dreamer4)
