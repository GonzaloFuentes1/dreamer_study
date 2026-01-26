#!/bin/bash
# Script para entrenar Dreamer V1 en CPU

echo "🖥️  Entrenando Dreamer V1 en CPU..."
echo ""

# Forzar CPU
export CUDA_VISIBLE_DEVICES=-1
export MUJOCO_GL=egl

# Ejecutar training
python scripts/train.py --version v1 --exp walker_walk --gpu -1

echo ""
echo "✅ Training completado"
