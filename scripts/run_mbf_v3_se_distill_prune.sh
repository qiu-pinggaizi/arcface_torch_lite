#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Training script: mbf_v3_se + Knowledge Distillation + Pruning (0.2)
# Teacher: mbf_large (no prune)
# Student: mbf_v3_se (prune_ratio=0.2)
# 使用 GPU 4-7 (GPU 0-3 被 Phase 3 占用)

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port 29526 --nproc_per_node=4 train_v2_distill.py \
    --config configs/glint360k_mbf_v3_se_distill_prune.py \
    --prune_ratio 0.2
