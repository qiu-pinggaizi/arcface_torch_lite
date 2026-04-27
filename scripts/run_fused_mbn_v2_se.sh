#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Training script for FusedMobileNetV2SE
# 轻量级单通道模型，带 SE 模块

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29527 --nproc_per_node=4 train_v2_prune.py \
    --config configs/glint360k_fused_mbn_v2_se.py
