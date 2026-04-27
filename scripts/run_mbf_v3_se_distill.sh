#!/bin/bash

# Navigate to project root (relative to this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Training script for mbf_v3_se + Knowledge Distillation (Phase 3)
# Teacher: mbf_large
# Student: mbf_v3_se

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29524 --nproc_per_node=4 train_v2_distill.py \
    --config configs/glint360k_mbf_v3_se_distill.py
